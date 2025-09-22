import csv
from pathlib import Path
import os
import time
import multiprocessing as mp
import matplotlib
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from queue import Empty

matplotlib.use("Agg")

OUTPUT_DIR = "../results/task01/"

# Global variables for multiprocessing worker initialization
# Clustering computation workers
_CLUSTER_A = None  # will hold neighbor_sets: dict(idx -> set(neighbor_idx))
_CLUSTER_NODES = None
_CLUSTER_N = None
_CLUSTER_PROG_Q = None
_CLUSTER_WORKER_IDX = None

# Common neighbors computation workers
_GLOBAL_A = None  # will hold neighbor_sets: dict(idx -> set(neighbor_idx))
_GLOBAL_NODES = None
_GLOBAL_NODE_TO_IDX = None
_GLOBAL_N = None
_GLOBAL_PROG_Q = None
_GLOBAL_WORKER_IDX = None


def load_data(input_path):
    path = Path(input_path)
    name = path.name
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]

    # Handle protein network format (string IDs)
    if name == "9606.protein.links.v10.5.txt":
        if lines and lines[0].startswith("protein1"):
            lines = lines[1:]  # Skip header
        return [ln.split()[:2] for ln in lines]

    # Handle YouTube network format (numeric IDs, skip comments)
    if name == "com-youtube.ungraph.txt":
        data_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
        return [[int(x) for x in ln.split()[:2]] for ln in data_lines]

    # Handle Facebook network format (MTX format, skip metadata)
    if name == "socfb-Penn94.mtx":
        data_lines = [ln for ln in lines if not ln.lstrip().startswith("%")]
        if data_lines:
            data_lines = data_lines[1:]  # Skip dimension line
        return [[int(x) for x in ln.split()[:2]] for ln in data_lines]

    return lines


def create_protein_dict(lines):
    """Convert protein string IDs to numeric IDs for consistent processing"""
    protein_to_id = {}
    mapped_edges = []
    current_id = 1

    # Map each unique protein string to a numeric ID
    for pair in lines:
        src, dst = pair
        if src not in protein_to_id:
            protein_to_id[src] = current_id
            current_id += 1
        if dst not in protein_to_id:
            protein_to_id[dst] = current_id
            current_id += 1
        mapped_edges.append([protein_to_id[src], protein_to_id[dst]])

    # Create reverse mapping (ID -> protein string) for later use
    id_to_protein = [""] * len(protein_to_id)
    for protein, idx in protein_to_id.items():
        id_to_protein[idx - 1] = protein

    return mapped_edges, id_to_protein


def create_dictionary_of_keys(lines):
    """Convert edge list to adjacency dictionary for efficient neighbor lookups"""
    dok = {}
    for src, dst in lines:
        # Add undirected edges (both directions)
        dok.setdefault(src, {})[dst] = 1
        dok.setdefault(dst, {})[src] = 1
    return dok


def calculate_average_degree(dok):
    degrees = [len(neigh) for neigh in dok.values()]
    return sum(degrees) / len(degrees) if degrees else 0.0


def calculate_max_degree(dok):
    degrees = [len(neigh) for neigh in dok.values()]
    return max(degrees) if degrees else 0


def _init_clustering_worker(neighbor_sets, nodes, n, prog_q=None, counter=None):
    """Initialize global variables in each worker process (using DoK sets)"""
    global _CLUSTER_A, _CLUSTER_NODES, _CLUSTER_N, _CLUSTER_PROG_Q, _CLUSTER_WORKER_IDX
    _CLUSTER_A = neighbor_sets
    _CLUSTER_NODES = nodes
    _CLUSTER_N = n
    _CLUSTER_PROG_Q = prog_q
    # Assign unique worker ID for progress tracking
    if counter is not None:
        with counter.get_lock():
            _CLUSTER_WORKER_IDX = counter.value
            counter.value += 1
    else:
        _CLUSTER_WORKER_IDX = None


def _clustering_worker_idx(idx):
    """Compute clustering coefficient for a single node using DoK neighbor sets (optimized)."""
    neighbor_sets = _CLUSTER_A
    nodes = _CLUSTER_NODES
    q = _CLUSTER_PROG_Q
    wid = _CLUSTER_WORKER_IDX

    # Get neighbors of node at index idx
    neigh_set = neighbor_sets.get(idx, set())
    k = len(neigh_set)

    # Clustering coefficient undefined for nodes with degree < 2
    if k < 2:
        if q is not None and wid is not None:
            q.put((wid, 1))
        return nodes[idx], 0.0

    # Efficiently count links among neighbors:
    # sum_{u in N(v)} |N(v) ∩ N(u)| = 2 * links_among_neighbors
    # so links = sum_intersections / 2
    sum_intersections = 0
    for ni in neigh_set:
        ni_neighbors = neighbor_sets.get(ni, set())
        # intersection cost is min(len(neigh_set), len(ni_neighbors))
        sum_intersections += len(neigh_set & ni_neighbors)

    links = sum_intersections / 2.0
    cc = (2.0 * links) / (k * (k - 1))

    if q is not None and wid is not None:
        q.put((wid, 1))
    return nodes[idx], cc


def compute_node_attributes(dok, attr_csv_path):
    attr_csv_path = Path(attr_csv_path)

    if attr_csv_path.exists():
        node_attrs = {}
        with attr_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node, degree, clustering = row
                node_attrs[int(node)] = (int(degree), float(clustering))
        print(f"Loaded node attributes from {attr_csv_path}")
        return node_attrs

    attr_csv_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(dok.keys())
    degrees = {node: len(dok[node]) for node in nodes}

    # Build DoK neighbor-sets (index-based) from dictionary of keys
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    neighbor_sets = {idx: set() for idx in range(len(nodes))}
    for node, neigh in dok.items():
        i = node_to_idx[node]
        for nb in neigh:
            j = node_to_idx.get(nb)
            if j is not None:
                neighbor_sets[i].add(j)

    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)
    clustering_results = []

    # Setup multiprocessing with progress tracking
    counter = mp.Value("i", 0)  # Shared counter for worker IDs
    prog_q = mp.Queue()  # Queue for progress updates

    # Start separate process to handle progress display
    listener = mp.Process(
        target=_progress_listener, args=(prog_q, n_proc, len(nodes)), daemon=True
    )
    listener.start()

    chunksize = max(1, len(nodes) // (n_proc * 8))
    print(
        f"Starting clustering coefficient computation for {attr_csv_path.name}: {len(nodes)} nodes, using {n_proc} processes..."
    )

    with mp.Pool(
        processes=n_proc,
        initializer=_init_clustering_worker,
        initargs=(neighbor_sets, nodes, len(nodes), prog_q, counter),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(
                _clustering_worker_idx, range(len(nodes)), chunksize=chunksize
            ),
            total=len(nodes),
            desc="Computing clustering",
            unit="nodes",
        ):
            clustering_results.append(res)

    # Signal listener to stop and cleanup
    prog_q.put((-1, 0))
    listener.join(timeout=10)
    if listener.is_alive():
        listener.terminate()
        listener.join()
        print("Progress listener terminated after timeout.")

    t1 = time.time()
    print(f"Clustering coefficients computed in {t1-t0:.2f} seconds.")

    clustering_dict = dict(clustering_results)

    with attr_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "degree", "clustering"])
        for node in nodes:
            writer.writerow([node, degrees[node], clustering_dict[node]])

    print(f"Saved node attributes to {attr_csv_path}")
    node_attrs = {node: (degrees[node], clustering_dict[node]) for node in nodes}
    return node_attrs


def common_neighbors_worker(node, all_neighbor_sets):
    node_neighbors = all_neighbor_sets[node]
    max_common = 0
    sum_common = 0
    count = 0

    for other_node, other_neighbors in all_neighbor_sets.items():
        if node == other_node:
            continue
        common = len(node_neighbors & other_neighbors)
        sum_common += common
        max_common = max(max_common, common)
        count += 1

    avg_common = sum_common / count if count else 0
    return node, avg_common, max_common


def compute_common_neighbors_worker(node, neighbor_sets):
    neighbors = neighbor_sets[node]
    avg_common = 0
    max_common = 0
    count = 0

    for other in neighbor_sets:
        if other == node:
            continue
        common = len(neighbors & neighbor_sets[other])
        avg_common += common
        max_common = max(max_common, common)
        count += 1

    avg_common = avg_common / count if count else 0
    return node, avg_common, max_common


def _init_common_worker(neighbor_sets, nodes, node_to_idx, n, prog_q, counter):
    """Initialize global variables in each worker process for common neighbors computation (DoK)"""
    global _GLOBAL_A, _GLOBAL_NODES, _GLOBAL_NODE_TO_IDX, _GLOBAL_N, _GLOBAL_PROG_Q, _GLOBAL_WORKER_IDX
    _GLOBAL_A = neighbor_sets
    _GLOBAL_NODES = nodes
    _GLOBAL_NODE_TO_IDX = node_to_idx
    _GLOBAL_N = n
    _GLOBAL_PROG_Q = prog_q
    # Atomically assign unique worker ID
    with counter.get_lock():
        _GLOBAL_WORKER_IDX = counter.value
        counter.value += 1


def _compute_stats_idx(idx):
    """Compute common neighbor statistics for a single node using DoK sets (optimized)."""
    neighbor_sets = _GLOBAL_A
    nodes = _GLOBAL_NODES
    n = _GLOBAL_N
    q = _GLOBAL_PROG_Q
    wid = _GLOBAL_WORKER_IDX

    neigh = neighbor_sets.get(idx, set())
    if not neigh:
        if q is not None:
            q.put((wid, 1))
        return nodes[idx], 0.0, 0

    # Accumulate counts of shared neighbors for nodes encountered by walking neighbors' neighbor-lists.
    counts = {}
    for w in neigh:
        w_neighbors = neighbor_sets.get(w, set())
        for u in w_neighbors:
            # increment count of common neighbors between idx and u
            counts[u] = counts.get(u, 0) + 1

    # Exclude self
    counts.pop(idx, None)

    if counts:
        sum_common = sum(counts.values())
        max_common = max(counts.values())
    else:
        sum_common = 0
        max_common = 0

    avg_common = sum_common / (n - 1) if n > 1 else 0.0

    if q is not None:
        q.put((wid, 1))
    return nodes[idx], avg_common, max_common


def _progress_listener(q, n_workers, total):
    """Display progress bars for multiprocessing tasks"""
    per_bars = []
    total_bar = tqdm(total=total, desc="Total", position=0, unit=" nodes")

    # Create individual progress bars for each worker
    for i in range(n_workers):
        per_bars.append(
            tqdm(
                total=None,  # This was glitching so I set to None
                desc=f"worker-{i} ",
                position=i + 1,
                unit=" nodes",
                leave=False,
            )
        )

    processed = 0
    while processed < total:
        try:
            wid, cnt = q.get(timeout=1.0)
        except Empty:
            continue  # No update available right now; continue waiting without crashing.

        if wid == -1:  # Termination signal
            break
        processed += cnt
        total_bar.update(cnt)
        if 0 <= wid < n_workers:
            per_bars[wid].update(cnt)

    # Cleanup progress bars
    for pb in per_bars:
        pb.close()
    total_bar.close()


def compute_common_neighbors_stats(dok, attr_csv_path):
    attr_csv_path = Path(attr_csv_path)

    if attr_csv_path.exists():
        stats = {}
        with attr_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node, avg_common, max_common = row
                stats[int(node)] = (float(avg_common), int(max_common))
        print(f"Loaded common neighbor stats from {attr_csv_path}")
        return stats

    attr_csv_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(dok.keys())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Build DoK neighbor-sets (index-based) from DoK input
    neighbor_sets = {idx: set() for idx in range(n)}
    for node, neighbors in dok.items():
        i = node_to_idx[node]
        for nb in neighbors:
            j = node_to_idx.get(nb)
            if j is not None:
                neighbor_sets[i].add(j)

    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)
    counter = mp.Value("i", 0)
    prog_q = mp.Queue()

    listener = mp.Process(
        target=_progress_listener, args=(prog_q, n_proc, n), daemon=True
    )
    listener.start()

    print(
        f"Starting common-neighbors computation for {attr_csv_path.name}: {n} nodes, using {n_proc} processes..."
    )

    chunksize = max(1, n // (n_proc * 8))
    with mp.Pool(
        processes=n_proc,
        initializer=_init_common_worker,
        initargs=(neighbor_sets, nodes, node_to_idx, n, prog_q, counter),
    ) as pool:
        it = pool.imap_unordered(_compute_stats_idx, range(n), chunksize=chunksize)
        results_list = list(it)

    prog_q.put((-1, 0))
    listener.join(timeout=10)
    if listener.is_alive():
        listener.terminate()
        listener.join()
        print("Progress listener terminated after timeout.")

    # Convert list of results to dictionary, maintaining original node order
    results_dict = {
        node: (avg_common, max_common) for node, avg_common, max_common in results_list
    }

    t1 = time.time()
    print(f"Common neighbors stats computed in {t1 - t0:.2f} seconds.")

    with attr_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "avg_common", "max_common"])
        for node in nodes:
            avg_common, max_common = results_dict[node]
            writer.writerow([node, avg_common, max_common])

    stats = {node: results_dict[node] for node in nodes}
    return stats


def plot_degree_distribution_from_attrs(node_attrs, title):
    degrees = [deg for deg, _ in node_attrs.values()]
    cnt = Counter(degrees)
    unique = sorted(cnt.keys())
    counts = [cnt[d] for d in unique]
    plt.figure()
    plt.loglog(unique, counts, marker="o", linestyle="None")
    plt.title(f"Degree Distribution: {title}")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}degree_distribution_{title.lower()}.png")
    plt.close()
    print(f"Saved degree distribution plot for {title}.")


def plot_clustering_distribution_from_attrs(node_attrs, title):
    """Plot average clustering coefficient vs degree (log-log scale)"""
    # Group clustering coefficients by degree
    deg_to_cc = {}
    for deg, cc in node_attrs.values():
        if deg > 0:
            deg_to_cc.setdefault(deg, []).append(cc)

    degrees = sorted(deg_to_cc.keys())
    avg_ccs = [sum(deg_to_cc[deg]) / len(deg_to_cc[deg]) for deg in degrees]

    plt.figure()
    plt.loglog(degrees, avg_ccs, marker="o", linestyle="None")
    plt.title(f"Degree vs. Clustering Coefficient: {title}")
    plt.xlabel("Degree")
    plt.ylabel("Average Clustering Coefficient")
    plt.grid(True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}degree_vs_clustering_{title.lower()}.png")
    plt.close()
    print(f"Saved degree vs. clustering coefficient plot for {title}.")


def plot_common_neighbors_distribution(stats, title):
    avg_common = [v[0] for v in stats.values()]
    max_common = [v[1] for v in stats.values()]

    plt.figure()
    plt.hist(avg_common, bins=100, log=True)
    plt.title(f"Average Common Neighbors Distribution: {title}")
    plt.xlabel("Average Common Neighbors")
    plt.ylabel("Frequency (log)")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}avg_common_neighbors_{title.lower()}.png")
    plt.close()

    plt.figure()
    plt.hist(max_common, bins=100, log=True)
    plt.title(f"Max Common Neighbors Distribution: {title}")
    plt.xlabel("Max Common Neighbors")
    plt.ylabel("Frequency (log)")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}max_common_neighbors_{title.lower()}.png")
    plt.close()
    print(f"Saved common neighbors plots for {title}.")


def print_network_stats(name, node_attrs, common_stats):
    degrees = [deg for deg, _ in node_attrs.values()]
    avg_common = [v[0] for v in common_stats.values()]
    max_common = [v[1] for v in common_stats.values()]

    avg_deg = sum(degrees) / len(degrees) if degrees else 0.0
    max_deg = max(degrees) if degrees else 0
    avg_of_avg_common = sum(avg_common) / len(avg_common) if avg_common else 0.0
    max_of_max_common = max(max_common) if max_common else 0

    # Compute average clustering coefficient from node_attrs (second tuple element)
    clustering_vals = [cc for _, cc in node_attrs.values()]
    avg_clustering = (
        sum(clustering_vals) / len(clustering_vals) if clustering_vals else 0.0
    )

    print(f"\n{name} average degree: {avg_deg}")
    print(f"{name} max degree: {max_deg}")
    print(f"{name} average of average common neighbors: {avg_of_avg_common}")
    print(f"{name} max of max common neighbors: {max_of_max_common}")
    print(f"{name} average clustering coefficient: {avg_clustering}\n")


def main():
    youtube_data = load_data("../data/com-youtube.ungraph.txt")
    youtube_dok = create_dictionary_of_keys(youtube_data)
    print("YouTube data loaded and DoK built.")

    youtube_attrs = compute_node_attributes(
        youtube_dok, f"{OUTPUT_DIR}youtube_node_attrs.csv"
    )
    youtube_common = compute_common_neighbors_stats(
        youtube_dok, f"{OUTPUT_DIR}youtube_common_neighbors.csv"
    )

    print_network_stats("YouTube", youtube_attrs, youtube_common)
    plot_degree_distribution_from_attrs(youtube_attrs, "YouTube")
    plot_clustering_distribution_from_attrs(youtube_attrs, "YouTube")
    plot_common_neighbors_distribution(youtube_common, "YouTube")

    facebook_data = load_data("../data/socfb-Penn94.mtx")
    facebook_dok = create_dictionary_of_keys(facebook_data)
    print("Facebook data loaded and DoK built.")

    facebook_attrs = compute_node_attributes(
        facebook_dok, f"{OUTPUT_DIR}facebook_node_attrs.csv"
    )
    facebook_common = compute_common_neighbors_stats(
        facebook_dok, f"{OUTPUT_DIR}facebook_common_neighbors.csv"
    )

    print_network_stats("Facebook", facebook_attrs, facebook_common)
    plot_degree_distribution_from_attrs(facebook_attrs, "Facebook")
    plot_clustering_distribution_from_attrs(facebook_attrs, "Facebook")
    plot_common_neighbors_distribution(facebook_common, "Facebook")

    protein_data = load_data("../data/9606.protein.links.v10.5.txt")
    protein_edges, _ = create_protein_dict(protein_data)
    protein_dok = create_dictionary_of_keys(protein_edges)
    print("Protein data loaded and DoK built.")

    protein_attrs = compute_node_attributes(
        protein_dok, f"{OUTPUT_DIR}protein_node_attrs.csv"
    )
    protein_common = compute_common_neighbors_stats(
        protein_dok, f"{OUTPUT_DIR}protein_common_neighbors.csv"
    )

    print_network_stats("Protein", protein_attrs, protein_common)
    plot_degree_distribution_from_attrs(protein_attrs, "Protein")
    plot_clustering_distribution_from_attrs(protein_attrs, "Protein")
    plot_common_neighbors_distribution(protein_common, "Protein")


if __name__ == "__main__":
    main()
