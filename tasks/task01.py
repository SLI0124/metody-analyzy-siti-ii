import csv
from pathlib import Path
import os
import time
import multiprocessing as mp
import matplotlib
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

matplotlib.use("Agg")

OUTPUT_DIR = "../results/task01/"


def load_data(input_path):
    path = Path(input_path)
    name = path.name
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]

    if name == "9606.protein.links.v10.5.txt":
        if lines and lines[0].startswith("protein1"):
            lines = lines[1:]
        return [ln.split()[:2] for ln in lines]

    if name == "com-youtube.ungraph.txt":
        data_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
        return [[int(x) for x in ln.split()[:2]] for ln in data_lines]

    if name == "socfb-Penn94.mtx":
        data_lines = [ln for ln in lines if not ln.lstrip().startswith("%")]
        if data_lines:
            data_lines = data_lines[1:]
        return [[int(x) for x in ln.split()[:2]] for ln in data_lines]

    return lines


def create_protein_dict(lines):
    protein_to_id = {}
    mapped_edges = []
    current_id = 1

    for pair in lines:
        src, dst = pair
        if src not in protein_to_id:
            protein_to_id[src] = current_id
            current_id += 1
        if dst not in protein_to_id:
            protein_to_id[dst] = current_id
            current_id += 1
        mapped_edges.append([protein_to_id[src], protein_to_id[dst]])

    id_to_protein = [""] * len(protein_to_id)
    for protein, idx in protein_to_id.items():
        id_to_protein[idx - 1] = protein

    return mapped_edges, id_to_protein


def create_dictionary_of_keys(lines):
    dok = {}
    for src, dst in lines:
        dok.setdefault(src, {})[dst] = 1
        dok.setdefault(dst, {})[src] = 1
    return dok


def calculate_average_degree(dok):
    degrees = np.fromiter((len(neigh) for neigh in dok.values()), dtype=int)
    return degrees.mean()


def calculate_max_degree(dok):
    degrees = np.fromiter((len(neigh) for neigh in dok.values()), dtype=int)
    return degrees.max()


# add clustering worker globals and initializer
_CLUSTER_A = None
_CLUSTER_NODES = None
_CLUSTER_N = None
_CLUSTER_PROG_Q = None
_CLUSTER_WORKER_IDX = None


def _init_clustering_worker(A_csr, nodes, n, prog_q=None, counter=None):
    global _CLUSTER_A, _CLUSTER_NODES, _CLUSTER_N, _CLUSTER_PROG_Q, _CLUSTER_WORKER_IDX
    _CLUSTER_A = A_csr
    _CLUSTER_NODES = nodes
    _CLUSTER_N = n
    _CLUSTER_PROG_Q = prog_q
    # assign a small worker id if counter provided
    if counter is not None:
        with counter.get_lock():
            _CLUSTER_WORKER_IDX = counter.value
            counter.value += 1
    else:
        _CLUSTER_WORKER_IDX = None


def _clustering_worker_idx(idx):
    A = _CLUSTER_A
    nodes = _CLUSTER_NODES
    q = _CLUSTER_PROG_Q
    wid = _CLUSTER_WORKER_IDX

    row = A.getrow(idx)
    neigh_idx = row.indices
    k = neigh_idx.size
    if k < 2:
        if q is not None and wid is not None:
            q.put((wid, 1))
        return nodes[idx], 0.0

    sub = A[neigh_idx][:, neigh_idx]  # k x k sparse
    s = sub.sum()
    try:
        links = float(s) / 2.0
    except Exception:
        links = float(np.asarray(s).sum()) / 2.0

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

    # Build CSR adjacency once (from DoK)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    rows, cols = [], []
    for node, neigh in dok.items():
        i = node_to_idx[node]
        for nb in neigh:
            j = node_to_idx.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)
    data_arr = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix((data_arr, (rows, cols)), shape=(len(nodes), len(nodes)))

    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)

    clustering_results = []

    # Create small shared counter + queue and start listener for progress
    counter = mp.Value("i", 0)
    prog_q = mp.Queue()
    listener = mp.Process(
        target=_progress_listener, args=(prog_q, n_proc, len(nodes)), daemon=True
    )
    listener.start()

    # initialize workers with CSR and progress queue
    chunksize = max(1, len(nodes) // (n_proc * 8))

    # Print a short informative message so user knows what's being computed
    print(
        f"Starting clustering coefficient computation for {attr_csv_path.name}: {len(nodes)} nodes, using {n_proc} processes..."
    )

    with mp.Pool(
        processes=n_proc,
        initializer=_init_clustering_worker,
        initargs=(A, nodes, len(nodes), prog_q, counter),
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

    # stop listener
    prog_q.put((-1, 0))
    # wait for listener to finish, but avoid indefinite hang
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


# worker globals for pool initializer
_GLOBAL_A = None
_GLOBAL_NODES = None
_GLOBAL_NODE_TO_IDX = None
_GLOBAL_N = None
_GLOBAL_PROG_Q = None
_GLOBAL_WORKER_IDX = None


def _init_common_worker(A_csr, nodes, node_to_idx, n, prog_q, counter):
    # initializer: share large objects and assign a small worker index
    global _GLOBAL_A, _GLOBAL_NODES, _GLOBAL_NODE_TO_IDX, _GLOBAL_N, _GLOBAL_PROG_Q, _GLOBAL_WORKER_IDX
    _GLOBAL_A = A_csr
    _GLOBAL_NODES = nodes
    _GLOBAL_NODE_TO_IDX = node_to_idx
    _GLOBAL_N = n
    _GLOBAL_PROG_Q = prog_q
    # assign a 0-based worker index atomically
    with counter.get_lock():
        _GLOBAL_WORKER_IDX = counter.value
        counter.value += 1


def _compute_stats_idx(idx):
    # uses globals set by initializer, and reports progress via _GLOBAL_PROG_Q
    A = _GLOBAL_A
    nodes = _GLOBAL_NODES
    n = _GLOBAL_N
    q = _GLOBAL_PROG_Q
    wid = _GLOBAL_WORKER_IDX

    row = A.getrow(idx)
    neighbors_idx = row.indices
    if neighbors_idx.size == 0:
        # report and return quickly
        if q is not None:
            q.put((wid, 1))
        return nodes[idx], 0.0, 0

    # sum adjacency rows of neighbors -> common neighbor counts between idx and all nodes
    sub = A[neighbors_idx]  # shape (k, n) sparse
    counts = np.asarray(sub.sum(axis=0)).ravel()  # 1D ndarray
    # exclude self
    if idx < counts.size:
        counts[idx] = 0
    sum_common = float(counts.sum())
    max_common = int(counts.max()) if counts.size > 0 else 0
    avg_common = sum_common / (n - 1) if n > 1 else 0.0

    # signal progress (one node processed)
    if q is not None:
        q.put((wid, 1))
    return nodes[idx], avg_common, max_common


def _progress_listener(q, n_workers, total):
    # run in a separate process; shows one bar per worker + a total bar

    per_bars = []
    # total bar at position 0
    total_bar = tqdm(total=total, desc="Total", position=0, unit=" nodes")
    # per-worker bars starting at position 1
    # use indeterminate bars (no total) so they don't show the global total per worker
    for i in range(n_workers):
        per_bars.append(
            tqdm(
                total=None,  # do not set total to overall total
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
        except Exception:
            continue
        if wid == -1:
            break
        processed += cnt
        total_bar.update(cnt)
        if 0 <= wid < n_workers:
            per_bars[wid].update(cnt)
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

    # Build sparse adjacency matrix (CSR) from DoK
    rows, cols = [], []
    for node, neighbors in dok.items():
        i = node_to_idx[node]
        for nb in neighbors:
            j = node_to_idx.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)

    # Create a small shared counter and a Queue for progress messages
    counter = mp.Value("i", 0)
    prog_q = mp.Queue()

    # start listener process (separate process to render tqdm without blocking)
    listener = mp.Process(
        target=_progress_listener, args=(prog_q, n_proc, n), daemon=True
    )
    listener.start()

    # Print a short informative message so user knows what's being computed
    print(
        f"Starting common-neighbors computation for {attr_csv_path.name}: {n} nodes, using {n_proc} processes..."
    )

    # parallel per-node, initializer shares A and mappings once and gets prog_q + counter
    chunksize = max(1, n // (n_proc * 8))
    with mp.Pool(
        processes=n_proc,
        initializer=_init_common_worker,
        initargs=(A, nodes, node_to_idx, n, prog_q, counter),
    ) as pool:
        it = pool.imap_unordered(_compute_stats_idx, range(n), chunksize=chunksize)
        results_list = list(it)

    # signal listener to finish and wait
    prog_q.put((-1, 0))
    # wait with timeout to avoid hanging if listener stalls
    listener.join(timeout=10)
    if listener.is_alive():
        listener.terminate()
        listener.join()
        print("Progress listener terminated after timeout.")

    # keep original node order when writing
    # results_list contains (node, avg_common, max_common) tuples;
    # convert into { node: (avg_common, max_common) }
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
    degrees = np.array([deg for deg, _ in node_attrs.values()])
    unique, counts = np.unique(degrees, return_counts=True)
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
    deg_to_cc = {}
    for deg, cc in node_attrs.values():
        if deg > 0:
            deg_to_cc.setdefault(deg, []).append(cc)

    degrees = sorted(deg_to_cc.keys())
    avg_ccs = [np.mean(deg_to_cc[deg]) for deg in degrees]

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
    avg_common = np.array([v[0] for v in stats.values()])
    max_common = np.array([v[1] for v in stats.values()])

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
    degrees = np.array([deg for deg, _ in node_attrs.values()])
    avg_common = np.array([v[0] for v in common_stats.values()])
    max_common = np.array([v[1] for v in common_stats.values()])

    print(f"{name} average degree: {np.mean(degrees)}")
    print(f"{name} max degree: {np.max(degrees)}")
    print(f"{name} average of average common neighbors: {np.mean(avg_common)}")
    print(f"{name} max of max common neighbors: {np.max(max_common)}")


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
