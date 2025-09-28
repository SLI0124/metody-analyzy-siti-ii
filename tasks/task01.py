import csv
from pathlib import Path
import os
import time
import multiprocessing as mp
import matplotlib
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

matplotlib.use("Agg")

OUTPUT_DIR = "../results/task01/"


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
    """Convert edge list to adjacency dictionary (DoK sparse matrix)"""
    dok = {}
    for src, dst in lines:
        # Add undirected edges (both directions)
        dok.setdefault(src, {})[dst] = 1
        dok.setdefault(dst, {})[src] = 1
    return dok


def calculate_degrees(dok):
    """Calculate degree for each node"""
    degrees = {}
    for node in dok:
        degrees[node] = len(dok[node])
    return degrees


def worker_clustering_coefficient(args):
    """Worker function for parallel clustering coefficient calculation"""
    nodes_chunk, dok, worker_id, total_position = args
    local_results = {}

    pbar = tqdm(
        nodes_chunk,
        desc=f"Worker {worker_id:02d} CC",
        position=total_position + worker_id + 1,
        leave=False,
    )

    for node in pbar:
        # Get neighbors directly from DoK
        if node not in dok:
            local_results[node] = 0.0
            continue

        # dok[node] is a dictionary where keys are neighbors
        neighbors_dict = dok[node]
        degree = len(neighbors_dict)

        if degree < 2:
            local_results[node] = 0.0
            continue

        # Count triangles using DoK structure for O(1) lookups
        triangle_count = 0
        neighbors_list = list(neighbors_dict.keys())

        for i, neighbor_i in enumerate(neighbors_list):
            # Check if neighbor_i exists in DoK (it should, but safety first)
            if neighbor_i in dok:
                # dok[neighbor_i] gives us all neighbors of neighbor_i as dict keys
                neighbor_i_neighbors = dok[neighbor_i]

                # Check remaining neighbors to see if they're connected to neighbor_i
                for j in range(i + 1, len(neighbors_list)):
                    neighbor_j = neighbors_list[j]
                    # Direct O(1) lookup in DoK
                    if neighbor_j in neighbor_i_neighbors:
                        triangle_count += 1

        # Calculate exact clustering coefficient
        possible_triangles = degree * (degree - 1) // 2
        local_results[node] = (
            triangle_count / possible_triangles if possible_triangles > 0 else 0.0
        )

    pbar.close()
    return local_results


def calculate_clustering_coefficients_parallel(dok, n_workers=None, network_name=""):
    """Calculate clustering coefficients using parallel processing with load balancing"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    nodes = list(dok.keys())

    # Sort nodes by degree for better load balancing
    node_degrees = [(node, len(dok[node])) for node in nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)

    # Distribute nodes in round-robin fashion
    chunks = [[] for _ in range(n_workers)]
    for i, (node, _) in enumerate(node_degrees):
        chunks[i % n_workers].append(node)

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]
    actual_workers = len(chunks)

    total_pbar = tqdm(
        total=len(nodes), desc=f"Clustering {network_name}", position=0, leave=True
    )

    worker_args = [(chunk, dok, i, 0) for i, chunk in enumerate(chunks)]

    print(
        f"Starting clustering coefficient calculation with {actual_workers} workers..."
    )
    print(f"   Load distribution: {[len(chunk) for chunk in chunks]} nodes per worker")

    def update_callback(result):
        total_pbar.update(len(result))

    with mp.Pool(actual_workers) as pool:
        async_results = []
        for args in worker_args:
            async_results.append(
                pool.apply_async(
                    worker_clustering_coefficient, (args,), callback=update_callback
                )
            )

        # Wait for all results
        results = [ar.get() for ar in async_results]

    total_pbar.close()

    # Merge results
    clustering_coeffs = {}
    for result in results:
        clustering_coeffs.update(result)

    return clustering_coeffs


def worker_common_neighbors_edges(args):
    """Worker function for parallel common neighbors calculation on edges"""
    edge_chunk, dok, worker_id, total_position = args
    local_results = []

    pbar = tqdm(
        edge_chunk,
        desc=f"Worker {worker_id:02d} CN",
        position=total_position + worker_id + 1,
        leave=False,
    )

    # Process each edge in the chunk, we do intersection of neighbors and remove the nodes themselves
    for node1, node2 in pbar:
        neighbors1 = set(dok.get(node1, {}).keys())
        neighbors2 = set(dok.get(node2, {}).keys())
        common_neighbors = neighbors1 & neighbors2
        common_neighbors.discard(node1)
        common_neighbors.discard(node2)
        common_count = len(common_neighbors)
        local_results.append((node1, node2, common_count))

    pbar.close()
    return local_results


def calculate_common_neighbors_parallel(dok, n_workers=None, network_name=""):
    """Calculate common neighbors for all edges (not all pairs)"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    # Get all unique edges (node1 < node2)
    edges = []
    for node in dok:
        for neighbor in dok[node]:
            if node < neighbor:
                edges.append((node, neighbor))

    total_edges = len(edges)
    print(f"Starting common neighbors calculation with {n_workers} workers...")
    print(f"   Processing {total_edges} edges (all directly connected pairs)")

    chunk_size = max(1, total_edges // n_workers)
    chunks = [edges[i : i + chunk_size] for i in range(0, total_edges, chunk_size)]
    chunks = [chunk for chunk in chunks if chunk]
    actual_workers = len(chunks)

    total_pbar = tqdm(
        total=total_edges,
        desc=f"Common Neighbors {network_name}",
        position=0,
        leave=True,
    )

    worker_args = [(chunk, dok, i, 0) for i, chunk in enumerate(chunks)]

    all_results = []
    with mp.Pool(actual_workers) as pool:
        async_results = []
        for args in worker_args:
            async_results.append(
                pool.apply_async(
                    worker_common_neighbors_edges,
                    (args,),
                    callback=lambda res: total_pbar.update(len(res)),
                )
            )
        results = [ar.get() for ar in async_results]

    total_pbar.close()
    for result in results:
        all_results.extend(result)
    print(f"   Completed processing {total_edges} edges")
    return all_results


def save_to_csv(data, filename, headers):
    """Save data to CSV file"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = Path(OUTPUT_DIR) / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        if isinstance(data, dict):
            for key, value in data.items():
                writer.writerow([key, value])
        else:
            writer.writerows(data)

    print(f"Data saved to {filepath}")


def load_from_csv(filename):
    """Load data from CSV file if it exists"""
    filepath = Path(OUTPUT_DIR) / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None


def plot_degree_distribution(degrees, network_name):
    """Plot degree distribution in log-log scale"""
    degree_counts = Counter(degrees.values())
    degrees_list = list(degree_counts.keys())
    counts_list = list(degree_counts.values())

    plt.figure(figsize=(10, 6))
    plt.loglog(degrees_list, counts_list, "bo", markersize=3)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"{network_name} - Degree Distribution (Log-Log)")
    plt.grid(True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(
        Path(OUTPUT_DIR) / f"{network_name.lower()}_degree_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_clustering_degree_correlation(degrees, clustering_coeffs, network_name):
    """Plot clustering coefficient vs degree in log-log scale"""
    degree_clustering = {}
    for node in degrees:
        degree = degrees[node]
        cc = clustering_coeffs.get(node, 0)
        if degree not in degree_clustering:
            degree_clustering[degree] = []
        degree_clustering[degree].append(cc)

    avg_clustering = {
        degree: np.mean(ccs)
        for degree, ccs in degree_clustering.items()
        if len(ccs) > 0
    }

    degrees_list = list(avg_clustering.keys())
    clustering_list = list(avg_clustering.values())

    plt.figure(figsize=(10, 6))
    plt.loglog(degrees_list, clustering_list, "ro", markersize=3)
    plt.xlabel("Degree")
    plt.ylabel("Average Clustering Coefficient")
    plt.title(f"{network_name} - Clustering Coefficient vs Degree (Log-Log)")
    plt.grid(True)

    plt.savefig(
        Path(OUTPUT_DIR) / f"{network_name.lower()}_clustering_degree.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def analyze_network(dok, network_name):
    """Complete network analysis with timing and caching"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {network_name.upper()} NETWORK")
    print(f"{'='*60}")

    total_edges = sum(len(neighbors) for neighbors in dok.values()) // 2
    degree_stats = [len(neighbors) for neighbors in dok.values()]
    max_degree = max(degree_stats)
    avg_degree = np.mean(degree_stats)

    print(f"Network size: {len(dok)} nodes, {total_edges} edges")
    print(f"Degree stats: avg={avg_degree:.1f}, max={max_degree}")
    print(
        "Using exact computation for clustering coefficients and all edges for common neighbors"
    )

    # Degree analysis
    print("\n1. Calculating degrees...")
    start_time = time.time()

    degrees_file = f"{network_name.lower()}_degrees.csv"
    degrees_df = load_from_csv(degrees_file)

    if degrees_df is not None:
        print("   Loaded degrees from cache")
        degrees = dict(zip(degrees_df["node"], degrees_df["degree"]))
    else:
        degrees = calculate_degrees(dok)
        save_to_csv(degrees, degrees_file, ["node", "degree"])
        print("   Degrees calculated and saved")

    degree_time = time.time() - start_time
    degree_values = list(degrees.values())
    avg_degree = np.mean(degree_values)
    max_degree = max(degree_values)

    print(f"   Time: {degree_time:.2f}s | Avg: {avg_degree:.2f} | Max: {max_degree}")

    plot_degree_distribution(degrees, network_name)
    print("   Degree distribution plot saved")

    # Clustering coefficient analysis
    print("\n2. Calculating clustering coefficients...")
    start_time = time.time()

    clustering_file = f"{network_name.lower()}_clustering.csv"
    clustering_df = load_from_csv(clustering_file)

    if clustering_df is not None:
        print("   Loaded clustering coefficients from cache")
        clustering_coeffs = dict(
            zip(clustering_df["node"], clustering_df["clustering_coefficient"])
        )
    else:
        clustering_coeffs = calculate_clustering_coefficients_parallel(
            dok, network_name=network_name
        )
        save_to_csv(
            clustering_coeffs, clustering_file, ["node", "clustering_coefficient"]
        )
        print("   Clustering coefficients calculated and saved")

    clustering_time = time.time() - start_time
    clustering_values = list(clustering_coeffs.values())
    avg_clustering = np.mean(clustering_values)

    print(f"   Time: {clustering_time:.2f}s | Avg clustering: {avg_clustering:.4f}")

    plot_clustering_degree_correlation(degrees, clustering_coeffs, network_name)
    print("   Clustering-degree correlation plot saved")

    # Common neighbors analysis
    print("\n3. Calculating common neighbors...")
    start_time = time.time()

    common_neighbors_file = f"{network_name.lower()}_common_neighbors.csv"
    common_neighbors_df = load_from_csv(common_neighbors_file)

    if common_neighbors_df is not None:
        print("   Loaded common neighbors from cache")
        common_neighbors_data = common_neighbors_df.values.tolist()
    else:
        common_neighbors_data = calculate_common_neighbors_parallel(
            dok, network_name=network_name
        )
        save_to_csv(
            common_neighbors_data,
            common_neighbors_file,
            ["node1", "node2", "common_neighbors"],
        )
        print("   Common neighbors calculated and saved")

    common_neighbors_time = time.time() - start_time

    if common_neighbors_data:
        common_counts = [row[2] for row in common_neighbors_data]
        avg_common = np.mean(common_counts)
        max_common = max(common_counts)
    else:
        avg_common = 0
        max_common = 0

    print(
        f"   Time: {common_neighbors_time:.2f}s | Avg: {avg_common:.2f} | Max: {max_common}"
    )
    print(f"\n{network_name} analysis completed!")

    return {
        "network": network_name,
        "nodes": len(dok),
        "avg_degree": avg_degree,
        "max_degree": max_degree,
        "avg_clustering": avg_clustering,
        "avg_common_neighbors": avg_common,
        "max_common_neighbors": max_common,
        "degree_time": degree_time,
        "clustering_time": clustering_time,
        "common_neighbors_time": common_neighbors_time,
    }


def process_single_dataset(data_info):
    """Process a single dataset completely before moving to the next"""
    data_path, network_name, loader_func = data_info

    print(f"\n{'='*80}")
    print(f"PROCESSING {network_name.upper()} DATASET")
    print(f"{'='*80}")

    print(f"Loading {network_name} data from {data_path}...")
    start_load = time.time()

    if loader_func:
        raw_data = load_data(data_path)
        processed_data, _ = loader_func(raw_data)
        dok = create_dictionary_of_keys(processed_data)
    else:
        raw_data = load_data(data_path)
        dok = create_dictionary_of_keys(raw_data)

    load_time = time.time() - start_load
    print(f"{network_name}: {len(dok)} nodes loaded in {load_time:.2f}s")

    results = analyze_network(dok, network_name)

    # Clean up memory
    del dok
    del raw_data
    if "processed_data" in locals():
        del processed_data

    print(f"{network_name} processing completed and memory freed")
    return results


def main():
    print("=" * 80)
    print("NETWORK ANALYSIS - SEQUENTIAL PROCESSING")
    print("=" * 80)

    # Define datasets to process
    datasets_info = [
        ("../data/socfb-Penn94.mtx", "Facebook", None),
        ("../data/com-youtube.ungraph.txt", "YouTube", None),
        ("../data/9606.protein.links.v10.5.txt", "Protein", create_protein_dict),
    ]

    # Process datasets one by one
    results = []
    total_start_time = time.time()

    for i, dataset_info in enumerate(datasets_info, 1):
        _, network_name, _ = dataset_info
        print(f"\n[{i}/{len(datasets_info)}] Starting {network_name} processing...")

        try:
            network_results = process_single_dataset(dataset_info)
            results.append(network_results)

            # Save intermediate results after each dataset
            if results:
                summary_df = pd.DataFrame(results)
                summary_path = Path(OUTPUT_DIR) / "network_analysis_summary_partial.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"   ✓ Partial results saved to {summary_path}")

        except Exception as e:
            print(f"   ✗ Error processing {network_name}: {e}")
            continue

        print(f"✓ {network_name} completed ({i}/{len(datasets_info)})")

        gc.collect()

    total_time = time.time() - total_start_time

    if results:
        # Save final summary results
        summary_df = pd.DataFrame(results)
        summary_path = Path(OUTPUT_DIR) / "network_analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Final summary results saved to {summary_path}")

        # Clean up temporary partial results file
        partial_summary_path = Path(OUTPUT_DIR) / "network_analysis_summary_partial.csv"
        if partial_summary_path.exists():
            partial_summary_path.unlink()
            print(f"✓ Temporary partial results file cleaned up")

        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        for result in results:
            print(f"\n{result['network']} Network:")
            print(f"  Nodes: {result['nodes']}")
            print(
                f"  Avg/Max Degree: {result['avg_degree']:.2f} / {result['max_degree']}"
            )
            print(f"  Avg Clustering: {result['avg_clustering']:.4f}")
            print(
                f"  Avg/Max Common Neighbors: {result['avg_common_neighbors']:.2f} / {result['max_common_neighbors']}"
            )
            total_comp_time = (
                result["degree_time"]
                + result["clustering_time"]
                + result["common_neighbors_time"]
            )
            print(f"  Total computation time: {total_comp_time:.2f}s")

        print(f"\n Total analysis time: {total_time:.2f}s")
        print("=" * 80)
    else:
        print("\n No datasets were successfully processed!")


if __name__ == "__main__":
    main()
