import matplotlib

matplotlib.use("Agg")

import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
import multiprocessing as mp
import networkx as nx

OUTPUT_DIR = "../results/task01/"


def load_data(input_path):
    path = Path(input_path)
    name = path.name
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    if name == "9606.protein.links.v10.5.txt":
        # Skip header if present and return first two columns as strings
        if lines and lines[0].startswith("protein1"):
            lines = lines[1:]
        return [ln.split()[:2] for ln in lines if ln]

    if name == "com-youtube.ungraph.txt":
        # Drop all comment lines and blank lines, parse first two columns as ints
        data_lines = [ln for ln in lines if ln and not ln.lstrip().startswith("#")]
        return [[int(x) for x in ln.split()[:2]] for ln in data_lines]

    if name == "socfb-Penn94.mtx":
        # Remove comment lines (%) and the dimension line, then parse ints
        data_lines = [ln for ln in lines if ln and not ln.lstrip().startswith("%")]
        if data_lines:
            data_lines = data_lines[1:]  # skip dimensions
        return [[int(x) for x in ln.split()[:2]] for ln in data_lines if ln]

    return [ln for ln in lines if ln]


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

    id_to_protein = [None] * len(protein_to_id)
    for protein, idx in protein_to_id.items():
        id_to_protein[idx - 1] = protein

    return mapped_edges, id_to_protein


def create_dictionary_of_keys(lines):
    dok = {}
    for src, dst in lines:
        # Ensure both endpoints have an entry and record the undirected edge
        dok.setdefault(src, {})[dst] = 1
        dok.setdefault(dst, {})[src] = 1
    return dok


def calculate_average_degree(dok):
    degrees = np.fromiter((len(neigh) for neigh in dok.values()), dtype=int)
    return degrees.mean()


def calculate_max_degree(dok):
    degrees = np.fromiter((len(neigh) for neigh in dok.values()), dtype=int)
    return degrees.max()


def clustering_worker(args):
    node, dok = args
    neighbors = list(dok[node].keys())
    k = len(neighbors)
    if k < 2:
        return node, 0.0
    links = 0
    for idx1 in range(k):
        for idx2 in range(idx1 + 1, k):
            n1, n2 = neighbors[idx1], neighbors[idx2]
            if n2 in dok[n1]:
                links += 1
    cc = (2 * links) / (k * (k - 1))
    return node, cc


def compute_node_attributes(dok, attr_csv_path):
    attr_csv_path = Path(attr_csv_path)
    node_attrs = {}
    if attr_csv_path.exists():
        with attr_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                node, degree, clustering = row
                node_attrs[int(node)] = (int(degree), float(clustering))
        print(f"Loaded node attributes from {attr_csv_path}")
        return node_attrs

    # Ensure output directory exists before writing
    attr_csv_path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(dok.keys())
    degrees = np.array([len(dok[node]) for node in nodes])

    # Parallel clustering coefficient computation (limit to n-1 cores)
    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=n_proc) as pool:
        clustering_results = pool.map(
            clustering_worker, [(node, dok) for node in nodes]
        )
    clustering = np.zeros(len(nodes), dtype=float)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    for node, cc in clustering_results:
        clustering[node_to_idx[node]] = cc
    t1 = time.time()
    print(f"Clustering coefficients computed in {t1-t0:.2f} seconds.")

    with attr_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "degree", "clustering"])
        for i, node in enumerate(nodes):
            writer.writerow([node, degrees[i], clustering[i]])
    print(f"Saved node attributes to {attr_csv_path}")
    node_attrs = {node: (degrees[i], clustering[i]) for i, node in enumerate(nodes)}
    return node_attrs


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
    degrees = []
    avg_ccs = []
    for deg in sorted(deg_to_cc):
        degrees.append(deg)
        avg_ccs.append(np.mean(deg_to_cc[deg]))
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


def common_neighbors_worker(args):
    node, dok = args
    neighbors = set(dok[node].keys())
    max_common = 0
    sum_common = 0
    count = 0
    for other in dok:
        if node == other:
            continue
        other_neighbors = set(dok[other].keys())
        common = len(neighbors & other_neighbors)
        sum_common += common
        if common > max_common:
            max_common = common
        count += 1
    avg_common = sum_common / count if count else 0
    return node, avg_common, max_common


def compute_common_neighbors_stats(dok, attr_csv_path):
    attr_csv_path = Path(attr_csv_path)
    stats = {}
    if attr_csv_path.exists():
        with attr_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node, avg_common, max_common = row
                stats[int(node)] = (float(avg_common), int(max_common))
        print(f"Loaded common neighbor stats from {attr_csv_path}")
        return stats

    # Ensure output directory exists before writing
    attr_csv_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = list(dok.keys())
    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=n_proc) as pool:
        results = pool.map(common_neighbors_worker, [(node, dok) for node in nodes])
    t1 = time.time()
    print(f"Common neighbors computed in {t1-t0:.2f} seconds.")

    with attr_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "avg_common", "max_common"])
        for node, avg_common, max_common in results:
            writer.writerow([node, avg_common, max_common])
    stats = {node: (avg_common, max_common) for node, avg_common, max_common in results}
    return stats


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


def main():
    # --- YouTube ---
    t0 = time.time()
    youtube_data = load_data("../data/com-youtube.ungraph.txt")
    youtube_dok = create_dictionary_of_keys(youtube_data)
    t1 = time.time()
    print(f"YouTube data loaded and DoK built in {t1-t0:.2f} seconds.")

    t0 = time.time()
    youtube_attrs = compute_node_attributes(
        youtube_dok, f"{OUTPUT_DIR}youtube_node_attrs.csv"
    )
    t1 = time.time()
    print(f"YouTube node attributes computed in {t1-t0:.2f} seconds.")

    t0 = time.time()
    youtube_common = compute_common_neighbors_stats(
        youtube_dok, f"{OUTPUT_DIR}youtube_common_neighbors.csv"
    )
    t1 = time.time()
    print(f"YouTube common neighbors stats computed in {t1-t0:.2f} seconds.")

    youtube_avg_degree = np.mean([deg for deg, _ in youtube_attrs.values()])
    youtube_max_degree = max(deg for deg, _ in youtube_attrs.values())
    print(f"YouTube average degree: {youtube_avg_degree}")
    print(f"YouTube max degree: {youtube_max_degree}")
    avg_common = np.mean([v[0] for v in youtube_common.values()])
    max_common = max([v[1] for v in youtube_common.values()])
    print(f"YouTube average of average common neighbors: {avg_common}")
    print(f"YouTube max of max common neighbors: {max_common}")

    plot_degree_distribution_from_attrs(youtube_attrs, "YouTube")
    plot_clustering_distribution_from_attrs(youtube_attrs, "YouTube")
    plot_common_neighbors_distribution(youtube_common, "YouTube")

    # --- Facebook ---
    t0 = time.time()
    facebook_data = load_data("../data/socfb-Penn94.mtx")
    facebook_dok = create_dictionary_of_keys(facebook_data)
    t1 = time.time()
    print(f"Facebook data loaded and DoK built in {t1-t0:.2f} seconds.")

    t0 = time.time()
    facebook_attrs = compute_node_attributes(
        facebook_dok, f"{OUTPUT_DIR}facebook_node_attrs.csv"
    )
    t1 = time.time()
    print(f"Facebook node attributes computed in {t1-t0:.2f} seconds.")

    t0 = time.time()
    facebook_common = compute_common_neighbors_stats(
        facebook_dok, f"{OUTPUT_DIR}facebook_common_neighbors.csv"
    )
    t1 = time.time()
    print(f"Facebook common neighbors stats computed in {t1-t0:.2f} seconds.")

    facebook_avg_degree = np.mean([deg for deg, _ in facebook_attrs.values()])
    facebook_max_degree = max(deg for deg, _ in facebook_attrs.values())
    print(f"Facebook average degree: {facebook_avg_degree}")
    print(f"Facebook max degree: {facebook_max_degree}")
    avg_common = np.mean([v[0] for v in facebook_common.values()])
    max_common = max([v[1] for v in facebook_common.values()])
    print(f"Facebook average of average common neighbors: {avg_common}")
    print(f"Facebook max of max common neighbors: {max_common}")

    plot_degree_distribution_from_attrs(facebook_attrs, "Facebook")
    plot_clustering_distribution_from_attrs(facebook_attrs, "Facebook")
    plot_common_neighbors_distribution(facebook_common, "Facebook")

    # --- Protein ---
    t0 = time.time()
    protein_data = load_data("../data/9606.protein.links.v10.5.txt")
    protein_edges, protein_id_map = create_protein_dict(protein_data)
    protein_dok = create_dictionary_of_keys(protein_edges)
    t1 = time.time()
    print(f"Protein data loaded and DoK built in {t1-t0:.2f} seconds.")

    t0 = time.time()
    protein_attrs = compute_node_attributes(
        protein_dok, f"{OUTPUT_DIR}protein_node_attrs.csv"
    )
    t1 = time.time()
    print(f"Protein node attributes computed in {t1-t0:.2f} seconds.")

    t0 = time.time()
    protein_common = compute_common_neighbors_stats(
        protein_dok, f"{OUTPUT_DIR}protein_common_neighbors.csv"
    )
    t1 = time.time()
    print(f"Protein common neighbors stats computed in {t1-t0:.2f} seconds.")

    protein_avg_degree = np.mean([deg for deg, _ in protein_attrs.values()])
    protein_max_degree = max(deg for deg, _ in protein_attrs.values())
    print(f"Protein average degree: {protein_avg_degree}")
    print(f"Protein max degree: {protein_max_degree}")
    avg_common = np.mean([v[0] for v in protein_common.values()])
    max_common = max([v[1] for v in protein_common.values()])
    print(f"Protein average of average common neighbors: {avg_common}")
    print(f"Protein max of max common neighbors: {max_common}")

    plot_degree_distribution_from_attrs(protein_attrs, "Protein")
    plot_clustering_distribution_from_attrs(protein_attrs, "Protein")
    plot_common_neighbors_distribution(protein_common, "Protein")

    # --- NetworkX Validation ---
    print("\n--- NetworkX Validation ---")
    for name, data, dok in [
        ("YouTube", youtube_data, youtube_dok),
        ("Facebook", facebook_data, facebook_dok),
        ("Protein", protein_edges, protein_dok),
    ]:
        G = nx.Graph()
        G.add_edges_from(data)
        avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
        max_deg = max(dict(G.degree()).values())
        print(
            f"{name} NetworkX: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"
        )
        print(f"{name} NetworkX average degree: {avg_deg}")
        print(f"{name} NetworkX max degree: {max_deg}")


if __name__ == "__main__":
    main()
