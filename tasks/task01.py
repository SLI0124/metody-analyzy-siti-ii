import matplotlib

matplotlib.use("Agg")

import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
import multiprocessing as mp
from functools import partial
import networkx as nx

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


def clustering_worker(args):
    node, neighbors_dict, all_neighbors = args
    neighbors = list(neighbors_dict.keys())
    k = len(neighbors)
    if k < 2:
        return node, 0.0

    links = 0
    for i in range(k):
        for j in range(i + 1, k):
            n1, n2 = neighbors[i], neighbors[j]
            if n2 in all_neighbors.get(n1, {}):
                links += 1

    cc = (2 * links) / (k * (k - 1))
    return node, cc


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

    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)
    worker_args = [(node, dok[node], dok) for node in nodes]

    with mp.Pool(processes=n_proc) as pool:
        clustering_results = pool.map(clustering_worker, worker_args)

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
    neighbor_sets = {node: set(dok[node].keys()) for node in nodes}

    t0 = time.time()
    n_proc = max(1, mp.cpu_count() - 1)
    worker_func = partial(common_neighbors_worker, all_neighbor_sets=neighbor_sets)

    with mp.Pool(processes=n_proc) as pool:
        results = pool.map(worker_func, nodes)

    t1 = time.time()
    print(f"Common neighbors computed in {t1-t0:.2f} seconds.")

    with attr_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "avg_common", "max_common"])
        for node, avg_common, max_common in results:
            writer.writerow([node, avg_common, max_common])

    stats = {node: (avg_common, max_common) for node, avg_common, max_common in results}
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
    protein_edges, protein_id_map = create_protein_dict(protein_data)
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

    print("\n--- NetworkX Validation ---")
    for name, data in [
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
