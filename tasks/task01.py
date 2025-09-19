import matplotlib

matplotlib.use("Agg")

import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import os

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
    clustering = np.zeros(len(nodes), dtype=float)
    node_idx = {node: i for i, node in enumerate(nodes)}

    for i, node in enumerate(nodes):
        neighbors = list(dok[node].keys())
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
            continue
        # Count edges between neighbors
        links = 0
        for idx1 in range(k):
            for idx2 in range(idx1 + 1, k):
                n1, n2 = neighbors[idx1], neighbors[idx2]
                if n2 in dok[n1]:
                    links += 1
        clustering[i] = (2 * links) / (k * (k - 1))

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


def main():
    # youtube
    youtube_data = load_data("../data/com-youtube.ungraph.txt")
    print(f"Loaded {len(youtube_data)} lines of YouTube data.")
    youtube_dok = create_dictionary_of_keys(youtube_data)
    print(f"YouTube DOK has {len(youtube_dok)} keys.")
    youtube_attrs = compute_node_attributes(
        youtube_dok, f"{OUTPUT_DIR}youtube_node_attrs.csv"
    )
    youtube_avg_degree = np.mean([deg for deg, _ in youtube_attrs.values()])
    print(f"Youtube average degree: {youtube_avg_degree}\n")

    # facebook
    facebook_data = load_data("../data/socfb-Penn94.mtx")
    print(f"Loaded {len(facebook_data)} lines of Facebook data.")
    facebook_dok = create_dictionary_of_keys(facebook_data)
    print(f"Facebook DOK has {len(facebook_dok)} keys.")
    facebook_attrs = compute_node_attributes(
        facebook_dok, f"{OUTPUT_DIR}facebook_node_attrs.csv"
    )
    facebook_avg_degree = np.mean([deg for deg, _ in facebook_attrs.values()])
    print(f"Facebook average degree: {facebook_avg_degree}\n")

    # protein
    protein_data = load_data("../data/9606.protein.links.v10.5.txt")
    print(f"Loaded {len(protein_data)} lines of Protein data.")
    protein_edges, protein_id_map = create_protein_dict(protein_data)
    print(f"Protein data mapped to {len(protein_id_map)} unique proteins.")
    protein_dok = create_dictionary_of_keys(protein_edges)
    print(f"Protein DOK has {len(protein_dok)} keys.")
    protein_attrs = compute_node_attributes(
        protein_dok, f"{OUTPUT_DIR}protein_node_attrs.csv"
    )
    protein_avg_degree = np.mean([deg for deg, _ in protein_attrs.values()])
    print(f"Protein average degree: {protein_avg_degree}\n")

    # max degree
    youtube_max_degree = max(deg for deg, _ in youtube_attrs.values())
    print(f"Youtube max degree: {youtube_max_degree}")
    facebook_max_degree = max(deg for deg, _ in facebook_attrs.values())
    print(f"Facebook max degree: {facebook_max_degree}")
    protein_max_degree = max(deg for deg, _ in protein_attrs.values())
    print(f"Protein max degree: {protein_max_degree}\n")

    # degree distribution (log-log plot)
    plot_degree_distribution_from_attrs(youtube_attrs, "YouTube")
    plot_degree_distribution_from_attrs(facebook_attrs, "Facebook")
    plot_degree_distribution_from_attrs(protein_attrs, "Protein")

    # clustering coefficient distribution (log-log plot)
    plot_clustering_distribution_from_attrs(youtube_attrs, "YouTube")
    plot_clustering_distribution_from_attrs(facebook_attrs, "Facebook")
    plot_clustering_distribution_from_attrs(protein_attrs, "Protein")


if __name__ == "__main__":
    main()
