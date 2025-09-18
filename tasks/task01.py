import matplotlib

matplotlib.use("Agg")

import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

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
    average_degree = 0
    for node in dok:
        average_degree += len(dok[node])
    average_degree /= len(dok)
    return average_degree


def calculate_max_degree(dok):
    max_degree = 0
    for node in dok:
        degree = len(dok[node])
        if degree > max_degree:
            max_degree = degree
    return max_degree


def plot_degree_distribution(dok, title):
    degree_count = {}
    for node in dok:
        degree = len(dok[node])
        degree_count[degree] = degree_count.get(degree, 0) + 1

    degrees = list(degree_count.keys())
    counts = list(degree_count.values())

    plt.figure()
    plt.loglog(degrees, counts, marker="o", linestyle="None")
    plt.title(f"Degree Distribution: {title}")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.grid(True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}degree_distribution_{title.lower()}.png")
    plt.close()
    print(f"Saved degree distribution plot for {title}.")


def plot_clustering_distribution(dok, title):
    # Build a NetworkX graph from dok
    G = nx.Graph()
    for node, neighbors in dok.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    # Compute clustering coefficients for all nodes at once (much faster)
    print(f"Calculating clustering coefficients for {title}...")
    clustering = nx.clustering(G)
    # Group clustering coefficients by degree
    degree_to_cc = {}
    for node in G.nodes():
        deg = G.degree(node)
        cc = clustering[node]
        if deg not in degree_to_cc:
            degree_to_cc[deg] = []
        degree_to_cc[deg].append(cc)
    # Compute average clustering coefficient for each degree
    degrees = []
    avg_ccs = []
    for deg in sorted(degree_to_cc):
        if deg > 0:
            degrees.append(deg)
            avg_ccs.append(sum(degree_to_cc[deg]) / len(degree_to_cc[deg]))
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
    youtube_avg_degree = calculate_average_degree(youtube_dok)
    print(f"Youtube average degree: {youtube_avg_degree}\n")

    # facebook
    facebook_data = load_data("../data/socfb-Penn94.mtx")
    print(f"Loaded {len(facebook_data)} lines of Facebook data.")
    facebook_dok = create_dictionary_of_keys(facebook_data)
    print(f"Facebook DOK has {len(facebook_dok)} keys.")
    facebook_avg_degree = calculate_average_degree(facebook_dok)
    print(f"Facebook average degree: {facebook_avg_degree}\n")

    # protein
    protein_data = load_data("../data/9606.protein.links.v10.5.txt")
    print(f"Loaded {len(protein_data)} lines of Protein data.")
    protein_edges, protein_id_map = create_protein_dict(protein_data)
    print(f"Protein data mapped to {len(protein_id_map)} unique proteins.")
    protein_dok = create_dictionary_of_keys(protein_edges)
    print(f"Protein DOK has {len(protein_dok)} keys.")
    protein_avg_degree = calculate_average_degree(protein_dok)
    print(f"Protein average degree: {protein_avg_degree}\n")

    # max degree
    # Youtube
    youtube_max_degree = calculate_max_degree(youtube_dok)
    print(f"Youtube max degree: {youtube_max_degree}")

    # Facebook
    facebook_max_degree = calculate_max_degree(facebook_dok)
    print(f"Facebook max degree: {facebook_max_degree}")

    # Protein
    protein_max_degree = calculate_max_degree(protein_dok)
    print(f"Protein max degree: {protein_max_degree}\n")

    # networkx validation
    # Youtube
    print("Validating with NetworkX:")
    youtube_graph = nx.Graph()
    youtube_graph.add_edges_from(youtube_data)
    print(
        f"Youtube NetworkX has {youtube_graph.number_of_nodes()} nodes and {youtube_graph.number_of_edges()} edges."
    )
    print(
        f"Youtube NetworkX average degree: {sum(dict(youtube_graph.degree()).values()) / youtube_graph.number_of_nodes()}"
    )
    print(
        f"Youtube NetworkX max degree: {max(dict(youtube_graph.degree()).values())}\n"
    )

    # Facebook
    facebook_graph = nx.Graph()
    facebook_graph.add_edges_from(facebook_data)
    print(
        f"Facebook NetworkX has {facebook_graph.number_of_nodes()} nodes and {facebook_graph.number_of_edges()} edges."
    )
    print(
        f"Facebook NetworkX average degree: {sum(dict(facebook_graph.degree()).values()) / facebook_graph.number_of_nodes()}"
    )
    print(
        f"Facebook NetworkX max degree: {max(dict(facebook_graph.degree()).values())}\n"
    )

    # Protein
    protein_graph = nx.Graph()
    protein_graph.add_edges_from(protein_edges)
    print(
        f"Protein NetworkX has {protein_graph.number_of_nodes()} nodes and {protein_graph.number_of_edges()} edges."
    )
    print(
        f"Protein NetworkX average degree: {sum(dict(protein_graph.degree()).values())
                                              / protein_graph.number_of_nodes()}"
    )
    print(
        f"Protein NetworkX max degree: {max(dict(protein_graph.degree()).values())}\n"
    )

    # degree distribution (log-log plot)
    plot_degree_distribution(youtube_dok, "YouTube")
    plot_degree_distribution(facebook_dok, "Facebook")
    plot_degree_distribution(protein_dok, "Protein")

    # clustering coefficient distribution (log-log plot)
    plot_clustering_distribution(youtube_dok, "YouTube")
    plot_clustering_distribution(facebook_dok, "Facebook")
    plot_clustering_distribution(protein_dok, "Protein")


if __name__ == "__main__":
    main()
