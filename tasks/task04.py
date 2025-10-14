import networkx as nx
import numpy as np
import os
import csv

RESULTS_DIR = "../results/task04"


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_edge_list_csv(G, filename):
    with open(os.path.join(RESULTS_DIR, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        for u, v in G.edges():
            writer.writerow([u, v])


def save_analysis_results(name, avg_degree, degree_dist, clustering):
    with open(os.path.join(RESULTS_DIR, f"{name}_analysis.txt"), "w") as f:
        f.write(f"Average degree: {avg_degree:.2f}\n")
        f.write(f"Degree distribution: {degree_dist}\n")
        f.write(f"Average clustering coefficient: {clustering:.4f}\n")


def link_selection_model(n_nodes, m_edges):
    # Start with a small connected graph
    G = nx.complete_graph(m_edges)
    for i in range(m_edges, n_nodes):
        targets = set()
        while len(targets) < m_edges:
            # Select a random edge and pick one of its endpoints
            edge = list(G.edges)[np.random.randint(len(G.edges))]
            targets.add(edge[np.random.randint(2)])
        G.add_node(i)
        for t in targets:
            G.add_edge(i, t)
    return G


def copying_model(n_nodes, m_edges, p_copy=0.8):
    # Start with a small connected graph
    G = nx.complete_graph(m_edges)
    for i in range(m_edges, n_nodes):
        # Choose a random node to copy from
        copy_from = np.random.randint(i)
        neighbors = list(G.neighbors(copy_from))
        targets = set()
        # Copy edges with probability p_copy
        for n in neighbors:
            if np.random.rand() < p_copy:
                targets.add(n)
        # Add random edges if not enough
        while len(targets) < m_edges:
            candidate = np.random.randint(i)
            if candidate != i:
                targets.add(candidate)
        G.add_node(i)
        for t in targets:
            G.add_edge(i, t)
    return G


def barabasi_albert_model(n_nodes, m_edges):
    # Use networkx built-in implementation
    return nx.barabasi_albert_graph(n_nodes, m_edges)


def analyze_network(G):
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    clustering = nx.average_clustering(G)
    degree_dist = np.bincount(degrees)
    return avg_degree, degree_dist, clustering


def main():
    ensure_results_dir()
    n_nodes = 2000
    m_edges = 3

    print("Generating Link Selection Model network...")
    G_link = link_selection_model(n_nodes, m_edges)
    avg_degree, degree_dist, clustering = analyze_network(G_link)
    save_edge_list_csv(G_link, "link_selection_model.csv")
    save_analysis_results("link_selection_model", avg_degree, degree_dist, clustering)

    print("Generating Copying Model network...")
    G_copy = copying_model(n_nodes, m_edges)
    avg_degree, degree_dist, clustering = analyze_network(G_copy)
    save_edge_list_csv(G_copy, "copying_model.csv")
    save_analysis_results("copying_model", avg_degree, degree_dist, clustering)

    print("Generating Barabási–Albert Model network...")
    G_ba = barabasi_albert_model(n_nodes, m_edges)
    avg_degree, degree_dist, clustering = analyze_network(G_ba)
    save_edge_list_csv(G_ba, "barabasi_albert_model.csv")
    save_analysis_results("barabasi_albert_model", avg_degree, degree_dist, clustering)


if __name__ == "__main__":
    main()
