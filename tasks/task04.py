import networkx as nx
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

RESULTS_DIR = "../results/task04"


def ensure_results_dir(subdir=None):
    path = RESULTS_DIR if subdir is None else os.path.join(RESULTS_DIR, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def save_edge_list_csv(G, prefix, outdir):
    with open(os.path.join(outdir, f"{prefix}_edges.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        for u, v in G.edges():
            writer.writerow([u, v])
    with open(os.path.join(outdir, f"{prefix}_nodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Label", "Degree", "ClusteringCoefficient"])
        for node in G.nodes():
            degree = G.degree(node)
            clustering = nx.clustering(G, node)
            writer.writerow([node, f"Node_{node}", degree, clustering])


def save_analysis_csv(results):
    with open(
        os.path.join(RESULTS_DIR, "network_comparison.csv"), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Nodes",
                "Edges",
                "Avg_Degree",
                "Max_Degree",
                "Avg_Clustering",
                "Density",
                "Connected_Components",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["name"],
                    r["nodes"],
                    r["edges"],
                    f"{r['avg_degree']:.2f}",
                    r["max_degree"],
                    f"{r['clustering']:.4f}",
                    f"{r['density']:.4f}",
                    r["components"],
                ]
            )


def plot_degree_distribution(degree_dist, prefix, plots_dir):
    degrees = list(degree_dist.keys())
    counts = list(degree_dist.values())
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, counts, "o", alpha=0.6, markersize=6)
    plt.xlabel("Degree (k)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Degree Distribution - {prefix}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(plots_dir, f"{prefix}_degree_dist.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()


def link_selection_model(n_nodes, m_edges, internal_links=False, internal_prob=0.1):
    G = nx.complete_graph(m_edges)
    edges_array = np.array(list(G.edges()))
    for i in tqdm(range(m_edges, n_nodes), desc="Link Selection"):
        targets = set()
        while len(targets) < m_edges:
            edge_idx = np.random.randint(len(edges_array))
            target = edges_array[edge_idx, np.random.randint(2)]
            targets.add(target)
        G.add_node(i)
        new_edges = [(i, t) for t in targets]
        G.add_edges_from(new_edges)
        edges_array = np.vstack([edges_array, new_edges])
        # Internal links: random attachment
        if internal_links and np.random.rand() < internal_prob and i > m_edges + 5:
            u = np.random.randint(i)
            v = np.random.randint(i)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                edges_array = np.vstack([edges_array, [[u, v]]])
    return G


def copying_model(
    n_nodes, m_edges, p_random=0.2, internal_links=False, internal_prob=0.1
):
    G = nx.complete_graph(m_edges)
    neighbors_cache = {i: set(G.neighbors(i)) for i in range(m_edges)}
    for i in tqdm(range(m_edges, n_nodes), desc="Copying Model"):
        targets = set()
        for _ in range(m_edges):
            target_node = np.random.randint(i)
            if np.random.rand() < p_random or len(neighbors_cache[target_node]) == 0:
                targets.add(target_node)
            else:
                copied_target = np.random.choice(list(neighbors_cache[target_node]))
                targets.add(copied_target)
        G.add_edges_from([(i, t) for t in targets])
        neighbors_cache[i] = targets
        for t in targets:
            neighbors_cache[t].add(i)
        if internal_links and np.random.rand() < internal_prob and i > m_edges + 5:
            u = np.random.randint(i)
            v = np.random.randint(i)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                neighbors_cache[u].add(v)
                neighbors_cache[v].add(u)
    return G


def barabasi_albert_model(n_nodes, m_edges, internal_links=False, internal_prob=0.1):
    if not internal_links:
        return nx.barabasi_albert_graph(n_nodes, m_edges)
    G = nx.complete_graph(m_edges)
    repeated_nodes = list(range(m_edges)) * m_edges
    for i in tqdm(range(m_edges, n_nodes), desc="BA with Internal Links"):
        targets = set()
        while len(targets) < m_edges:
            target = repeated_nodes[np.random.randint(len(repeated_nodes))]
            if target not in targets:
                targets.add(target)
        G.add_node(i)
        G.add_edges_from([(i, t) for t in targets])
        repeated_nodes.extend(targets)
        repeated_nodes.extend([i] * m_edges)
        if np.random.rand() < internal_prob and i > m_edges + 5:
            u = repeated_nodes[np.random.randint(len(repeated_nodes))]
            v = repeated_nodes[np.random.randint(len(repeated_nodes))]
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                repeated_nodes.extend([u, v])
    return G


def analyze_network(G, name):
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    results = {
        "name": name,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees),
        "max_degree": max(degrees),
        "clustering": nx.average_clustering(G),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "degree_dist": degree_counts,
    }
    return results


def random_node_deletion(G, fraction=0.05):
    """Delete a fraction of nodes chosen uniformly at random."""
    n_remove = int(fraction * G.number_of_nodes())
    nodes_to_remove = np.random.choice(G.nodes(), n_remove, replace=False)
    G.remove_nodes_from(nodes_to_remove)
    return G


def process_model(model_name, n_nodes, m_edges, deletion_fraction=0.05):
    model_dir = ensure_results_dir(model_name)
    orig_dir = ensure_results_dir(os.path.join(model_name, "original"))
    internal_dir = ensure_results_dir(os.path.join(model_name, "internal"))
    deletion_dir = ensure_results_dir(os.path.join(model_name, "deletion"))
    plots_dir = ensure_results_dir(os.path.join(model_name, "plots"))
    results_all = []

    # Original
    G = (
        link_selection_model(n_nodes, m_edges)
        if model_name == "Link_Selection"
        else (
            copying_model(n_nodes, m_edges)
            if model_name == "Copying_Model"
            else barabasi_albert_model(n_nodes, m_edges)
        )
    )
    res_name = "original"
    results = analyze_network(G, res_name)
    save_edge_list_csv(G, res_name, orig_dir)
    plot_degree_distribution(results["degree_dist"], res_name, plots_dir)
    print(
        f"  {res_name}: Nodes: {results['nodes']}, Edges: {results['edges']}, "
        f"Avg Degree: {results['avg_degree']:.2f}, Clustering: {results['clustering']:.4f}"
    )
    results_all.append(results)

    # Internal links
    if model_name == "Link_Selection":
        G_internal = link_selection_model(n_nodes, m_edges, internal_links=True)
    elif model_name == "Copying_Model":
        G_internal = copying_model(n_nodes, m_edges, internal_links=True)
    else:
        G_internal = barabasi_albert_model(n_nodes, m_edges, internal_links=True)
    res_name = "internal"
    results_internal = analyze_network(G_internal, res_name)
    save_edge_list_csv(G_internal, res_name, internal_dir)
    plot_degree_distribution(results_internal["degree_dist"], res_name, plots_dir)
    print(
        f"  {res_name}: Nodes: {results_internal['nodes']}, Edges: {results_internal['edges']}, "
        f"Avg Degree: {results_internal['avg_degree']:.2f}, Clustering: {results_internal['clustering']:.4f}"
    )
    results_all.append(results_internal)

    # Node deletion (from original)
    G_deleted = G.copy()
    G_deleted = random_node_deletion(G_deleted, fraction=deletion_fraction)
    res_name = "deletion"
    results_deleted = analyze_network(G_deleted, res_name)
    save_edge_list_csv(G_deleted, res_name, deletion_dir)
    plot_degree_distribution(results_deleted["degree_dist"], res_name, plots_dir)
    print(
        f"  {res_name}: Nodes: {results_deleted['nodes']}, Edges: {results_deleted['edges']}, "
        f"Avg Degree: {results_deleted['avg_degree']:.2f}, Clustering: {results_deleted['clustering']:.4f}"
    )
    results_all.append(results_deleted)

    return results_all


def main():
    ensure_results_dir()
    n_nodes = 2000
    m_edges = 3
    deletion_fraction = 0.05

    model_names = ["Link_Selection", "Copying_Model", "Barabasi_Albert"]
    all_results = []
    for model_name in model_names:
        print(f"\nProcessing {model_name}...")
        results_list = process_model(model_name, n_nodes, m_edges, deletion_fraction)
        all_results.extend(results_list)
    save_analysis_csv(all_results)
    print(f"\n✓ All results saved to {RESULTS_DIR}")
    print("Files generated in subdirectories per model.")
    print("  - edge list CSVs (for Gephi)")
    print("  - node attribute CSVs (for Gephi)")
    print("  - degree distribution plots")
    print("  - 1 network comparison CSV (in main results dir)")


if __name__ == "__main__":
    main()
