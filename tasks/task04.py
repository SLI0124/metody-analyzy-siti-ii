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


def save_edge_list_csv(G, model_name, outdir):
    with open(os.path.join(outdir, f"{model_name}_edges.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        for u, v in G.edges():
            writer.writerow([u, v])
    with open(os.path.join(outdir, f"{model_name}_nodes.csv"), "w", newline="") as f:
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


def plot_degree_distribution(degree_dist, name, outdir):
    degrees = list(degree_dist.keys())
    counts = list(degree_dist.values())
    plt.figure(figsize=(10, 6))
    plt.loglog(degrees, counts, "o", alpha=0.6, markersize=6)
    plt.xlabel("Degree (k)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Degree Distribution - {name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(outdir, f"{name}_degree_dist.png"), dpi=150, bbox_inches="tight"
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


def main():
    ensure_results_dir()
    n_nodes = 2000
    m_edges = 3
    models = [
        ("Link_Selection", lambda: link_selection_model(n_nodes, m_edges)),
        ("Copying_Model", lambda: copying_model(n_nodes, m_edges)),
        ("Barabasi_Albert", lambda: barabasi_albert_model(n_nodes, m_edges)),
        (
            "Internal_Link_Selection",
            lambda: link_selection_model(n_nodes, m_edges, internal_links=True),
        ),
        (
            "Internal_Copying",
            lambda: copying_model(n_nodes, m_edges, internal_links=True),
        ),
        (
            "Internal_BA",
            lambda: barabasi_albert_model(n_nodes, m_edges, internal_links=True),
        ),
    ]
    all_results = []
    for name, model_func in models:
        print(f"\nGenerating {name} network...")
        outdir = ensure_results_dir(name)
        G = model_func()
        results = analyze_network(G, name)
        all_results.append(results)
        save_edge_list_csv(G, name, outdir)
        plot_degree_distribution(results["degree_dist"], name, outdir)
        print(
            f"  Nodes: {results['nodes']}, Edges: {results['edges']}, "
            f"Avg Degree: {results['avg_degree']:.2f}, "
            f"Clustering: {results['clustering']:.4f}"
        )
    save_analysis_csv(all_results)
    print(f"\n✓ All results saved to {RESULTS_DIR}")
    print("Files generated in subdirectories per model.")
    print("  - edge list CSVs (for Gephi)")
    print("  - node attribute CSVs (for Gephi)")
    print("  - degree distribution plots")
    print("  - 1 network comparison CSV (in main results dir)")


if __name__ == "__main__":
    main()
