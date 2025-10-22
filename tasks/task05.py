from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("default")
sns.set_palette("husl")

DATA_DIR = Path("../data/CS-Aarhus_Multiplex_Social/CS-Aarhus_Multiplex_Social/Dataset")
RESULTS_DIR = Path("../results/task05")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    layers_df = pd.read_csv(DATA_DIR / "CS-Aarhus_layers.txt", sep=" ")
    nodes_df = pd.read_csv(DATA_DIR / "CS-Aarhus_nodes.txt", sep=" ")
    edges_df = pd.read_csv(
        DATA_DIR / "CS-Aarhus_multiplex.edges",
        sep=" ",
        names=["layerID", "nodeID1", "nodeID2", "weight"],
    )
    return layers_df, nodes_df, edges_df


def build_networks(edges_df, layers_df):
    layer_names = dict(zip(layers_df["layerID"], layers_df["layerLabel"]))
    layer_networks = {}

    for layer_id in layers_df["layerID"]:
        layer_edges = edges_df[edges_df["layerID"] == layer_id]
        graph = nx.Graph()
        for _, row in layer_edges.iterrows():
            graph.add_edge(row["nodeID1"], row["nodeID2"], weight=row["weight"])
        layer_networks[layer_names[layer_id]] = graph

    graph_unweighted = nx.Graph()
    for _, row in edges_df.iterrows():
        graph_unweighted.add_edge(row["nodeID1"], row["nodeID2"])

    return layer_networks, graph_unweighted


def compute_required_measures(layer_networks, nodes_df):
    results = pd.DataFrame(
        {"nodeID": nodes_df["nodeID"], "nodeLabel": nodes_df["nodeLabel"]}
    )

    all_layers = set(layer_networks.keys())

    # Degree centrality per layer
    degree_dicts = {}
    for layer_name, graph in layer_networks.items():
        degree_dicts[layer_name] = dict(graph.degree())
        results[f"degree_{layer_name}"] = results["nodeID"].map(
            lambda x, d=degree_dicts[layer_name]: d.get(x, 0)
        )

    layer_columns = [f"degree_{layer}" for layer in layer_networks.keys()]

    # Degree deviation (standard deviation across layers)
    results["degree_deviation"] = results[layer_columns].std(axis=1)

    # Neighborhood centrality - distinct neighbors per layer
    neighbors_dicts = {}
    for layer_name, graph in layer_networks.items():
        neighbors_dicts[layer_name] = {
            node: set(graph.neighbors(node)) for node in graph.nodes()
        }
        results[f"neighborhood_{layer_name}"] = results["nodeID"].map(
            lambda x, nd=neighbors_dicts[layer_name]: len(nd.get(x, set()))
        )

    # Total neighborhood across all layers (distinct neighbors)
    def get_all_neighbors(node_id):
        all_neighbors = set()
        for graph in layer_networks.values():
            if node_id in graph:
                all_neighbors.update(graph.neighbors(node_id))
        return len(all_neighbors)

    results["neighborhood_all_layers"] = results["nodeID"].apply(get_all_neighbors)

    # Connective redundancy: 1 - (neighborhood / degree)
    results["degree_all_layers"] = results[layer_columns].sum(axis=1)
    results["connective_redundancy"] = results.apply(
        lambda row: (
            1 - (row["neighborhood_all_layers"] / row["degree_all_layers"])
            if row["degree_all_layers"] > 0
            else 0
        ),
        axis=1,
    )

    # Exclusive neighborhood - neighbors in specific layers but not in others
    for layer_name, graph in layer_networks.items():
        other_layer_set = all_layers - {layer_name}

        def get_exclusive_neighbors(
            node_id, target_graph=graph, other_layers=other_layer_set
        ):
            if node_id not in target_graph:
                return 0

            layer_neighbors = set(target_graph.neighbors(node_id))
            other_neighbors = set()

            for other_layer in other_layers:
                other_graph = layer_networks[other_layer]
                if node_id in other_graph:
                    other_neighbors.update(other_graph.neighbors(node_id))

            exclusive = layer_neighbors - other_neighbors
            return len(exclusive)

        results[f"exclusive_neighborhood_{layer_name}"] = results["nodeID"].apply(
            get_exclusive_neighbors
        )

    return results


def analyze_layers(layer_networks):
    stats = {}
    for layer_name, graph in layer_networks.items():
        stats[layer_name] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "avg_degree": (
                sum(dict(graph.degree()).values()) / graph.number_of_nodes()
                if graph.number_of_nodes() > 0
                else 0
            ),
        }
    return pd.DataFrame(stats).T.round(4)


def plot_layer_basics(layer_stats_df):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    layers = layer_stats_df.index
    ax1.bar(layers, layer_stats_df["edges"])
    ax1.set_title("Number of Edges per Layer")
    ax1.set_ylabel("Number of Edges")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(layers, layer_stats_df["density"])
    ax2.set_title("Network Density per Layer")
    ax2.set_ylabel("Density")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "layer_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_degree_distributions(results, layer_networks):
    layer_columns = [f"degree_{layer}" for layer in layer_networks.keys()]

    num_plots = len(layer_columns) + 1  # +1 for aggregated
    cols = 3
    rows = (num_plots + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, col in enumerate(layer_columns):
        layer_name = col.replace("degree_", "")
        degree_values = results[col]
        axes[i].hist(
            degree_values,
            bins=range(0, int(max(degree_values)) + 2),
            alpha=0.7,
            edgecolor="black",
        )
        axes[i].set_title(f"{layer_name.capitalize()} Layer Degree Distribution")
        axes[i].set_xlabel("Degree")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)

    # Add aggregated network degree distribution
    agg_index = len(layer_columns)
    axes[agg_index].hist(
        results["degree_all_layers"],
        bins=range(0, int(max(results["degree_all_layers"])) + 2),
        alpha=0.7,
        edgecolor="black",
        color="red",
    )
    axes[agg_index].set_title("Aggregated Network Degree Distribution")
    axes[agg_index].set_xlabel("Degree")
    axes[agg_index].set_ylabel("Frequency")
    axes[agg_index].grid(True, alpha=0.3)

    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "degree_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlations(results, layer_networks):
    layer_columns = [f"degree_{layer}" for layer in layer_networks.keys()]
    correlation_data = results[layer_columns].corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_data,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".3f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Degree Correlation Between Layers")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "degree_correlation_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_top_nodes_and_activity(results, layer_networks):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 10 nodes by total degree
    top_nodes = results.nlargest(10, "degree_all_layers")
    axes[0].barh(range(len(top_nodes)), top_nodes["degree_all_layers"])
    axes[0].set_yticks(range(len(top_nodes)))
    axes[0].set_yticklabels(top_nodes["nodeLabel"])
    axes[0].set_title("Top 10 Nodes by Total Degree")
    axes[0].invert_yaxis()

    # Layer activity distribution
    layer_cols = [f"degree_{layer}" for layer in layer_networks.keys()]
    layer_activity = results[layer_cols].sum()
    layer_activity.index = [
        col.replace("degree_", "").capitalize() for col in layer_activity.index
    ]
    axes[1].pie(layer_activity.values, labels=layer_activity.index, autopct="%1.1f%%")
    axes[1].set_title("Layer Activity Distribution")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_nodes_activity.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_network_overview(layer_networks):
    _, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, (layer_name, graph) in enumerate(layer_networks.items()):
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
        node_sizes = [
            dict(graph.degree()).get(node, 0) * 50 + 50 for node in graph.nodes()
        ]
        nx.draw(
            graph,
            pos,
            ax=axes[i],
            node_size=node_sizes,
            node_color="lightblue",
            edge_color="gray",
            alpha=0.7,
            with_labels=False,
        )
        axes[i].set_title(
            f"{layer_name.capitalize()}\n{graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

    # Aggregated network
    graph_agg = nx.Graph()
    for layer_name, graph in layer_networks.items():
        graph_agg.add_edges_from(graph.edges())

    pos = nx.spring_layout(graph_agg, k=1, iterations=50, seed=42)
    node_sizes = [
        dict(graph_agg.degree()).get(node, 0) * 30 + 30 for node in graph_agg.nodes()
    ]
    nx.draw(
        graph_agg,
        pos,
        ax=axes[5],
        node_size=node_sizes,
        node_color="red",
        edge_color="darkgray",
        alpha=0.7,
        with_labels=False,
    )
    axes[5].set_title(
        f"Aggregated\n{graph_agg.number_of_nodes()} nodes, "
        f"{graph_agg.number_of_edges()} edges"
    )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "network_overview.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_edge_overlap(layer_networks):
    layer_names = list(layer_networks.keys())
    layer_edges = {name: set(G.edges()) for name, G in layer_networks.items()}
    overlap_matrix = np.zeros((len(layer_names), len(layer_names)))

    for i, l1 in enumerate(layer_names):
        for j, l2 in enumerate(layer_names):
            if i != j:
                intersection = len(layer_edges[l1] & layer_edges[l2])
                union = len(layer_edges[l1] | layer_edges[l2])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(overlap_matrix, dtype=bool))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix,
        mask=mask,
        xticklabels=[n.capitalize() for n in layer_names],
        yticklabels=[n.capitalize() for n in layer_names],
        annot=True,
        cmap="Blues",
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Edge Overlap (Jaccard)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "edge_overlap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_path_lengths(layer_networks):
    structure_data = []
    for layer_name, graph in layer_networks.items():
        if nx.is_connected(graph):
            diameter = nx.diameter(graph)
            avg_path = nx.average_shortest_path_length(graph)
        else:
            if graph.number_of_nodes() > 1:
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                diameter = (
                    nx.diameter(subgraph) if subgraph.number_of_nodes() > 1 else 0
                )
                avg_path = (
                    nx.average_shortest_path_length(subgraph)
                    if subgraph.number_of_nodes() > 1
                    else 0
                )
            else:
                diameter = avg_path = 0
        structure_data.append((layer_name, diameter, avg_path))

    x = np.arange(len(structure_data))
    diameters = [d[1] for d in structure_data]
    avg_paths = [d[2] for d in structure_data]

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, diameters, 0.4, label="Diameter", alpha=0.7)
    plt.bar(x + 0.2, avg_paths, 0.4, label="Avg Path Length", alpha=0.7)
    plt.xticks(x, [d[0].capitalize() for d in structure_data], rotation=45)
    plt.title("Path Lengths")
    plt.ylabel("Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "path_lengths.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_15_heatmap(results, layer_networks):
    layer_cols = [f"degree_{layer}" for layer in layer_networks.keys()]
    top_15 = results.nlargest(15, "degree_all_layers")
    heatmap_data = top_15[layer_cols].set_index(top_15["nodeLabel"])
    heatmap_data.columns = [
        col.replace("degree_", "").capitalize() for col in heatmap_data.columns
    ]

    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", fmt="d")
    plt.title("Top 15 Nodes Across Layers")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_15_nodes_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(results, layer_stats_df):
    results.to_csv(RESULTS_DIR / "multilayer_measures.csv", index=False)
    layer_stats_df.to_csv(RESULTS_DIR / "layer_statistics.csv")


def main():
    print("Starting multilayer network analysis...")

    layers_df, nodes_df, edges_df = load_data()
    print(
        f"Data loaded: {len(nodes_df)} nodes, {len(edges_df)} edges, {len(layers_df)} layers"
    )

    layer_networks, _ = build_networks(edges_df, layers_df)
    print(f"✓ Networks created: {list(layer_networks.keys())}")

    results = compute_required_measures(layer_networks, nodes_df)
    print("✓ All required measures computed:")
    print("  - Degree Centrality (per layer)")
    print("  - Degree Deviation")
    print("  - Neighborhood Centrality")
    print("  - Connective Redundancy")
    print("  - Exclusive Neighborhood")

    layer_stats_df = analyze_layers(layer_networks)
    print("✓ Layer statistics computed")

    print("\nGenerating visualizations...")
    plot_layer_basics(layer_stats_df)
    print("  ✓ Layer comparison (edges & density)")

    plot_degree_distributions(results, layer_networks)
    print("  ✓ Degree distributions")

    plot_correlations(results, layer_networks)
    print("  ✓ Layer correlations")

    plot_top_nodes_and_activity(results, layer_networks)
    print("  ✓ Top 10 nodes & layer activity")

    plot_network_overview(layer_networks)
    print("  ✓ Network overview")

    plot_edge_overlap(layer_networks)
    print("  ✓ Edge overlap")

    plot_path_lengths(layer_networks)
    print("  ✓ Path lengths")

    plot_top_15_heatmap(results, layer_networks)
    print("  ✓ Top 15 nodes heatmap")

    save_results(results, layer_stats_df)
    print(f"\n✓ Results saved to {RESULTS_DIR}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
