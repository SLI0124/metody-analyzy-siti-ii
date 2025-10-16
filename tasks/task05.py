from collections import defaultdict
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

    edge_weights = defaultdict(int)
    for _, row in edges_df.iterrows():
        edge = (row["nodeID1"], row["nodeID2"])
        edge_weights[edge] += row["weight"]

    graph_weighted = nx.Graph()
    for (u, v), weight in edge_weights.items():
        graph_weighted.add_edge(u, v, weight=weight)

    graph_unweighted = nx.Graph()
    for _, row in edges_df.iterrows():
        graph_unweighted.add_edge(row["nodeID1"], row["nodeID2"])

    return layer_networks, graph_weighted, graph_unweighted


def compute_degrees(layer_networks, graph_weighted, graph_unweighted, nodes_df):
    results = pd.DataFrame(
        {"nodeID": nodes_df["nodeID"], "nodeLabel": nodes_df["nodeLabel"]}
    )

    for layer_name, graph in layer_networks.items():
        degrees = dict(graph.degree())
        results[f"degree_{layer_name}"] = results["nodeID"].map(
            lambda x: degrees.get(x, 0)
        )

    weighted_degrees = dict(graph_weighted.degree())
    unweighted_degrees = dict(graph_unweighted.degree())

    results["degree_weighted_agg"] = results["nodeID"].map(
        lambda x: weighted_degrees.get(x, 0)
    )
    results["degree_unweighted_agg"] = results["nodeID"].map(
        lambda x: unweighted_degrees.get(x, 0)
    )

    layer_columns = [f"degree_{layer}" for layer in layer_networks.keys()]
    results["degree_total"] = results[layer_columns].sum(axis=1)
    results["degree_mean"] = results[layer_columns].mean(axis=1)
    results["degree_std"] = results[layer_columns].std(axis=1)
    results["degree_cv"] = results["degree_std"] / (results["degree_mean"] + 1e-6)

    for col in layer_columns:
        layer_name = col.replace("degree_", "")
        results[f"deviation_{layer_name}"] = results[col] - results["degree_mean"]

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
            "max_degree": (
                max(dict(graph.degree()).values()) if graph.number_of_nodes() > 0 else 0
            ),
            "clustering_coefficient": nx.average_clustering(graph),
            "connected_components": nx.number_connected_components(graph),
        }
    return pd.DataFrame(stats).T.round(4)


def find_specialists(degree_results):
    layer_columns = [
        col
        for col in degree_results.columns
        if col.startswith("degree_")
        and "agg" not in col
        and "total" not in col
        and "mean" not in col
        and "std" not in col
        and "cv" not in col
    ]

    high_cv_threshold = degree_results["degree_cv"].quantile(0.8)
    specialists = degree_results[
        degree_results["degree_cv"] >= high_cv_threshold
    ].copy()

    high_degree_threshold = {
        col: degree_results[col].quantile(0.8) for col in layer_columns
    }

    layer_specialists = []
    for _, row in degree_results.iterrows():
        high_layers = [
            col.replace("degree_", "")
            for col in layer_columns
            if row[col] >= high_degree_threshold[col]
        ]
        low_layers = [
            col.replace("degree_", "") for col in layer_columns if row[col] <= 1
        ]

        if len(high_layers) >= 1 and len(low_layers) >= 1:
            layer_specialists.append(
                {
                    "nodeID": row["nodeID"],
                    "nodeLabel": row["nodeLabel"],
                    "high_layers": ", ".join(high_layers),
                    "low_layers": ", ".join(low_layers),
                    "cv": row["degree_cv"],
                }
            )

    return specialists, pd.DataFrame(layer_specialists)


def detect_communities(layer_networks, graph_unweighted):
    results = {}
    for layer_name, graph in layer_networks.items():
        if graph.number_of_nodes() > 0:
            try:
                communities = nx.community.louvain_communities(graph, seed=42)
                results[layer_name] = {
                    "num_communities": len(communities),
                    "communities": communities,
                    "modularity": nx.community.modularity(graph, communities),
                }
            except Exception:
                results[layer_name] = {
                    "num_communities": 0,
                    "communities": [],
                    "modularity": 0,
                }

    if graph_unweighted.number_of_nodes() > 0:
        try:
            agg_communities = nx.community.louvain_communities(
                graph_unweighted, seed=42
            )
            results["aggregated"] = {
                "num_communities": len(agg_communities),
                "communities": agg_communities,
                "modularity": nx.community.modularity(
                    graph_unweighted, agg_communities
                ),
            }
        except Exception:
            pass
    return results


def plot_basic_stats(degree_results, layer_stats_df):
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

    layer_columns = [
        col
        for col in degree_results.columns
        if col.startswith("degree_")
        and "agg" not in col
        and "total" not in col
        and "mean" not in col
        and "std" not in col
        and "cv" not in col
    ]

    num_plots = len(layer_columns) + 1
    cols = 3
    rows = (num_plots + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, col in enumerate(layer_columns):
        layer_name = col.replace("degree_", "")
        degree_values = degree_results[col]
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

    agg_index = len(layer_columns)
    axes[agg_index].hist(
        degree_results["degree_unweighted_agg"],
        bins=range(0, int(max(degree_results["degree_unweighted_agg"])) + 2),
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

    correlation_data = degree_results[layer_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_data, annot=True, cmap="coolwarm", center=0, square=True, fmt=".3f"
    )
    plt.title("Degree Correlation Between Layers")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "degree_correlation_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    sorted_cv = degree_results.sort_values("degree_cv", ascending=False)
    plt.subplot(1, 2, 1)
    plt.bar(range(len(sorted_cv)), sorted_cv["degree_cv"])
    plt.title("Degree Coefficient of Variation by Node")
    plt.xlabel("Node (sorted by CV)")
    plt.ylabel("Coefficient of Variation")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(degree_results["degree_mean"], degree_results["degree_std"], alpha=0.7)
    plt.xlabel("Mean Degree Across Layers")
    plt.ylabel("Standard Deviation of Degree")
    plt.title("Degree Mean vs Standard Deviation")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "degree_deviation_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_networks_and_analysis(layer_networks, degree_results, edges_df):
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
            f"{layer_name.capitalize()}\n{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )

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
        f"Aggregated\n{graph_agg.number_of_nodes()} nodes, {graph_agg.number_of_edges()} edges"
    )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "network_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

    _, axes = plt.subplots(2, 2, figsize=(16, 12))

    top_nodes = degree_results.nlargest(10, "degree_total")
    axes[0, 0].barh(range(len(top_nodes)), top_nodes["degree_total"])
    axes[0, 0].set_yticks(range(len(top_nodes)))
    axes[0, 0].set_yticklabels(top_nodes["nodeLabel"])
    axes[0, 0].set_title("Top 10 Nodes by Total Degree")
    axes[0, 0].invert_yaxis()

    layer_cols = [
        col
        for col in degree_results.columns
        if col.startswith("degree_")
        and "agg" not in col
        and "total" not in col
        and "mean" not in col
        and "std" not in col
        and "cv" not in col
    ]
    layer_activity = degree_results[layer_cols].sum()
    layer_activity.index = [
        col.replace("degree_", "").capitalize() for col in layer_activity.index
    ]
    axes[0, 1].pie(
        layer_activity.values, labels=layer_activity.index, autopct="%1.1f%%"
    )
    axes[0, 1].set_title("Layer Activity Distribution")

    axes[1, 0].hist(
        degree_results["degree_total"], bins=20, alpha=0.7, label="Total", color="blue"
    )
    axes[1, 0].hist(
        degree_results["degree_unweighted_agg"],
        bins=20,
        alpha=0.7,
        label="Aggregated",
        color="red",
    )
    axes[1, 0].legend()
    axes[1, 0].set_title("Degree Distribution Comparison")

    axes[1, 1].scatter(
        degree_results["degree_mean"], degree_results["degree_cv"], alpha=0.7
    )
    axes[1, 1].set_xlabel("Mean Degree")
    axes[1, 1].set_ylabel("Coefficient of Variation")
    axes[1, 1].set_title("Specialization Analysis")

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "detailed_degree_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top_15 = degree_results.nlargest(15, "degree_total")
    heatmap_data = top_15[layer_cols].set_index(top_15["nodeLabel"])
    heatmap_data.columns = [
        col.replace("degree_", "").capitalize() for col in heatmap_data.columns
    ]
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", ax=axes[0, 0], fmt="d")
    axes[0, 0].set_title("Top 15 Nodes Across Layers")

    corr_matrix = degree_results[layer_cols].corr()
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[0, 1], fmt=".2f"
    )
    axes[0, 1].set_title("Layer Correlations")

    activity_counts = [
        sum(1 for col in layer_cols if row[col] > 0)
        for _, row in degree_results.iterrows()
    ]
    axes[1, 0].hist(activity_counts, bins=range(0, 7), alpha=0.7, edgecolor="black")
    axes[1, 0].set_title("Multi-Layer Activity")
    axes[1, 0].set_xlabel("Number of Active Layers")

    scatter = axes[1, 1].scatter(
        degree_results["degree_total"],
        degree_results["degree_cv"],
        c=degree_results["degree_mean"],
        cmap="viridis",
        alpha=0.7,
    )
    axes[1, 1].set_title("Activity vs Specialization")
    plt.colorbar(scatter, ax=axes[1, 1], label="Mean Degree")

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "node_specialization_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.colormaps.get_cmap("Set3")(np.linspace(0, 1, len(layer_networks)))
    for i, (layer_name, graph) in enumerate(layer_networks.items()):
        degrees = [d for n, d in graph.degree()]
        if degrees:
            axes[0, 0].hist(
                degrees,
                bins=range(0, max(degrees) + 2),
                alpha=0.6,
                label=layer_name.capitalize(),
                color=colors[i],
                density=True,
            )
    axes[0, 0].set_title("Degree Distributions")
    axes[0, 0].legend()

    layer_names = list(layer_networks.keys())
    layer_edges = {name: set(G.edges()) for name, G in layer_networks.items()}
    overlap_matrix = np.zeros((len(layer_names), len(layer_names)))
    for i, l1 in enumerate(layer_names):
        for j, l2 in enumerate(layer_names):
            if i != j:
                intersection = len(layer_edges[l1] & layer_edges[l2])
                union = len(layer_edges[l1] | layer_edges[l2])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0

    sns.heatmap(
        overlap_matrix,
        xticklabels=[n.capitalize() for n in layer_names],
        yticklabels=[n.capitalize() for n in layer_names],
        annot=True,
        cmap="Blues",
        ax=axes[0, 1],
        fmt=".2f",
    )
    axes[0, 1].set_title("Edge Overlap (Jaccard)")

    edge_counts = {}
    for _, row in edges_df.iterrows():
        edge = tuple(sorted([row["nodeID1"], row["nodeID2"]]))
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    axes[1, 0].hist(
        list(edge_counts.values()),
        bins=range(1, max(edge_counts.values()) + 2),
        alpha=0.7,
        edgecolor="black",
    )
    axes[1, 0].set_title("Edge Multi-Layer Participation")

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

    axes[1, 1].bar(x - 0.2, diameters, 0.4, label="Diameter", alpha=0.7)
    axes[1, 1].bar(x + 0.2, avg_paths, 0.4, label="Avg Path Length", alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([d[0].capitalize() for d in structure_data], rotation=45)
    axes[1, 1].set_title("Path Lengths")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "network_structure_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def save_data(
    degree_results, layer_stats_df, communities_results, specialists, specialists_df
):
    degree_results.to_csv(RESULTS_DIR / "degree_centralities.csv", index=False)
    layer_stats_df.to_csv(RESULTS_DIR / "layer_statistics.csv")

    community_summary = []
    for layer_name, results in communities_results.items():
        community_summary.append(
            {
                "layer": layer_name,
                "num_communities": results["num_communities"],
                "modularity": results["modularity"],
            }
        )

    community_df = pd.DataFrame(community_summary)
    community_df.to_csv(RESULTS_DIR / "community_summary.csv", index=False)

    if not specialists.empty:
        specialists.to_csv(RESULTS_DIR / "high_cv_specialists.csv", index=False)
    if not specialists_df.empty:
        specialists_df.to_csv(RESULTS_DIR / "layer_specialists.csv", index=False)


def main():
    print("Starting multiplex network analysis...")

    layers_df, nodes_df, edges_df = load_data()
    print(
        f"Data loaded: {len(nodes_df)} nodes, {len(edges_df)} edges, {len(layers_df)} layers"
    )

    layer_networks, graph_weighted, graph_unweighted = build_networks(
        edges_df, layers_df
    )
    print(f"✓ Networks created: {list(layer_networks.keys())}")

    degree_results = compute_degrees(
        layer_networks, graph_weighted, graph_unweighted, nodes_df
    )
    print("✓ Degree centralities computed")

    layer_stats_df = analyze_layers(layer_networks)
    print("✓ Layer significance analyzed")

    specialists, specialists_df = find_specialists(degree_results)
    print(
        f"✓ Layer specialists identified: {len(specialists)} high CV nodes, {len(specialists_df)} mixed activity nodes"
    )

    communities_results = detect_communities(layer_networks, graph_unweighted)
    print("✓ Community detection completed")

    plot_basic_stats(degree_results, layer_stats_df)
    print("✓ Basic visualizations created")

    plot_networks_and_analysis(layer_networks, degree_results, edges_df)
    print("Additional visualizations created")

    save_data(
        degree_results, layer_stats_df, communities_results, specialists, specialists_df
    )
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
