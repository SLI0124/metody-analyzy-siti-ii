import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from pathlib import Path
import os

DATA_DIR = Path("../data/CS-Aarhus_Multiplex_Social/CS-Aarhus_Multiplex_Social/Dataset")
RESULTS_DIR = Path("../results/task05")
RESULTS_DIR.mkdir(exist_ok=True)


def load_multiplex_data():
    layers_df = pd.read_csv(DATA_DIR / "CS-Aarhus_layers.txt", sep=" ")
    nodes_df = pd.read_csv(DATA_DIR / "CS-Aarhus_nodes.txt", sep=" ")
    edges_df = pd.read_csv(
        DATA_DIR / "CS-Aarhus_multiplex.edges",
        sep=" ",
        names=["layerID", "nodeID1", "nodeID2", "weight"],
    )
    return layers_df, nodes_df, edges_df


def create_layer_networks(edges_df, layers_df):
    layer_networks = {}
    layer_names = dict(zip(layers_df["layerID"], layers_df["layerLabel"]))

    for layer_id in layers_df["layerID"]:
        layer_edges = edges_df[edges_df["layerID"] == layer_id]
        G = nx.Graph()
        for _, row in layer_edges.iterrows():
            G.add_edge(row["nodeID1"], row["nodeID2"], weight=row["weight"])
        layer_networks[layer_names[layer_id]] = G

    return layer_networks


def create_aggregated_networks(edges_df):
    # Weighted aggregated network
    G_weighted = nx.Graph()
    edge_weights = defaultdict(int)

    for _, row in edges_df.iterrows():
        edge = (row["nodeID1"], row["nodeID2"])
        edge_weights[edge] += row["weight"]

    for (u, v), weight in edge_weights.items():
        G_weighted.add_edge(u, v, weight=weight)

    # Unweighted aggregated network
    G_unweighted = nx.Graph()
    for _, row in edges_df.iterrows():
        G_unweighted.add_edge(row["nodeID1"], row["nodeID2"])

    return G_weighted, G_unweighted


def compute_degree_centralities(layer_networks, G_weighted, G_unweighted, nodes_df):
    results = pd.DataFrame(
        {"nodeID": nodes_df["nodeID"], "nodeLabel": nodes_df["nodeLabel"]}
    )

    for layer_name, G in layer_networks.items():
        degrees = dict(G.degree())
        results[f"degree_{layer_name}"] = results["nodeID"].map(
            lambda x: degrees.get(x, 0)
        )

    weighted_degrees = dict(G_weighted.degree())
    unweighted_degrees = dict(G_unweighted.degree())

    results["degree_weighted_agg"] = results["nodeID"].map(
        lambda x: weighted_degrees.get(x, 0)
    )
    results["degree_unweighted_agg"] = results["nodeID"].map(
        lambda x: unweighted_degrees.get(x, 0)
    )

    layer_columns = [f"degree_{layer}" for layer in layer_networks.keys()]
    results["degree_total"] = results[layer_columns].sum(axis=1)

    return results


def analyze_layer_significance(layer_networks):
    layer_stats = {}

    for layer_name, G in layer_networks.items():
        stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": (
                sum(dict(G.degree()).values()) / G.number_of_nodes()
                if G.number_of_nodes() > 0
                else 0
            ),
            "max_degree": (
                max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
            ),
            "clustering_coefficient": nx.average_clustering(G),
            "connected_components": nx.number_connected_components(G),
        }
        layer_stats[layer_name] = stats

    layer_stats_df = pd.DataFrame(layer_stats).T
    layer_stats_df = layer_stats_df.round(4)

    return layer_stats_df


def calculate_degree_deviation(degree_results):
    layer_columns = [
        col
        for col in degree_results.columns
        if col.startswith("degree_") and "agg" not in col and "total" not in col
    ]

    degree_results["degree_mean"] = degree_results[layer_columns].mean(axis=1)
    degree_results["degree_std"] = degree_results[layer_columns].std(axis=1)
    degree_results["degree_cv"] = degree_results["degree_std"] / (
        degree_results["degree_mean"] + 1e-6
    )  # Coefficient of variation

    for col in layer_columns:
        layer_name = col.replace("degree_", "")
        degree_results[f"deviation_{layer_name}"] = (
            degree_results[col] - degree_results["degree_mean"]
        )

    return degree_results


def identify_layer_specialists(degree_results):
    layer_columns = [
        col
        for col in degree_results.columns
        if col.startswith("degree_") and "agg" not in col and "total" not in col
    ]

    high_cv_threshold = degree_results["degree_cv"].quantile(0.8)
    specialists = degree_results[
        degree_results["degree_cv"] >= high_cv_threshold
    ].copy()

    high_degree_threshold = {}
    for col in layer_columns:
        high_degree_threshold[col] = degree_results[col].quantile(0.8)

    layer_specialists = []
    for _, row in degree_results.iterrows():
        high_layers = []
        low_layers = []

        for col in layer_columns:
            layer_name = col.replace("degree_", "")
            if row[col] >= high_degree_threshold[col]:
                high_layers.append(layer_name)
            elif row[col] <= 1:  # Low degree threshold
                low_layers.append(layer_name)

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

    specialists_df = pd.DataFrame(layer_specialists)
    return specialists, specialists_df


def detect_communities(layer_networks, G_unweighted):
    communities_results = {}

    for layer_name, G in layer_networks.items():
        if G.number_of_nodes() > 0:
            try:
                # Use Louvain community detection
                communities = nx.community.louvain_communities(G, seed=42)
                communities_results[layer_name] = {
                    "num_communities": len(communities),
                    "communities": communities,
                    "modularity": nx.community.modularity(G, communities),
                }
            except Exception as e:
                communities_results[layer_name] = {
                    "num_communities": 0,
                    "communities": [],
                    "modularity": 0,
                }

    if G_unweighted.number_of_nodes() > 0:
        try:
            agg_communities = nx.community.louvain_communities(G_unweighted, seed=42)
            communities_results["aggregated"] = {
                "num_communities": len(agg_communities),
                "communities": agg_communities,
                "modularity": nx.community.modularity(G_unweighted, agg_communities),
            }
        except Exception as e:
            print(f"Could not compute communities for aggregated network: {e}")

    return communities_results


def create_visualizations(layer_networks, degree_results, layer_stats_df):
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Layer comparison - number of edges and density
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    layers = layer_stats_df.index
    edges = layer_stats_df["edges"]
    density = layer_stats_df["density"]

    ax1.bar(layers, edges)
    ax1.set_title("Number of Edges per Layer")
    ax1.set_ylabel("Number of Edges")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(layers, density)
    ax2.set_title("Network Density per Layer")
    ax2.set_ylabel("Density")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "layer_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Degree distribution across layers
    layer_columns = [
        col
        for col in degree_results.columns
        if col.startswith("degree_") and "agg" not in col and "total" not in col
    ]

    num_layers = len(layer_columns)
    num_plots = num_layers + 1  # +1 for aggregated network

    # Calculate subplot layout
    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
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

    # Aggregated network degree distribution
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

    # Hide any unused subplots
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


def create_additional_visualizations(
    layer_networks, degree_results, layer_stats_df, edges_df
):
    # Network overview
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, (layer_name, G) in enumerate(layer_networks.items()):
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        node_sizes = [dict(G.degree()).get(node, 0) * 50 + 50 for node in G.nodes()]
        nx.draw(
            G,
            pos,
            ax=axes[i],
            node_size=node_sizes,
            node_color="lightblue",
            edge_color="gray",
            alpha=0.7,
            with_labels=False,
        )
        axes[i].set_title(
            f"{layer_name.capitalize()}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

    # Aggregated network
    G_agg = nx.Graph()
    for layer_name, G in layer_networks.items():
        G_agg.add_edges_from(G.edges())

    pos = nx.spring_layout(G_agg, k=1, iterations=50, seed=42)
    node_sizes = [dict(G_agg.degree()).get(node, 0) * 30 + 30 for node in G_agg.nodes()]
    nx.draw(
        G_agg,
        pos,
        ax=axes[5],
        node_size=node_sizes,
        node_color="red",
        edge_color="darkgray",
        alpha=0.7,
        with_labels=False,
    )
    axes[5].set_title(
        f"Aggregated\n{G_agg.number_of_nodes()} nodes, {G_agg.number_of_edges()} edges"
    )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "network_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Top nodes and specialization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

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

    # Heatmaps and correlations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top_15 = degree_results.nlargest(15, "degree_total")
    heatmap_data = top_15[layer_cols].set_index(top_15["nodeLabel"])
    heatmap_data.columns = [
        col.replace("degree_", "").capitalize() for col in heatmap_data.columns
    ]
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", ax=axes[0, 0], fmt="d")
    axes[0, 0].set_title("Top 15 Nodes Across Layers")

    layer_names = [col.replace("degree_", "") for col in layer_cols]
    corr_matrix = degree_results[layer_cols].corr()
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[0, 1], fmt=".2f"
    )
    axes[0, 1].set_title("Layer Correlations")

    activity_counts = [
        sum([1 for col in layer_cols if row[col] > 0])
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

    # Network structure analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.Set3(np.linspace(0, 1, len(layer_networks)))
    for i, (layer_name, G) in enumerate(layer_networks.items()):
        degrees = [d for n, d in G.degree()]
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

    # Edge overlap matrix
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

    # Multi-layer edge participation
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

    # Path lengths
    structure_data = []
    for layer_name, G in layer_networks.items():
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            avg_path = nx.average_shortest_path_length(G)
        else:
            if G.number_of_nodes() > 1:
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
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


def save_results(
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
    
    layers_df, nodes_df, edges_df = load_multiplex_data()
    print(f"Data loaded: {len(nodes_df)} nodes, {len(edges_df)} edges, {len(layers_df)} layers")

    layer_networks = create_layer_networks(edges_df, layers_df)
    print(f"Layer networks created: {list(layer_networks.keys())}")
    
    G_weighted, G_unweighted = create_aggregated_networks(edges_df)
    print(f"Aggregated networks created: {G_unweighted.number_of_nodes()} nodes, {G_unweighted.number_of_edges()} edges")
    
    degree_results = compute_degree_centralities(
        layer_networks, G_weighted, G_unweighted, nodes_df
    )
    print("Degree centralities computed")
    
    layer_stats_df = analyze_layer_significance(layer_networks)
    print("Layer significance analyzed")

    degree_results = calculate_degree_deviation(degree_results)
    print("Degree deviation calculated")

    specialists, specialists_df = identify_layer_specialists(degree_results)
    print(f"Layer specialists identified: {len(specialists)} high CV nodes, {len(specialists_df)} mixed activity nodes")

    communities_results = detect_communities(layer_networks, G_unweighted)
    print("Community detection completed")
    
    create_visualizations(layer_networks, degree_results, layer_stats_df)
    print("Basic visualizations created")
    
    create_additional_visualizations(
        layer_networks, degree_results, layer_stats_df, edges_df
    )
    print("Additional visualizations created")
    
    save_results(
        degree_results, layer_stats_df, communities_results, specialists, specialists_df
    )
    print(f"Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
