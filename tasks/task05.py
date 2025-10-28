from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import combinations

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
        names=["layerID", "actorID1", "actorID2", "weight"],
    )
    return layers_df, nodes_df, edges_df


def build_networks(edges_df, layers_df):
    layer_names = dict(zip(layers_df["layerID"], layers_df["layerLabel"]))
    layer_networks = {}

    for layer_id in layers_df["layerID"]:
        layer_edges = edges_df[edges_df["layerID"] == layer_id]
        graph = nx.Graph()
        for _, row in layer_edges.iterrows():
            graph.add_edge(row["actorID1"], row["actorID2"], weight=row["weight"])
        layer_networks[layer_names[layer_id]] = graph

    return layer_networks


def get_actor_neighbors(actor_id, graphs):
    neighbors = set()
    for graph in graphs:
        if actor_id in graph:
            neighbors.update(graph.neighbors(actor_id))
    return neighbors


def compute_measures(layer_networks, nodes_df):
    print("Computing multilayer measures...")

    results = pd.DataFrame(
        {"actorID": nodes_df["nodeID"], "actorLabel": nodes_df["nodeLabel"]}
    )

    layer_names = list(layer_networks.keys())

    # Basic degree measures
    for layer_name, graph in layer_networks.items():
        degree_dict = dict(graph.degree())
        results[f"degree_{layer_name}"] = results["actorID"].map(
            lambda x: degree_dict.get(x, 0)
        )

    degree_columns = [f"degree_{layer}" for layer in layer_names]
    results["degree_all_layers"] = results[degree_columns].sum(axis=1)
    results["degree_deviation"] = results[degree_columns].std(axis=1)

    # Neighborhood measures
    for layer_name, graph in layer_networks.items():
        neighbors_dict = {
            actor: len(set(graph.neighbors(actor))) if actor in graph else 0
            for actor in results["actorID"]
        }
        results[f"neighborhood_{layer_name}"] = results["actorID"].map(neighbors_dict)

    # Total neighborhood across all layers
    def get_total_neighbors(actor_id):
        return len(get_actor_neighbors(actor_id, layer_networks.values()))

    results["neighborhood_all_layers"] = results["actorID"].apply(get_total_neighbors)

    # Connective redundancy
    results["connective_redundancy"] = results.apply(
        lambda row: (
            1 - (row["neighborhood_all_layers"] / row["degree_all_layers"])
            if row["degree_all_layers"] > 0
            else 0
        ),
        axis=1,
    )

    # Individual layer connective redundancy
    for layer_name in layer_names:
        results[f"connective_redundancy_{layer_name}"] = results.apply(
            lambda row: (
                1 - (row[f"neighborhood_{layer_name}"] / row[f"degree_{layer_name}"])
                if row[f"degree_{layer_name}"] > 0
                else 0
            ),
            axis=1,
        )

    # Exclusive neighborhood
    for layer_name, graph in layer_networks.items():
        other_graphs = [g for name, g in layer_networks.items() if name != layer_name]

        def get_exclusive(actor_id):
            if actor_id not in graph:
                return 0
            layer_neighbors = set(graph.neighbors(actor_id))
            other_neighbors = get_actor_neighbors(actor_id, other_graphs)
            return len(layer_neighbors - other_neighbors)

        results[f"exclusive_neighborhood_{layer_name}"] = results["actorID"].apply(
            get_exclusive
        )

    # Layer combinations
    for r in range(2, len(layer_names) + 1):
        for combo in combinations(layer_names, r):
            combo_name = "_".join(combo)
            combo_graphs = [layer_networks[layer] for layer in combo]

            def get_combo_neighbors(actor_id):
                return len(get_actor_neighbors(actor_id, combo_graphs))

            results[f"neighborhood_{combo_name}"] = results["actorID"].apply(
                get_combo_neighbors
            )

            degree_sum = sum(results[f"degree_{layer}"] for layer in combo)
            results[f"connective_redundancy_{combo_name}"] = results.apply(
                lambda row, combo=combo_name: (
                    1 - (row[f"neighborhood_{combo}"] / degree_sum[row.name])
                    if degree_sum[row.name] > 0
                    else 0
                ),
                axis=1,
            )

    return results


def analyze_layers(layer_networks):
    stats = {}
    for layer_name, graph in layer_networks.items():
        stats[layer_name] = {
            "actors": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "avg_degree": (
                np.mean(list(dict(graph.degree()).values()))
                if graph.number_of_nodes() > 0
                else 0
            ),
        }
    return pd.DataFrame(stats).T.round(4)


def plot_layer_comparison(layer_stats_df):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    layers = layer_stats_df.index
    ax1.bar(layers, layer_stats_df["edges"])
    ax1.set_title("Edges per Layer")
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
    num_plots = len(layer_columns) + 1
    cols = 3
    rows = (num_plots + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, col in enumerate(layer_columns):
        layer_name = col.replace("degree_", "")
        degree_values = results[col]
        axes[i].hist(degree_values, bins=30, alpha=0.7, edgecolor="black")
        axes[i].set_title(f"{layer_name.capitalize()} Layer")
        axes[i].set_xlabel("Degree")
        axes[i].set_ylabel("Frequency")

    axes[len(layer_columns)].hist(
        results["degree_all_layers"], bins=30, alpha=0.7, edgecolor="black", color="red"
    )
    axes[len(layer_columns)].set_title("Aggregated Network")
    axes[len(layer_columns)].set_xlabel("Degree")
    axes[len(layer_columns)].set_ylabel("Frequency")

    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "degree_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_correlations(results, layer_networks):
    layer_columns = [f"degree_{layer}" for layer in layer_networks.keys()]
    correlation_data = results[layer_columns].corr()
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
    )
    plt.title("Degree Correlation Between Layers")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "degree_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_actors(results, layer_networks):
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    top_actors = results.nlargest(10, "degree_all_layers")
    axes[0].barh(range(len(top_actors)), top_actors["degree_all_layers"])
    axes[0].set_yticks(range(len(top_actors)))
    axes[0].set_yticklabels(top_actors["actorLabel"])
    axes[0].set_title("Top 10 Actors by Total Degree")
    axes[0].invert_yaxis()

    layer_cols = [f"degree_{layer}" for layer in layer_networks.keys()]
    layer_activity = results[layer_cols].sum()
    layer_activity.index = [
        col.replace("degree_", "").capitalize() for col in layer_activity.index
    ]
    axes[1].pie(layer_activity.values, labels=layer_activity.index, autopct="%1.1f%%")
    axes[1].set_title("Activity Distribution Across Layers")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_actors.png", dpi=300, bbox_inches="tight")
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
            f"{layer_name.capitalize()}\n{graph.number_of_nodes()} actors, "
            f"{graph.number_of_edges()} edges"
        )

    # Aggregated network
    graph_agg = nx.Graph()
    for graph in layer_networks.values():
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
        f"Aggregated\n{graph_agg.number_of_nodes()} actors, "
        f"{graph_agg.number_of_edges()} edges"
    )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "network_overview.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_edge_overlap(layer_networks):
    layer_names = list(layer_networks.keys())
    layer_edges = {name: set(graph.edges()) for name, graph in layer_networks.items()}
    overlap_matrix = np.zeros((len(layer_names), len(layer_names)))

    for i, l1 in enumerate(layer_names):
        for j, l2 in enumerate(layer_names):
            if i != j:
                intersection = len(layer_edges[l1] & layer_edges[l2])
                union = len(layer_edges[l1] | layer_edges[l2])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0

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
    )
    plt.title("Edge Overlap (Jaccard Index)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "edge_overlap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_actor_heatmap(results, layer_networks):
    layer_cols = [f"degree_{layer}" for layer in layer_networks.keys()]
    top_15 = results.nlargest(15, "degree_all_layers")
    heatmap_data = top_15[layer_cols].set_index(top_15["actorLabel"])
    heatmap_data.columns = [
        col.replace("degree_", "").capitalize() for col in heatmap_data.columns
    ]

    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", fmt="d")
    plt.title("Top 15 Actors Across Layers")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_actors_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(results, layer_stats_df):
    results.to_csv(RESULTS_DIR / "actor_measures.csv", index=False)
    layer_stats_df.to_csv(RESULTS_DIR / "layer_statistics.csv")


def main():
    print("Starting multilayer network analysis...")

    layers_df, nodes_df, edges_df = load_data()
    print(
        f"Loaded: {len(nodes_df)} actors, {len(edges_df)} edges, {len(layers_df)} layers"
    )

    layer_networks = build_networks(edges_df, layers_df)
    print(f"Created networks: {list(layer_networks.keys())}")

    results = compute_measures(layer_networks, nodes_df)
    layer_stats_df = analyze_layers(layer_networks)

    print("Generating visualizations...")
    plot_layer_comparison(layer_stats_df)
    print("Layer comparison plot saved.")
    plot_degree_distributions(results, layer_networks)
    print("Degree distributions plot saved.")
    plot_correlations(results, layer_networks)
    print("Correlations plot saved.")
    plot_top_actors(results, layer_networks)
    print("Top actors plot saved.")
    plot_network_overview(layer_networks)
    print("Network overview plot saved.")
    plot_edge_overlap(layer_networks)
    print("Edge overlap plot saved.")
    plot_actor_heatmap(results, layer_networks)
    print("Actor heatmap plot saved.")

    save_results(results, layer_stats_df)
    print(f"Results saved to {RESULTS_DIR}/")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
