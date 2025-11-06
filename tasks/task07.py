from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
import seaborn as sns
import igraph as ig
from matplotlib.patches import Wedge
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity

DATA_DIR = Path("../data/CS-Aarhus_Multiplex_Social/CS-Aarhus_Multiplex_Social/Dataset")
RESULTS_DIR = Path("../results/task07")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_TYPES = [
    "individual_layer",
    "progressive_merge",
    "combo_merge",
    "all_layers_merged",
]

COLORS = ["#2E86C1", "#28B463", "#F39C12", "#E74C3C"]
COLOR_MAP = dict(zip(NETWORK_TYPES, COLORS))


def load_data():
    layers_df = pd.read_csv(DATA_DIR / "CS-Aarhus_layers.txt", sep=" ")
    nodes_df = pd.read_csv(DATA_DIR / "CS-Aarhus_nodes.txt", sep=" ")
    edges_df = pd.read_csv(
        DATA_DIR / "CS-Aarhus_multiplex.edges",
        sep=" ",
        names=["layerID", "nodeID1", "nodeID2", "weight"],
    )
    return layers_df, nodes_df, edges_df


def build_layer_graphs(layers_df, nodes_df, edges_df):
    layer_names = dict(zip(layers_df["layerID"], layers_df["layerLabel"]))
    layer_networks = {}

    for layer_id in layers_df["layerID"]:
        graph = nx.Graph()
        graph.add_nodes_from(nodes_df["nodeID"].tolist())
        layer_edges = edges_df[edges_df["layerID"] == layer_id]

        for _, row in layer_edges.iterrows():
            graph.add_edge(
                int(row["nodeID1"]), int(row["nodeID2"]), weight=row["weight"]
            )

        layer_networks[layer_names[layer_id]] = graph

    print(f"Built {len(layer_networks)} layer networks")
    return layer_networks, list(layer_names.values())


def create_flattened_network(layer_networks, weighted=False):
    flattened = nx.Graph()
    for graph in layer_networks.values():
        flattened.add_nodes_from(graph.nodes())
        if weighted:
            for u, v, data in graph.edges(data=True):
                weight = data.get("weight", 1.0)
                if flattened.has_edge(u, v):
                    flattened[u][v]["weight"] += weight
                else:
                    flattened.add_edge(u, v, weight=weight)
        else:
            for u, v in graph.edges():
                if not flattened.has_edge(u, v):
                    flattened.add_edge(u, v)
    print(
        f"Created flattened network: {flattened.number_of_nodes()} nodes, {flattened.number_of_edges()} edges"
    )
    return flattened


def compute_layout(graph):
    nodes = list(graph.nodes())
    if not nodes:
        return {}

    idx_map = {n: i for i, n in enumerate(nodes)}
    edges = [(idx_map[u], idx_map[v]) for u, v in graph.edges()]

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    if edges:
        ig_graph.add_edges(edges)

    layout = ig_graph.layout_fruchterman_reingold(niter=200)
    print("Computed network layout")
    return {nodes[i]: (layout[i][0], layout[i][1]) for i in range(len(nodes))}


def get_node_layer_membership(layer_networks):
    membership = defaultdict(list)
    for layer, graph in layer_networks.items():
        for node in graph.nodes():
            if graph.degree(node) > 0:
                membership[node].append(layer)
    print("Computed node layer membership")
    return membership


def plot_layer_slices(layer_networks, layer_list, layout):
    for layer in layer_list:
        graph = layer_networks[layer]
        degrees = dict(graph.degree())

        plt.figure(figsize=(9, 7))
        sizes = [50 + (degrees.get(n, 0) ** 1.2) * 40 for n in graph.nodes()]
        colors = [
            "#1f77b4" if degrees.get(n, 0) > 0 else "#f0f0f0" for n in graph.nodes()
        ]

        nx.draw_networkx_edges(graph, layout, alpha=0.25, edge_color="#555555")
        nx.draw_networkx_nodes(
            graph,
            layout,
            node_size=sizes,
            node_color=colors,
            linewidths=0.6,
            edgecolors="#222222",
        )

        labels = {n: str(n) for n in graph.nodes() if degrees.get(n, 0) >= 3}
        nx.draw_networkx_labels(graph, layout, labels=labels, font_size=8)

        plt.title(f"Layer slice: {layer}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"layer_slice_{layer.replace(' ', '_')}.png", dpi=220)
        plt.close()
    print(f"Plotted {len(layer_list)} layer slices")


def draw_membership_pies(nodes, layout, membership, layers_order, palette_hex, sizes):
    ax = plt.gca()
    for i, node in enumerate(nodes):
        x, y = layout[i]
        mem = membership.get(node, [])
        pie = [1 if layer in mem else 0 for layer in layers_order]
        total = sum(pie)

        radius = 0.06 + 0.12 * (
            (sizes[i] - min(sizes)) / (max(sizes) - min(sizes))
            if max(sizes) > min(sizes)
            else 1
        )

        if total == 0:
            circle = Wedge(
                (x, y), radius, 0, 360, facecolor="#cccccc", ec="#222", lw=1.0
            )
            ax.add_patch(circle)
            continue

        fracs = [v / total for v in pie]
        start = 0.0
        for frac, cidx in zip(fracs, range(len(fracs))):
            if frac <= 0:
                start += frac
                continue

            theta1, theta2 = start * 360, (start + frac) * 360
            wedge = Wedge(
                (x, y),
                radius,
                theta1,
                theta2,
                facecolor=palette_hex[cidx],
                ec="#222",
                lw=1.0,
            )
            ax.add_patch(wedge)
            start += frac


def plot_augmented_flattened(flattened_network, layer_list, layout, membership):
    nodes = list(flattened_network.nodes())
    degrees = dict(flattened_network.degree())

    max_deg = max(degrees.values()) if degrees.values() else 1
    sizes = (
        [20 + 80 * (max(1, degrees.get(n, 0)) - 1) / (max_deg - 1) for n in nodes]
        if max_deg > 1
        else [20] * len(nodes)
    )

    palette = sns.color_palette("husl", n_colors=len(layer_list))
    palette_hex = [mcolors.to_hex(c) for c in palette]
    layout_coords = [layout.get(n, (0.0, 0.0)) for n in nodes]

    plt.figure(figsize=(9, 7))
    pos_dict = {n: layout_coords[i] for i, n in enumerate(nodes)}

    nx.draw_networkx_edges(
        flattened_network, pos_dict, alpha=0.25, edge_color="#777777"
    )
    draw_membership_pies(
        nodes, layout_coords, membership, layer_list, palette_hex, sizes
    )

    for idx, layer in enumerate(layer_list):
        plt.scatter([], [], c=palette_hex[idx], label=layer, s=80)
    plt.legend(ncol=2, fontsize=10, frameon=False, title="Layers")

    plt.title("Augmented Flattened Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "augmented_flattened.png", dpi=200)
    plt.close()
    print("Plotted augmented flattened network")


def plot_community_detection(graph, pos, name):
    communities = list(louvain_communities(graph))
    mod_score = modularity(graph, communities) if graph.number_of_edges() > 0 else 0.0

    cmap = sns.color_palette("husl", n_colors=max(2, len(communities)))
    node_color = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            node_color[node] = cmap[idx]

    plt.figure(figsize=(11, 7))
    degrees = dict(graph.degree())
    sizes = [140 + (degrees.get(n, 0) ** 1.1) * 18 for n in graph.nodes()]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=sizes,
        node_color=[node_color.get(n, "#cccccc") for n in graph.nodes()],
        edgecolors="black",
        linewidths=0.7,
    )

    labels = {n: str(n) for n in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(graph, pos, alpha=0.4)

    title = f"Communities (Louvain) - {name}"
    if mod_score is not None:
        title += f"  |  modularity={mod_score:.3f}"

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"communities_{name.replace(' ', '_')}.png", dpi=200)
    plt.close()
    print(f"Plotted community detection for {name}")

    return mod_score, len(communities)


def merge_networks(networks, weighted=False):
    merged = nx.Graph()
    for graph in networks.values():
        merged.add_nodes_from(graph.nodes())
        if weighted:
            for u, v, data in graph.edges(data=True):
                weight = data.get("weight", 1.0)
                if merged.has_edge(u, v):
                    merged[u][v]["weight"] += weight
                else:
                    merged.add_edge(u, v, weight=weight)
        else:
            for u, v in graph.edges():
                if not merged.has_edge(u, v):
                    merged.add_edge(u, v)
    return merged


def plot_growth_step(cumulative_layers, layer_networks, spread_pos, step):
    cumulative_networks = {layer: layer_networks[layer] for layer in cumulative_layers}
    cumulative_flat = merge_networks(cumulative_networks)

    degrees = dict(cumulative_flat.degree())
    max_deg = max(degrees.values()) if degrees.values() else 1
    node_colors = [degrees.get(n, 0) / max_deg for n in cumulative_flat.nodes()]
    sizes = [100 + (degrees.get(n, 0) ** 1.1) * 25 for n in cumulative_flat.nodes()]

    nx.draw_networkx_edges(cumulative_flat, spread_pos, alpha=0.3, edge_color="#666")
    nx.draw_networkx_nodes(
        cumulative_flat,
        spread_pos,
        node_size=sizes,
        node_color=node_colors,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.7,
        vmin=0,
        vmax=1,
    )

    high_deg_nodes = {
        n: str(n) for n in cumulative_flat.nodes() if degrees.get(n, 0) >= 3
    }
    nx.draw_networkx_labels(
        cumulative_flat, spread_pos, labels=high_deg_nodes, font_size=9
    )

    layer_list = " + ".join(cumulative_layers)
    edges_count = cumulative_flat.number_of_edges()
    nodes_count = cumulative_flat.number_of_nodes()
    plt.title(
        f"Step {step}: {layer_list}\n{nodes_count} nodes, {edges_count} edges",
        fontsize=11,
    )
    plt.axis("off")


def plot_progressive_growth(layer_networks, layer_list, layout):
    n_layers = len(layer_list)
    cols = min(2, n_layers)
    rows = (n_layers + cols - 1) // cols
    plt.figure(figsize=(8 * cols, 6 * rows))

    spread_pos = {node: (x * 1.5, y * 1.5) for node, (x, y) in layout.items()}

    for i in range(n_layers):
        plt.subplot(rows, cols, i + 1)
        plot_growth_step(layer_list[: i + 1], layer_networks, spread_pos, i + 1)

    plt.suptitle("Progressive Layer Growth", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "progressive_layer_growth.png", dpi=200, bbox_inches="tight"
    )
    plt.close()
    print("Plotted progressive layer growth")


def calculate_modularity_statistics(layer_networks, flattened_network):
    modularity_stats = []

    # Individual layers
    for layer_name, graph in layer_networks.items():
        if graph.number_of_edges() > 0:
            communities = list(louvain_communities(graph))
            mod_score = modularity(graph, communities)
            community_sizes = [len(comm) for comm in communities]
            modularity_stats.append(
                {
                    "network_type": "individual_layer",
                    "network_name": layer_name,
                    "modularity": mod_score,
                    "num_communities": len(communities),
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "community_sizes": community_sizes,
                    "avg_community_size": (
                        sum(community_sizes) / len(community_sizes)
                        if community_sizes
                        else 0
                    ),
                    "max_community_size": (
                        max(community_sizes) if community_sizes else 0
                    ),
                    "min_community_size": (
                        min(community_sizes) if community_sizes else 0
                    ),
                }
            )

    # All layers merged
    if flattened_network.number_of_edges() > 0:
        communities = list(louvain_communities(flattened_network))
        mod_score = modularity(flattened_network, communities)
        community_sizes = [len(comm) for comm in communities]
        modularity_stats.append(
            {
                "network_type": "all_layers_merged",
                "network_name": "all_layers",
                "modularity": mod_score,
                "num_communities": len(communities),
                "num_nodes": flattened_network.number_of_nodes(),
                "num_edges": flattened_network.number_of_edges(),
                "community_sizes": community_sizes,
                "avg_community_size": (
                    sum(community_sizes) / len(community_sizes)
                    if community_sizes
                    else 0
                ),
                "max_community_size": max(community_sizes) if community_sizes else 0,
                "min_community_size": min(community_sizes) if community_sizes else 0,
            }
        )

    # Progressive layer combinations
    layer_list = list(layer_networks.keys())
    layer_edge_counts = {
        layer: graph.number_of_edges() for layer, graph in layer_networks.items()
    }
    sorted_layers = sorted(layer_list, key=lambda l: layer_edge_counts[l])

    for i in range(1, len(sorted_layers) + 1):
        cumulative_layers = sorted_layers[:i]
        cumulative_networks = {
            layer: layer_networks[layer] for layer in cumulative_layers
        }
        cumulative_flat = merge_networks(cumulative_networks)

        if cumulative_flat.number_of_edges() > 0:
            communities = list(louvain_communities(cumulative_flat))
            mod_score = modularity(cumulative_flat, communities)
            community_sizes = [len(comm) for comm in communities]
            modularity_stats.append(
                {
                    "network_type": "progressive_merge",
                    "network_name": f"layers_1_to_{i}",
                    "layers_included": "+".join(cumulative_layers),
                    "modularity": mod_score,
                    "num_communities": len(communities),
                    "num_nodes": cumulative_flat.number_of_nodes(),
                    "num_edges": cumulative_flat.number_of_edges(),
                    "community_sizes": community_sizes,
                    "avg_community_size": (
                        sum(community_sizes) / len(community_sizes)
                        if community_sizes
                        else 0
                    ),
                    "max_community_size": (
                        max(community_sizes) if community_sizes else 0
                    ),
                    "min_community_size": (
                        min(community_sizes) if community_sizes else 0
                    ),
                }
            )

    # Specific layer combinations
    combos = [("facebook", "work"), ("coauthor", "leisure")]
    for combo in combos:
        existing = [layer for layer in combo if layer in layer_networks]
        if len(existing) >= 2:
            sub_networks = {layer: layer_networks[layer] for layer in existing}
            flat_sub = merge_networks(sub_networks)

            if flat_sub.number_of_edges() > 0:
                communities = list(louvain_communities(flat_sub))
                mod_score = modularity(flat_sub, communities)
                community_sizes = [len(comm) for comm in communities]
                modularity_stats.append(
                    {
                        "network_type": "combo_merge",
                        "network_name": "_".join(existing),
                        "layers_included": "+".join(existing),
                        "modularity": mod_score,
                        "num_communities": len(communities),
                        "num_nodes": flat_sub.number_of_nodes(),
                        "num_edges": flat_sub.number_of_edges(),
                        "community_sizes": community_sizes,
                        "avg_community_size": (
                            sum(community_sizes) / len(community_sizes)
                            if community_sizes
                            else 0
                        ),
                        "max_community_size": (
                            max(community_sizes) if community_sizes else 0
                        ),
                        "min_community_size": (
                            min(community_sizes) if community_sizes else 0
                        ),
                    }
                )

    print(f"Calculated modularity statistics for {len(modularity_stats)} networks")
    return modularity_stats


def create_analysis_plots(df):
    plt.style.use("default")
    _, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Individual layer modularity (left)
    individual_data = df[df["network_type"] == "individual_layer"].sort_values(
        "modularity", ascending=True
    )
    ax_left = axes[0]
    if not individual_data.empty:
        ax_left.barh(
            range(len(individual_data)),
            individual_data["modularity"],
            color=COLOR_MAP["individual_layer"],
            alpha=0.85,
        )
        ax_left.set_yticks(range(len(individual_data)))
        ax_left.set_yticklabels(individual_data["network_name"])
        ax_left.set_xlabel("Modularity Score", fontweight="bold")
        ax_left.set_title("Individual Layer Modularity", fontweight="bold")
        ax_left.grid(axis="x", alpha=0.25)
        for i, val in enumerate(individual_data["modularity"]):
            ax_left.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)

    # Progressive merge trend (right)
    ax_right = axes[1]
    progressive_data = df[df["network_type"] == "progressive_merge"].sort_values(
        "num_edges"
    )
    if not progressive_data.empty:
        x = list(range(1, len(progressive_data) + 1))
        y = progressive_data["modularity"].tolist()
        ax_right.plot(x, y, "o-", color=COLOR_MAP["progressive_merge"], linewidth=2)
        ax_right.set_xlabel("Progressive Merge Step", fontweight="bold")
        ax_right.set_ylabel("Modularity Score", fontweight="bold")
        ax_right.set_title(
            "Modularity Decline in Progressive Merging", fontweight="bold"
        )
        ax_right.grid(True, alpha=0.25)

        # simple step labels (short)
        step_labels = []
        for i, layers in enumerate(progressive_data.get("layers_included", [])):
            layer_names = layers.split("+") if isinstance(layers, str) else []
            label = (
                "+".join(layer_names)
                if len(layer_names) <= 2
                else f"{len(layer_names)} layers"
            )
            step_labels.append(f"Step {i+1}: {label}")

        ax_right.set_xticks(x)
        ax_right.set_xticklabels(step_labels, rotation=30, ha="right", fontsize=8)
        for xi, yi in zip(x, y):
            ax_right.text(
                xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8
            )

    plt.savefig(RESULTS_DIR / "modularity_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Created modularity analysis plots (individual layers + progressive merge)")


def save_modularity_analysis(modularity_stats):
    df = pd.DataFrame(modularity_stats)
    df.to_csv(RESULTS_DIR / "modularity_statistics.csv", index=False)

    summary_stats = []
    for network_type in df["network_type"].unique():
        subset = df[df["network_type"] == network_type]
        modularity_values = subset["modularity"]
        summary_stats.append(
            {
                "network_type": network_type,
                "count": len(modularity_values),
                "mean_modularity": modularity_values.mean(),
                "std_modularity": modularity_values.std(),
                "min_modularity": modularity_values.min(),
                "max_modularity": modularity_values.max(),
                "median_modularity": modularity_values.median(),
            }
        )

    create_analysis_plots(df)
    print("Saved modularity analysis and created plots")


def main():
    layers_df, nodes_df, edges_df = load_data()
    layer_networks, layer_list = build_layer_graphs(layers_df, nodes_df, edges_df)

    flattened_network = create_flattened_network(layer_networks)
    layout = compute_layout(flattened_network)

    plot_layer_slices(layer_networks, layer_list, layout)

    membership = get_node_layer_membership(layer_networks)
    layer_edge_counts = {
        layer: graph.number_of_edges() for layer, graph in layer_networks.items()
    }
    sorted_layers = sorted(layer_list, key=lambda l: layer_edge_counts[l])
    plot_augmented_flattened(flattened_network, sorted_layers, layout, membership)

    plot_community_detection(flattened_network, layout, "all_layers")
    plot_progressive_growth(layer_networks, sorted_layers, layout)

    modularity_stats = calculate_modularity_statistics(
        layer_networks, flattened_network
    )
    save_modularity_analysis(modularity_stats)

    combos = [("facebook", "work"), ("coauthor", "leisure")]
    for combo in combos:
        existing = [layer for layer in combo if layer in layer_networks]
        if existing:
            sub_networks = {layer: layer_networks[layer] for layer in existing}
            flat_sub = merge_networks(sub_networks)
            sub_pos = {n: layout[n] for n in layout if n in flat_sub.nodes()}
            plot_community_detection(flat_sub, sub_pos, "_".join(existing))

    print(f"Analysis complete. Results saved to {RESULTS_DIR}")
    print(f"Total networks analyzed: {len(modularity_stats)}")


if __name__ == "__main__":
    main()
