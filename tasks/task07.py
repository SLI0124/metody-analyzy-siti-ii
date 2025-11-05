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
    return {nodes[i]: (layout[i][0], layout[i][1]) for i in range(len(nodes))}


def get_node_layer_membership(layer_networks):
    membership = defaultdict(list)
    for layer, graph in layer_networks.items():
        for node in graph.nodes():
            if graph.degree(node) > 0:
                membership[node].append(layer)
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


def draw_membership_pies(nodes, layout, membership, layers_order, palette_hex, sizes):
    ax = plt.gca()
    for i, node in enumerate(nodes):
        x, y = layout[i]
        mem = membership.get(node, [])
        pie = [1 if layer in mem else 0 for layer in layers_order]
        total = sum(pie)

        if total == 0:
            # Draw a solid gray circle for nodes with no layer membership
            radius = 0.06 + 0.12 * (
                (sizes[i] - min(sizes)) / (max(sizes) - min(sizes))
                if max(sizes) > min(sizes)
                else 1
            )
            circle = Wedge(
                (x, y), radius, 0, 360, facecolor="#cccccc", ec="#222", lw=1.0
            )
            ax.add_patch(circle)
            continue

        fracs = [v / total for v in pie]
        radius = 0.06 + 0.12 * (
            (sizes[i] - min(sizes)) / (max(sizes) - min(sizes))
            if max(sizes) > min(sizes)
            else 1
        )

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
    layers_order = layer_list
    degrees = dict(flattened_network.degree())

    max_deg = max(degrees.values()) if degrees.values() else 1
    sizes = (
        [20 + 80 * (max(1, degrees.get(n, 0)) - 1) / (max_deg - 1) for n in nodes]
        if max_deg > 1
        else [20] * len(nodes)
    )

    palette = sns.color_palette("husl", n_colors=len(layers_order))
    palette_hex = [mcolors.to_hex(c) for c in palette]

    layout_coords = [layout.get(n, (0.0, 0.0)) for n in nodes]

    plt.figure(figsize=(9, 7))
    pos_dict = {n: layout_coords[i] for i, n in enumerate(nodes)}

    nx.draw_networkx_edges(
        flattened_network, pos_dict, alpha=0.25, edge_color="#777777"
    )

    # Draw pies as the nodes themselves
    draw_membership_pies(
        nodes, layout_coords, membership, layers_order, palette_hex, sizes
    )

    for idx, layer in enumerate(layers_order):
        plt.scatter([], [], c=palette_hex[idx], label=layer, s=80)
    plt.legend(ncol=2, fontsize=10, frameon=False, title="Layers")

    plt.title("Augmented Flattened Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "augmented_flattened.png", dpi=200)
    plt.close()


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
    layers = layer_list
    n_layers = len(layers)

    cols = min(2, n_layers)
    rows = (n_layers + cols - 1) // cols
    plt.figure(figsize=(8 * cols, 6 * rows))

    spread_pos = {node: (x * 1.5, y * 1.5) for node, (x, y) in layout.items()}

    for i in range(n_layers):
        plt.subplot(rows, cols, i + 1)
        plot_growth_step(layers[: i + 1], layer_networks, spread_pos, i + 1)

    plt.suptitle("Progressive Layer Growth", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "progressive_layer_growth.png", dpi=200, bbox_inches="tight"
    )
    plt.close()


def main():
    print(" - Loading data...")
    layers_df, nodes_df, edges_df = load_data()
    layer_networks, layer_list = build_layer_graphs(layers_df, nodes_df, edges_df)

    flattened_network = create_flattened_network(layer_networks)
    layout = compute_layout(flattened_network)

    print(" - Plotting layer slices...")
    plot_layer_slices(layer_networks, layer_list, layout)

    print(" - Plotting augmented flattened network...")
    membership = get_node_layer_membership(layer_networks)
    # Sort layers by number of edges (ascending, smallest first)
    layer_edge_counts = {layer: graph.number_of_edges() for layer, graph in layer_networks.items()}
    sorted_layers = sorted(layer_list, key=lambda l: layer_edge_counts[l])
    plot_augmented_flattened(flattened_network, sorted_layers, layout, membership)

    print(" - Community detection...")
    plot_community_detection(flattened_network, layout, "all_layers")

    print(" - Plotting progressive layer growth...")
    # Use the same sorted_layers for progressive growth
    plot_progressive_growth(layer_networks, sorted_layers, layout)

    combos = [("facebook", "work"), ("coauthor", "leisure")]
    for combo in combos:
        existing = [layer for layer in combo if layer in layer_networks]
        if existing:
            sub_networks = {layer: layer_networks[layer] for layer in existing}
            flat_sub = merge_networks(sub_networks)
            sub_pos = {n: layout[n] for n in layout if n in flat_sub.nodes()}
            plot_community_detection(flat_sub, sub_pos, "_".join(existing))

    print(f"Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
