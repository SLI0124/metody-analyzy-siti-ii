from pathlib import Path
import itertools
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns

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
        g = nx.Graph()
        g.add_nodes_from(nodes_df["nodeID"].tolist())
        layer_edges = edges_df[edges_df["layerID"] == layer_id]
        for _, r in layer_edges.iterrows():
            g.add_edge(
                int(r["nodeID1"]), int(r["nodeID2"]), weight=r.get("weight", 1.0)
            )
        layer_networks[layer_names[layer_id]] = g

    return layer_networks, list(layer_names.values())


def compute_shared_layout(flattened):
    # Use igraph Fruchterman-Reingold layout for node positions
    import igraph as ig

    nodes = list(flattened.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in flattened.edges()]
    g_ig = ig.Graph()
    g_ig.add_vertices(len(nodes))
    if edges:
        g_ig.add_edges(edges)
    layout = g_ig.layout_fruchterman_reingold(niter=200)
    return {nodes[i]: (layout[i][0], layout[i][1]) for i in range(len(nodes))}


def plot_layer_slices(layer_networks, pos):
    layers = list(layer_networks.keys())
    for layer in layers:
        g = layer_networks[layer]
        plt.figure(figsize=(9, 7))
        degrees = dict(g.degree())
        sizes = [50 + (degrees.get(n, 0) ** 1.2) * 40 for n in g.nodes()]
        node_colors = [
            "#1f77b4" if degrees.get(n, 0) > 0 else "#f0f0f0" for n in g.nodes()
        ]
        nx.draw_networkx_edges(g, pos, alpha=0.25, edge_color="#555555")
        nx.draw_networkx_nodes(
            g,
            pos,
            node_size=sizes,
            node_color=node_colors,
            linewidths=0.6,
            edgecolors="#222222",
        )
        labels = {n: str(n) for n in g.nodes() if degrees.get(n, 0) >= 3}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
        plt.title(f"Layer slice: {layer}", fontsize=14)
        plt.axis("off")
        out = RESULTS_DIR / f"layer_slice_{layer.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()


def merge_flattened(layer_networks, weighted=False):
    F = nx.Graph()
    for g in layer_networks.values():
        F.add_nodes_from(g.nodes())
        if weighted:
            for u, v, d in g.edges(data=True):
                w = d.get("weight", 1.0)
                if F.has_edge(u, v):
                    F[u][v]["weight"] += w
                else:
                    F.add_edge(u, v, weight=w)
        else:
            for u, v in g.edges():
                if not F.has_edge(u, v):
                    F.add_edge(u, v)
    return F


def node_layer_membership(layer_networks):
    membership = defaultdict(list)
    for layer, g in layer_networks.items():
        for n in g.nodes():
            if g.degree(n) > 0:
                membership[n].append(layer)
    return membership


def compute_relevance_scores(layer_networks):
    """Compute relevance r(a,l) = degree(a,l) / degree(a, all_layers) for each actor and layer."""
    all_layers = list(layer_networks.keys())
    actors = set()
    for g in layer_networks.values():
        actors.update(g.nodes())

    # degrees per layer
    deg_layer = {l: dict(g.degree()) for l, g in layer_networks.items()}
    # total degrees
    total_deg = defaultdict(int)
    for l in all_layers:
        for a, d in deg_layer[l].items():
            total_deg[a] += d

    relevance = {l: {} for l in all_layers}
    for l in all_layers:
        for a in actors:
            den = total_deg.get(a, 0)
            relevance[l][a] = (deg_layer[l].get(a, 0) / den) if den > 0 else 0.0

    return relevance


def local_simplification_flattening(
    layer_networks, selected_layers, relevance, theta=0.3
):
    """Create flattened graph from selected layers, keeping only edges where both endpoints have relevance >= theta in the layer the edge comes from."""
    F = nx.Graph()
    layers = selected_layers
    for l in layers:
        g = layer_networks[l]
        for u, v, data in g.edges(data=True):
            # keep edge if both endpoints relevance in this layer >= theta
            if relevance[l].get(u, 0.0) >= theta and relevance[l].get(v, 0.0) >= theta:
                if F.has_edge(u, v):
                    # preserve weights if provided
                    if "weight" in data:
                        F[u][v]["weight"] = F[u][v].get("weight", 0) + data.get(
                            "weight", 1.0
                        )
                else:
                    w = data.get("weight", 1.0)
                    F.add_edge(u, v, weight=w)
    # add isolated nodes with membership so layout stable
    for l in layers:
        for n in layer_networks[l].nodes():
            if n not in F:
                F.add_node(n)
    return F


def partition_to_labels(communities, nodes_order):
    """Convert community set partition into labels aligned to nodes_order."""
    label_map = {}
    for idx, comm in enumerate(communities):
        for n in comm:
            label_map[n] = idx
    labels = [label_map.get(n, -1) for n in nodes_order]
    return labels


def compare_communities_combo(layer_networks, combo, pos_base, relevance):
    """Compare communities on flattened vs local-simplified graphs for a given combo of layers."""
    existing = [l for l in combo if l in layer_networks]
    if not existing:
        return

    sub = {l: layer_networks[l] for l in existing}
    flat_sub = merge_flattened(sub, weighted=False)

    # layout: use pos_base for nodes in union
    pos = {n: pos_base[n] for n in pos_base if n in flat_sub.nodes()}
    # ensure positions for all nodes
    missing = [n for n in flat_sub.nodes() if n not in pos]
    if missing:
        ext = nx.spring_layout(flat_sub.subgraph(missing), seed=42)
        pos.update(ext)

    # communities on flattened
    from networkx.algorithms.community import greedy_modularity_communities
    from networkx.algorithms.community.quality import modularity

    comm_flat = list(greedy_modularity_communities(flat_sub))
    mod_flat = (
        modularity(flat_sub, comm_flat) if flat_sub.number_of_edges() > 0 else 0.0
    )

    # local simplification (theta=0.3)
    simplified = local_simplification_flattening(
        layer_networks, existing, relevance, theta=0.3
    )
    # align pos for simplified
    pos_simp = {n: pos[n] for n in pos if n in simplified.nodes()}
    miss2 = [n for n in simplified.nodes() if n not in pos_simp]
    if miss2:
        ext2 = nx.spring_layout(simplified.subgraph(miss2), seed=42)
        pos_simp.update(ext2)

    comm_simp = list(greedy_modularity_communities(simplified))
    mod_simp = (
        modularity(simplified, comm_simp) if simplified.number_of_edges() > 0 else 0.0
    )

    # NMI between partitions (restrict to nodes present in both graphs)
    common_nodes = sorted(
        list(set(flat_sub.nodes()).intersection(set(simplified.nodes())))
    )
    labels_flat = partition_to_labels(comm_flat, common_nodes)
    labels_simp = partition_to_labels(comm_simp, common_nodes)
    try:
        nmi_score = normalized_mutual_info_score(labels_flat, labels_simp)
    except Exception:
        nmi_score = None

    # make side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].set_title(f"Flattened ({'+'.join(existing)}) - modularity={mod_flat:.3f}")
    # color nodes by community
    cmap = sns.color_palette("husl", n_colors=max(2, len(comm_flat)))
    color_map_flat = {n: "#cccccc" for n in flat_sub.nodes()}
    for idx, comm in enumerate(comm_flat):
        for n in comm:
            color_map_flat[n] = cmap[idx]
    nx.draw_networkx_edges(flat_sub, pos, ax=axes[0], alpha=0.2)
    nx.draw_networkx_nodes(
        flat_sub,
        pos,
        node_size=80,
        node_color=[color_map_flat[n] for n in flat_sub.nodes()],
        ax=axes[0],
    )
    axes[0].axis("off")

    axes[1].set_title(
        f"LocalSimplified theta=0.3 - modularity={mod_simp:.3f}\nNMI={nmi_score:.3f}"
        if nmi_score is not None
        else f"LocalSimplified theta=0.3 - modularity={mod_simp:.3f}"
    )
    cmap2 = sns.color_palette("husl", n_colors=max(2, len(comm_simp)))
    color_map_simp = {n: "#cccccc" for n in simplified.nodes()}
    for idx, comm in enumerate(comm_simp):
        for n in comm:
            color_map_simp[n] = cmap2[idx]
    nx.draw_networkx_edges(simplified, pos_simp, ax=axes[1], alpha=0.2)
    nx.draw_networkx_nodes(
        simplified,
        pos_simp,
        node_size=80,
        node_color=[color_map_simp[n] for n in simplified.nodes()],
        ax=axes[1],
    )
    axes[1].axis("off")

    out = RESULTS_DIR / f"compare_communities_{'_'.join(existing)}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


# draw_pie_on_nodes removed — plotting of pies is now done using igraph in plot_augmented_flattened


def plot_augmented_flattened(flattened, layer_networks, membership, pos):
    nodes = list(flattened.nodes())
    layers_order = list(layer_networks.keys())
    degrees = dict(flattened.degree())
    sizes = [
        20
        + 80
        * (max(1, degrees.get(n, 0)) - 1)
        / (max(degrees.values()) - 1 if max(degrees.values()) > 1 else 1)
        for n in nodes
    ]
    palette = sns.color_palette("husl", n_colors=len(layers_order))
    import matplotlib.colors as mcolors

    palette_hex = [mcolors.to_hex(c) for c in palette]
    layout = [pos[n] if n in pos else (0.0, 0.0) for n in nodes]
    fig, ax = plt.subplots(figsize=(12, 9))
    pos_dict = {n: layout[i] for i, n in enumerate(nodes)}
    nx.draw_networkx_edges(flattened, pos_dict, ax=ax, alpha=0.25, edge_color="#777777")
    from matplotlib.patches import Wedge

    pies_drawn = 0
    for i, n in enumerate(nodes):
        x, y = layout[i]
        mem = membership.get(n, [])
        pie = [1 if layer in mem else 0 for layer in layers_order]
        total = sum(pie)
        if total == 0:
            continue
        pies_drawn += 1
        fracs = [v / total for v in pie]
        # Make pies larger for visibility
        radius = 0.06 + 0.12 * (
            (sizes[i] - min(sizes))
            / (max(sizes) - min(sizes) if max(sizes) > min(sizes) else 1)
        )
        start = 0.0
        for frac, cidx in zip(fracs, range(len(fracs))):
            if frac <= 0:
                start += frac
                continue
            theta1 = start * 360
            theta2 = (start + frac) * 360
            wedge = Wedge(
                (x, y),
                radius,
                theta1,
                theta2,
                facecolor=palette_hex[cidx],
                ec="#222",
                lw=0.7,
                alpha=1.0,
            )
            ax.add_patch(wedge)
            start += frac
    if pies_drawn == 0:
        print("WARNING: No pies drawn. Check node_layer_membership and layer data.")
        # Make node overlay semi-transparent so pies are visible
        nx.draw_networkx_nodes(
            flattened,
            pos_dict,
            ax=ax,
            node_size=[s * 6 for s in sizes],
            node_color="#f5f5f5",
            edgecolors="#222",
            linewidths=1.2,
        )
    for idx_l, layer in enumerate(layers_order):
        ax.scatter([], [], c=palette_hex[idx_l], label=layer, s=80)
    ax.legend(ncol=2, fontsize=10, frameon=False, title="Layers")
    ax.set_title(
        "Augmented flattened network: pies show layer membership; size ~ degree"
    )
    ax.axis("off")
    out = RESULTS_DIR / "augmented_flattened.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def community_detection_and_plot(flattened, pos, combo_name):
    # Use greedy modularity communities from networkx
    from networkx.algorithms.community import greedy_modularity_communities
    from networkx.algorithms.community.quality import modularity

    communities = list(greedy_modularity_communities(flattened))
    # compute modularity score
    try:
        mod_score = modularity(flattened, communities)
    except Exception:
        mod_score = None
    # assign color per community
    cmap = sns.color_palette("husl", n_colors=max(2, len(communities)))
    node_color = {}
    for idx, comm in enumerate(communities):
        for n in comm:
            node_color[n] = cmap[idx]

    plt.figure(figsize=(11, 7))
    degrees = dict(flattened.degree())
    sizes = [140 + (degrees.get(n, 0) ** 1.1) * 18 for n in flattened.nodes()]
    nx.draw_networkx_nodes(
        flattened,
        pos,
        node_size=sizes,
        node_color=[node_color.get(n, "#cccccc") for n in flattened.nodes()],
        edgecolors="black",
        linewidths=0.7,
    )
    # Draw node ID labels

    labels = {n: str(n) for n in flattened.nodes()}
    nx.draw_networkx_labels(flattened, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(flattened, pos, alpha=0.4)
    title = f"Communities (greedy modularity) - {combo_name}"
    if mod_score is not None:
        title += f"  |  modularity={mod_score:.3f}"
    plt.title(title)
    plt.axis("off")
    out = RESULTS_DIR / f"communities_{combo_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_layer_vs_flattened(layer_networks, flattened, pos, layer_name):
    """Side-by-side comparison of one layer and the flattened network to highlight differences."""
    if layer_name not in layer_networks:
        return
    g = layer_networks[layer_name]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # layer
    axes[0].set_title(f"Layer: {layer_name}")
    nx.draw_networkx_edges(g, pos, ax=axes[0], alpha=0.25)
    nx.draw_networkx_nodes(g, pos, node_size=80, node_color="#1f77b4", ax=axes[0])
    axes[0].axis("off")

    # flattened
    axes[1].set_title("Flattened (all layers)")
    nx.draw_networkx_edges(flattened, pos, ax=axes[1], alpha=0.15)
    nx.draw_networkx_nodes(
        flattened, pos, node_size=80, node_color="#ff7f0e", ax=axes[1]
    )
    axes[1].axis("off")

    out = RESULTS_DIR / f"layer_vs_flattened_{layer_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_stacked_layers(layer_networks, base_pos=None, vgap=0.35):
    """Create a stacked (2.5D) visualization by vertically offsetting each layer's layout.
    This helps compare node positions across layers."""
    layers = list(layer_networks.keys())
    # take base positions from flattened network or compute common spring layout
    if base_pos is None:
        flattened = merge_flattened(layer_networks, weighted=False)
        base_pos = compute_shared_layout(flattened)

    fig, ax = plt.subplots(figsize=(10, 3 + 0.6 * len(layers)))
    palette = sns.color_palette("husl", n_colors=len(layers))

    # for each layer, offset y by index
    for idx, layer in enumerate(layers):
        offset = idx * vgap
        g = layer_networks[layer]
        # compute shifted positions
        pos_shift = {n: (x, y + offset) for n, (x, y) in base_pos.items()}
        nx.draw_networkx_edges(g, pos_shift, ax=ax, alpha=0.25, edge_color=palette[idx])
        nx.draw_networkx_nodes(
            g,
            pos_shift,
            node_size=40,
            node_color=[
                palette[idx] if g.degree(n) > 0 else "#eeeeee" for n in g.nodes()
            ],
            ax=ax,
        )
        # label the layer on the left
        ax.text(
            -1.05,
            offset,
            layer,
            fontsize=10,
            color=palette[idx],
            transform=ax.transData,
        )

    ax.axis("off")
    plt.title("Stacked layers (2.5D-ish)")
    out = RESULTS_DIR / "stacked_layers.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def plot_degree_distributions(layer_networks):
    """Plot degree distributions (CDF + hist) for each layer."""
    plt.figure(figsize=(10, 6))
    layers = list(layer_networks.keys())
    palette = sns.color_palette("husl", n_colors=len(layers))
    for idx, layer in enumerate(layers):
        degrees = [d for _, d in layer_networks[layer].degree()]
        if not degrees:
            continue
        # plot KDE / histogram
        sns.kdeplot(degrees, label=layer, color=palette[idx], fill=False)

    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.title("Degree distributions per layer (KDE)")
    plt.legend(frameon=False)
    out = RESULTS_DIR / "degree_distributions_kde.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_layer_overlap_matrix(layer_networks):
    """Compute pairwise edge-overlap (Jaccard on edge sets) between layers and plot as heatmap."""
    layers = list(layer_networks.keys())
    n = len(layers)
    M = np.zeros((n, n))
    edge_sets = {}
    for i, l in enumerate(layers):
        edges = set(tuple(sorted(e)) for e in layer_networks[l].edges())
        edge_sets[l] = edges

    for i, a in enumerate(layers):
        for j, b in enumerate(layers):
            A = edge_sets[a]
            B = edge_sets[b]
            if not A and not B:
                s = 0.0
            else:
                s = len(A & B) / len(A | B) if len(A | B) > 0 else 0.0
            M[i, j] = s

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        M, xticklabels=layers, yticklabels=layers, annot=True, fmt=".2f", cmap="viridis"
    )
    plt.title("Layer edge overlap (Jaccard)")
    out = RESULTS_DIR / "layer_overlap_jaccard.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_ego_across_layers(layer_networks, nodes_df, pos, top_k=4):
    """For top_k actors by total degree across layers, plot their ego networks side-by-side for each layer."""
    # compute total degree
    total_deg = defaultdict(int)
    for g in layer_networks.values():
        for n, d in g.degree():
            total_deg[n] += d

    top_nodes = sorted(total_deg.keys(), key=lambda n: total_deg[n], reverse=True)[
        :top_k
    ]
    layers = list(layer_networks.keys())

    for actor in top_nodes:
        fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4))
        if len(layers) == 1:
            axes = [axes]
        for ax, layer in zip(axes, layers):
            g = layer_networks[layer]
            ego = nx.ego_graph(g, actor, radius=1)
            ax.set_title(f"{actor} - {layer}")
            # use positions from global pos but fallback to spring for missing
            ego_pos = {n: pos[n] for n in ego.nodes() if n in pos}
            miss = [n for n in ego.nodes() if n not in ego_pos]
            if miss:
                ext = nx.spring_layout(ego.subgraph(miss), seed=42)
                ego_pos.update(ext)
            nx.draw_networkx_edges(ego, ego_pos, ax=ax, alpha=0.4)
            nx.draw_networkx_nodes(
                ego, ego_pos, ax=ax, node_size=120, node_color="#1f77b4"
            )
            nx.draw_networkx_labels(ego, ego_pos, font_size=8)
            ax.axis("off")

        out = RESULTS_DIR / f"ego_actor_{actor}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()


def plot_progressive_layer_growth(layer_networks, pos):
    """Show progressive growth by adding layers one by one."""
    layers = list(layer_networks.keys())
    n_layers = len(layers)

    # Create a grid of subplots - make it bigger and wider
    cols = min(2, n_layers)  # Reduce columns for more space
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(8 * cols, 6 * rows)
    )  # Increased figure size

    # Handle single subplot case
    if n_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Create spread out positions by scaling the original layout
    spread_pos = {}
    for node, (x, y) in pos.items():
        spread_pos[node] = (x * 1.5, y * 1.5)  # Scale positions for more spread

    for i in range(n_layers):
        ax = axes[i]

        # Build cumulative network with layers up to index i
        cumulative_layers = layers[: i + 1]
        cumulative_networks = {l: layer_networks[l] for l in cumulative_layers}
        cumulative_flat = merge_flattened(cumulative_networks, weighted=False)

        # Node colors by degree
        degrees = dict(cumulative_flat.degree())
        max_deg = max(degrees.values()) if degrees.values() else 1
        node_colors = [degrees.get(n, 0) / max_deg for n in cumulative_flat.nodes()]

        # Node sizes proportional to degree - make them bigger
        sizes = [100 + (degrees.get(n, 0) ** 1.1) * 25 for n in cumulative_flat.nodes()]

        # Draw network with spread out positions
        nx.draw_networkx_edges(
            cumulative_flat, spread_pos, ax=ax, alpha=0.3, edge_color="#666"
        )
        nx.draw_networkx_nodes(
            cumulative_flat,
            spread_pos,
            ax=ax,
            node_size=sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            edgecolors="black",
            linewidths=0.7,  # Thicker outlines
            vmin=0,
            vmax=1,
        )

        # Add node labels for high degree nodes
        high_deg_nodes = {
            n: str(n) for n in cumulative_flat.nodes() if degrees.get(n, 0) >= 3
        }
        nx.draw_networkx_labels(
            cumulative_flat, spread_pos, labels=high_deg_nodes, font_size=9, ax=ax
        )

        # Title with layer info
        layer_list = " + ".join(cumulative_layers)
        edges_count = cumulative_flat.number_of_edges()
        nodes_count = cumulative_flat.number_of_nodes()
        ax.set_title(
            f"Step {i+1}: {layer_list}\n{nodes_count} nodes, {edges_count} edges",
            fontsize=11,
        )
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Progressive Layer Growth", fontsize=16, y=0.98)
    plt.tight_layout()
    out = RESULTS_DIR / "progressive_layer_growth.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("Task07: multilayer visualizations and simple community detection")
    layers_df, nodes_df, edges_df = load_data()
    layer_networks, layer_list = build_layer_graphs(layers_df, nodes_df, edges_df)

    # Flatten network for shared layout
    flattened_unweighted = merge_flattened(layer_networks, weighted=False)
    pos = compute_shared_layout(flattened_unweighted)

    print(" - plotting layer slices...")
    plot_layer_slices(layer_networks, pos)

    print(" - plotting augmented flattened network...")
    membership = node_layer_membership(layer_networks)
    plot_augmented_flattened(flattened_unweighted, layer_networks, membership, pos)

    print(" - community detection on flattened and on two-layer combos...")
    # community on full flattened
    community_detection_and_plot(flattened_unweighted, pos, "all_layers")

    print(" - plotting progressive layer growth...")
    plot_progressive_layer_growth(layer_networks, pos)

    # two different combinations: ('facebook','work') and ('coauthor','leisure') if exist
    combos = [("facebook", "work"), ("coauthor", "leisure")]
    for combo in combos:
        existing = [l for l in combo if l in layer_networks]
        if not existing:
            continue
        # build flattened of selected layers
        sub = {l: layer_networks[l] for l in existing}
        flat_sub = merge_flattened(sub, weighted=False)
        # keep pos for nodes that exist; for missing nodes, spring_layout locally
        sub_pos = {n: pos[n] for n in pos if n in flat_sub.nodes()}
        if len(sub_pos) < flat_sub.number_of_nodes():
            # extend layout
            extra = [n for n in flat_sub.nodes() if n not in sub_pos]
            ext_pos = nx.spring_layout(flat_sub.subgraph(extra), seed=42)
            # shift keys
            for k, v in ext_pos.items():
                sub_pos[k] = v

        community_detection_and_plot(flat_sub, sub_pos, "_".join(existing))

    print(f"Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
