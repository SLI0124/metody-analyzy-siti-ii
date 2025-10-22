from pathlib import Path
import random
from collections import defaultdict, Counter
from itertools import combinations
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)
np.random.seed(42)

plt.style.use("default")
sns.set_palette("husl")

DATA_DIR = Path("../data/CS-Aarhus_Multiplex_Social/CS-Aarhus_Multiplex_Social/Dataset")
RESULTS_DIR = Path("../results/task06")
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


def build_multilayer_network(edges_df, layers_df, nodes_df):
    layer_names = dict(zip(layers_df["layerID"], layers_df["layerLabel"]))
    layer_networks = {}

    for layer_id in layers_df["layerID"]:
        layer_edges = edges_df[edges_df["layerID"] == layer_id]
        graph = nx.Graph()

        for node_id in nodes_df["nodeID"]:
            graph.add_node(node_id)

        for _, row in layer_edges.iterrows():
            graph.add_edge(row["nodeID1"], row["nodeID2"], weight=row["weight"])

        layer_networks[layer_names[layer_id]] = graph

    all_actors = set(nodes_df["nodeID"])
    return layer_networks, layer_names, all_actors


def get_neighbors(actor, layers, layer_networks):
    if isinstance(layers, str):
        layers = [layers]

    all_neighbors = set()
    for layer in layers:
        if layer in layer_networks and actor in layer_networks[layer]:
            all_neighbors.update(layer_networks[layer].neighbors(actor))

    return all_neighbors


def calculate_relevance(actor, layers, layer_networks, all_layers):
    """Task 1: Standard relevance - relevance(a, L) = |neighbors(a, L)| / |neighbors(a, L)|"""
    neighbors_in_layers = get_neighbors(actor, layers, layer_networks)
    neighbors_in_all = get_neighbors(actor, all_layers, layer_networks)

    if len(neighbors_in_all) == 0:
        return 0.0

    return len(neighbors_in_layers) / len(neighbors_in_all)


def calculate_exclusive_relevance(actor, layers, layer_networks, all_layers):
    """
    Task 1: Exclusive layer relevance -
    exclusive_relevance(a, L) = |xneighborhood(a, L)| / |neighbors(a, L)|
    """
    if isinstance(layers, str):
        layers = [layers]

    layers_set = set(layers)
    complement_layers = [l for l in all_layers if l not in layers_set]

    neighbors_in_layers = get_neighbors(actor, layers, layer_networks)
    neighbors_in_complement = get_neighbors(actor, complement_layers, layer_networks)
    neighbors_in_all = get_neighbors(actor, all_layers, layer_networks)

    # Exclusive neighborhood: neighbors only in specified layers
    exclusive_neighbors = neighbors_in_layers - neighbors_in_complement

    if len(neighbors_in_all) == 0:
        return 0.0

    return len(exclusive_neighbors) / len(neighbors_in_all)


def perform_layered_random_walk(
    layer_networks, all_actors, num_walks=10000, walk_length=1000, interlayer_prob=0.2
):
    """Random walk on multilayer network"""
    layer_names = list(layer_networks.keys())
    visit_counts = Counter()

    for _ in range(num_walks):
        current_actor = random.choice(list(all_actors))
        current_layer = random.choice(layer_names)

        if current_actor not in layer_networks[current_layer]:
            for layer in layer_names:
                if (
                    current_actor in layer_networks[layer]
                    and layer_networks[layer].degree(current_actor) > 0
                ):
                    current_layer = layer
                    break
            else:
                continue

        for _ in range(walk_length):
            visit_counts[(current_actor, current_layer)] += 1

            if random.random() < interlayer_prob:
                # Switch to different layer with same actor
                available_layers = [
                    l
                    for l in layer_names
                    if l != current_layer
                    and current_actor in layer_networks[l]
                    and layer_networks[l].degree(current_actor) > 0
                ]
                if available_layers:
                    current_layer = random.choice(available_layers)
            else:
                # Move to neighboring actor within same layer
                current_graph = layer_networks[current_layer]
                if (
                    current_actor in current_graph
                    and current_graph.degree(current_actor) > 0
                ):
                    neighbors_list = list(current_graph.neighbors(current_actor))
                    if neighbors_list:
                        current_actor = random.choice(neighbors_list)

    return visit_counts


def calculate_occupation_centrality(visit_counts, all_actors):
    """
    Task 2: Occupation centrality -
    occupation_centrality(a) = P(random walker is found on any node corresponding to a)
    """
    total_visits = sum(visit_counts.values())
    if total_visits == 0:
        return {actor: 0.0 for actor in all_actors}

    actor_visits = defaultdict(int)
    for (actor, _), count in visit_counts.items():
        actor_visits[actor] += count

    occupation_centralities = {}
    for actor in all_actors:
        occupation_centralities[actor] = actor_visits[actor] / total_visits

    return occupation_centralities


def merge_layer_networks_flattening(layer_networks, all_actors, weighted=False):
    """
    Task 3: Network flattening -
    V_f = {a | (a, l) ∈ V}, E_f = {(a_i, a_j) | {(a_i, l_q), (a_j, l_r)} ∈ E}
    """
    flattened = nx.Graph()
    flattened.add_nodes_from(all_actors)

    if weighted:
        # Weighted flattening - sum weights across layers
        edge_weights = defaultdict(float)
        for graph in layer_networks.values():
            for edge in graph.edges(data=True):
                node1, node2, data = edge
                weight = data.get("weight", 1.0)
                edge_key = tuple(sorted([node1, node2]))
                edge_weights[edge_key] += weight

        for (node1, node2), weight in edge_weights.items():
            flattened.add_edge(node1, node2, weight=weight)
    else:
        # Unweighted flattening
        for graph in layer_networks.values():
            for edge in graph.edges():
                flattened.add_edge(edge[0], edge[1])

    return flattened


def compute_degree_analysis(layer_networks, all_actors, all_layers):
    """Compute degree analysis for each layer and overall"""
    degree_analysis = []

    # Analyze each layer separately
    for layer_name, graph in layer_networks.items():
        degrees = dict(graph.degree())
        total_degree = sum(degrees.values())

        for actor in all_actors:
            degree = degrees.get(actor, 0)
            degree_prob = degree / total_degree if total_degree > 0 else 0.0

            degree_analysis.append(
                {
                    "actor": actor,
                    "layer": layer_name,
                    "degree": degree,
                    "degree_probability": degree_prob,
                }
            )

    # Overall multilayer degree
    for actor in all_actors:
        total_degree = sum(
            layer_networks[layer].degree(actor) if actor in layer_networks[layer] else 0
            for layer in all_layers
        )

        overall_total = sum(
            sum(dict(graph.degree()).values()) for graph in layer_networks.values()
        )
        overall_prob = total_degree / overall_total if overall_total > 0 else 0.0

        degree_analysis.append(
            {
                "actor": actor,
                "layer": "ALL_LAYERS",
                "degree": total_degree,
                "degree_probability": overall_prob,
            }
        )

    return pd.DataFrame(degree_analysis)


def compare_rw_with_degree_probabilities(visit_counts, degree_analysis, all_actors):
    """Compare random walk probabilities with degree probabilities"""

    # Calculate RW probabilities per layer
    rw_probs_by_layer = {}
    for layer in degree_analysis["layer"].unique():
        if layer != "ALL_LAYERS":
            layer_visits = {actor: 0 for actor in all_actors}
            total_layer_visits = 0

            for (actor, visit_layer), count in visit_counts.items():
                if visit_layer == layer:
                    layer_visits[actor] += count
                    total_layer_visits += count

            if total_layer_visits > 0:
                rw_probs_by_layer[layer] = {
                    actor: visits / total_layer_visits
                    for actor, visits in layer_visits.items()
                }
            else:
                rw_probs_by_layer[layer] = {actor: 0.0 for actor in all_actors}

    # Overall RW probabilities
    total_visits = sum(visit_counts.values())
    actor_visits = defaultdict(int)
    for (actor, _), count in visit_counts.items():
        actor_visits[actor] += count

    overall_rw_probs = {
        actor: visits / total_visits if total_visits > 0 else 0.0
        for actor, visits in actor_visits.items()
    }

    comparison_data = []

    for _, row in degree_analysis.iterrows():
        actor = row["actor"]
        layer = row["layer"]
        degree = row["degree"]
        degree_prob = row["degree_probability"]

        if layer == "ALL_LAYERS":
            rw_prob = overall_rw_probs.get(actor, 0.0)
        else:
            rw_prob = rw_probs_by_layer.get(layer, {}).get(actor, 0.0)

        comparison_data.append(
            {
                "actor": actor,
                "layer": layer,
                "degree": degree,
                "degree_probability": degree_prob,
                "random_walk_probability": rw_prob,
                "prob_difference": abs(degree_prob - rw_prob),
            }
        )

    return pd.DataFrame(comparison_data)


def compute_relevance_measures(results, layer_networks, all_layers):
    # Task 1: Relevance measures for individual layers
    print("  - Standard relevance for individual layers...")
    for layer in all_layers:
        col_name = f"relevance_{layer}"
        results[col_name] = results["nodeID"].apply(
            lambda actor, l=layer: calculate_relevance(
                actor, [l], layer_networks, all_layers
            )
        )

    print("  - Exclusive relevance for individual layers...")
    for layer in all_layers:
        col_name = f"exclusive_relevance_{layer}"
        results[col_name] = results["nodeID"].apply(
            lambda actor, l=layer: calculate_exclusive_relevance(
                actor, [l], layer_networks, all_layers
            )
        )

    # Relevance for layer combinations
    print("  - Relevance for layer combinations...")
    layer_combinations = list(combinations(all_layers, 2))
    for layer_combo in layer_combinations:
        combo_name = "_".join(layer_combo)
        results[f"relevance_{combo_name}"] = results["nodeID"].apply(
            lambda actor, lc=layer_combo: calculate_relevance(
                actor, lc, layer_networks, all_layers
            )
        )
        results[f"exclusive_relevance_{combo_name}"] = results["nodeID"].apply(
            lambda actor, lc=layer_combo: calculate_exclusive_relevance(
                actor, lc, layer_networks, all_layers
            )
        )

    return results


def compute_network_statistics(
    layer_networks, flattened_unweighted, flattened_weighted, all_actors, all_layers
):
    layer_densities = [nx.density(g) for g in layer_networks.values()]
    avg_layer_density = sum(layer_densities) / len(layer_densities)

    total_degree_sum = 0
    total_weight_sum = 0
    total_edges = 0

    for g in layer_networks.values():
        total_degree_sum += sum(dict(g.degree()).values())
        total_edges += g.number_of_edges()
        total_weight_sum += sum(
            data.get("weight", 1.0) for _, _, data in g.edges(data=True)
        )

    avg_degree_all_layers = total_degree_sum / (len(all_actors) * len(all_layers))
    avg_weight_original = total_weight_sum / total_edges if total_edges > 0 else 1.0

    original_stats = {
        "total_nodes": len(all_actors),
        "total_edges": total_edges,
        "layers": len(all_layers),
        "avg_layer_density": avg_layer_density,
        "avg_degree_all_layers": avg_degree_all_layers,
        "avg_weight": avg_weight_original,
    }

    unweighted_stats = {
        "nodes": flattened_unweighted.number_of_nodes(),
        "edges": flattened_unweighted.number_of_edges(),
        "density": nx.density(flattened_unweighted),
        "avg_degree": sum(dict(flattened_unweighted.degree()).values())
        / flattened_unweighted.number_of_nodes(),
        "avg_weight": 1.0,  # Unweighted networks have weight 1.0 by definition
    }

    weighted_stats = {
        "nodes": flattened_weighted.number_of_nodes(),
        "edges": flattened_weighted.number_of_edges(),
        "density": nx.density(flattened_weighted),
        "avg_degree": sum(dict(flattened_weighted.degree()).values())
        / flattened_weighted.number_of_nodes(),
        "avg_weight": (
            sum(
                data.get("weight", 1.0)
                for _, _, data in flattened_weighted.edges(data=True)
            )
            / flattened_weighted.number_of_edges()
            if flattened_weighted.number_of_edges() > 0
            else 0
        ),
    }

    return original_stats, unweighted_stats, weighted_stats


def compute_all_measures(layer_networks, nodes_df):
    all_layers = list(layer_networks.keys())
    all_actors = set(nodes_df["nodeID"])

    print("Computing relevance measures...")
    results = pd.DataFrame(
        {"nodeID": nodes_df["nodeID"], "nodeLabel": nodes_df["nodeLabel"]}
    )

    # Task 1: Relevance measures
    results = compute_relevance_measures(results, layer_networks, all_layers)

    # Degree analysis
    print("Computing degree analysis...")
    degree_analysis = compute_degree_analysis(layer_networks, all_actors, all_layers)

    # Task 2: Random walk and occupation centrality
    print("Computing random walk and occupation centrality...")
    visit_counts = perform_layered_random_walk(
        layer_networks, all_actors, num_walks=5000, walk_length=500
    )
    occupation_centralities = calculate_occupation_centrality(visit_counts, all_actors)

    results["occupation_centrality"] = results["nodeID"].map(occupation_centralities)

    # Compare RW with degree probabilities
    print("Comparing random walk with degree probabilities...")
    rw_degree_comparison = compare_rw_with_degree_probabilities(
        visit_counts, degree_analysis, all_actors
    )

    # Task 3: Network flattening (both weighted and unweighted)
    print("Computing flattened networks...")
    flattened_unweighted = merge_layer_networks_flattening(
        layer_networks, all_actors, weighted=False
    )
    flattened_weighted = merge_layer_networks_flattening(
        layer_networks, all_actors, weighted=True
    )

    original_stats, unweighted_stats, weighted_stats = compute_network_statistics(
        layer_networks, flattened_unweighted, flattened_weighted, all_actors, all_layers
    )

    return (
        results,
        degree_analysis,
        rw_degree_comparison,
        flattened_unweighted,
        flattened_weighted,
        original_stats,
        unweighted_stats,
        weighted_stats,
        visit_counts,
    )


def save_results(
    results,
    degree_analysis,
    rw_degree_comparison,
    flattened_unweighted,
    flattened_weighted,
    original_stats,
    unweighted_stats,
    weighted_stats,
    visit_counts,
    nodes_df,
):

    # Save relevance measures
    results.to_csv(RESULTS_DIR / "relevance_measures.csv", index=False)
    print(f"Saved relevance measures to {RESULTS_DIR / 'relevance_measures.csv'}")

    # Save degree analysis
    degree_analysis.to_csv(RESULTS_DIR / "degree_analysis.csv", index=False)
    print(f"Saved degree analysis to {RESULTS_DIR / 'degree_analysis.csv'}")

    # Save RW vs degree probability comparison
    rw_degree_comparison.to_csv(RESULTS_DIR / "rw_degree_comparison.csv", index=False)
    print(f"Saved RW-degree comparison to {RESULTS_DIR / 'rw_degree_comparison.csv'}")

    # Save unweighted flattened network as CSV for Gephi
    edges_list = []
    for edge in flattened_unweighted.edges():
        edges_list.append({"Source": edge[0], "Target": edge[1]})

    edges_df = pd.DataFrame(edges_list)
    edges_df.to_csv(RESULTS_DIR / "flattened_unweighted_edgelist.csv", index=False)
    print(
        f"Saved unweighted network to {RESULTS_DIR / 'flattened_unweighted_edgelist.csv'}"
    )

    # Save weighted flattened network as CSV for Gephi
    weighted_edges_list = []
    for edge in flattened_weighted.edges(data=True):
        node1, node2, data = edge
        weight = data.get("weight", 1.0)
        weighted_edges_list.append({"Source": node1, "Target": node2, "Weight": weight})

    weighted_edges_df = pd.DataFrame(weighted_edges_list)
    weighted_edges_df.to_csv(
        RESULTS_DIR / "flattened_weighted_edgelist.csv", index=False
    )
    print(
        f"Saved weighted network to {RESULTS_DIR / 'flattened_weighted_edgelist.csv'}"
    )

    # Save nodes with labels
    nodes_df_gephi = nodes_df[["nodeID", "nodeLabel"]].copy()
    nodes_df_gephi.columns = ["Id", "Label"]
    nodes_df_gephi.to_csv(RESULTS_DIR / "flattened_network_nodes.csv", index=False)
    print(f"Saved nodes to {RESULTS_DIR / 'flattened_network_nodes.csv'}")

    # Save network statistics comparison
    stats_comparison = pd.DataFrame(
        [
            {
                "metric": "nodes",
                "original": original_stats["total_nodes"],
                "unweighted_flattened": unweighted_stats["nodes"],
                "weighted_flattened": weighted_stats["nodes"],
            },
            {
                "metric": "edges",
                "original": original_stats["total_edges"],
                "unweighted_flattened": unweighted_stats["edges"],
                "weighted_flattened": weighted_stats["edges"],
            },
            {
                "metric": "layers",
                "original": original_stats["layers"],
                "unweighted_flattened": 1,
                "weighted_flattened": 1,
            },
            {
                "metric": "density",
                "original": original_stats["avg_layer_density"],
                "unweighted_flattened": unweighted_stats["density"],
                "weighted_flattened": weighted_stats["density"],
            },
            {
                "metric": "avg_degree",
                "original": original_stats["avg_degree_all_layers"],
                "unweighted_flattened": unweighted_stats["avg_degree"],
                "weighted_flattened": weighted_stats["avg_degree"],
            },
            {
                "metric": "avg_weight",
                "original": original_stats["avg_weight"],
                "unweighted_flattened": unweighted_stats["avg_weight"],
                "weighted_flattened": weighted_stats["avg_weight"],
            },
        ]
    )
    stats_comparison.to_csv(RESULTS_DIR / "network_comparison.csv", index=False)

    # Save random walk details
    walk_details = pd.DataFrame(
        [
            {"actor": actor, "layer": layer, "visit_count": count}
            for (actor, layer), count in visit_counts.items()
        ]
    )
    walk_details.to_csv(RESULTS_DIR / "random_walk_visits.csv", index=False)

    print(f"All results saved to {RESULTS_DIR}")

    print(f"Total actors: {original_stats['total_nodes']}")
    print(f"Total layers: {original_stats['layers']}")
    print(f"Total edges across all layers: {original_stats['total_edges']}")
    print(f"Unweighted flattened network edges: {unweighted_stats['edges']}")
    print(f"Weighted flattened network edges: {weighted_stats['edges']}")
    print(f"Unweighted network density: {unweighted_stats['density']:.4f}")
    print(f"Weighted network density: {weighted_stats['density']:.4f}")
    print(f"Average weight in weighted network: {weighted_stats['avg_weight']:.2f}")


def main():
    print("=== TASK 6: MULTILAYER SOCIAL NETWORKS - EXTENDED ANALYSIS ===")
    print("Loading CS-Aarhus multilayer social network...")

    layers_df, nodes_df, edges_df = load_data()
    layer_networks, layer_names, all_actors = build_multilayer_network(
        edges_df, layers_df, nodes_df
    )

    print(
        f"Loaded network with {len(all_actors)} actors and {len(layer_networks)} layers:"
    )
    for layer_id, layer_name in layer_names.items():
        edges_count = layer_networks[layer_name].number_of_edges()
        print(f"  - Layer {layer_id} ({layer_name}): {edges_count} edges")

    (
        results,
        degree_analysis,
        rw_degree_comparison,
        flattened_unweighted,
        flattened_weighted,
        original_stats,
        unweighted_stats,
        weighted_stats,
        visit_counts,
    ) = compute_all_measures(layer_networks, nodes_df)

    save_results(
        results,
        degree_analysis,
        rw_degree_comparison,
        flattened_unweighted,
        flattened_weighted,
        original_stats,
        unweighted_stats,
        weighted_stats,
        visit_counts,
        nodes_df,
    )

    print("\n=== TASK COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
