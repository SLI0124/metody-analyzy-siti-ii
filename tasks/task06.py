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


def merge_layer_networks_flattening(layer_networks, all_actors):
    """
    Task 3: Network flattening -
    V_f = {a | (a, l) ∈ V}, E_f = {(a_i, a_j) | {(a_i, l_q), (a_j, l_r)} ∈ E}
    """
    flattened = nx.Graph()
    flattened.add_nodes_from(all_actors)

    for graph in layer_networks.values():
        for edge in graph.edges():
            flattened.add_edge(edge[0], edge[1])

    return flattened


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
    layer_networks, flattened_network, all_actors, all_layers
):
    layer_densities = [nx.density(g) for g in layer_networks.values()]
    avg_layer_density = sum(layer_densities) / len(layer_densities)

    total_degree_sum = 0
    for g in layer_networks.values():
        total_degree_sum += sum(dict(g.degree()).values())
    avg_degree_all_layers = total_degree_sum / (len(all_actors) * len(all_layers))

    original_stats = {
        "total_nodes": len(all_actors),
        "total_edges": sum(g.number_of_edges() for g in layer_networks.values()),
        "layers": len(all_layers),
        "avg_layer_density": avg_layer_density,
        "avg_degree_all_layers": avg_degree_all_layers,
    }

    flattened_stats = {
        "nodes": flattened_network.number_of_nodes(),
        "edges": flattened_network.number_of_edges(),
        "density": nx.density(flattened_network),
        "avg_degree": sum(dict(flattened_network.degree()).values())
        / flattened_network.number_of_nodes(),
    }

    return original_stats, flattened_stats


def compute_all_measures(layer_networks, nodes_df):
    all_layers = list(layer_networks.keys())
    all_actors = set(nodes_df["nodeID"])

    print("Computing relevance measures...")
    results = pd.DataFrame(
        {"nodeID": nodes_df["nodeID"], "nodeLabel": nodes_df["nodeLabel"]}
    )

    results = compute_relevance_measures(results, layer_networks, all_layers)

    # Task 2: Random walk and occupation centrality
    print("Computing random walk and occupation centrality...")
    visit_counts = perform_layered_random_walk(
        layer_networks, all_actors, num_walks=5000, walk_length=500
    )
    occupation_centralities = calculate_occupation_centrality(visit_counts, all_actors)

    results["occupation_centrality"] = results["nodeID"].map(occupation_centralities)

    # Task 3: Network flattening
    print("Computing flattened network...")
    flattened_network = merge_layer_networks_flattening(layer_networks, all_actors)

    original_stats, flattened_stats = compute_network_statistics(
        layer_networks, flattened_network, all_actors, all_layers
    )

    return results, flattened_network, original_stats, flattened_stats, visit_counts


def save_results(results, flattened_network, stats_data, visit_counts, nodes_df):
    original_stats, flattened_stats = stats_data

    results.to_csv(RESULTS_DIR / "relevance_measures.csv", index=False)
    print(f"Saved relevance measures to {RESULTS_DIR / 'relevance_measures.csv'}")

    # Save flattened network as CSV, i.e. edgelist for Gephi
    edges_list = []
    for edge in flattened_network.edges():
        edges_list.append({"Source": edge[0], "Target": edge[1]})

    edges_df = pd.DataFrame(edges_list)
    edges_df.to_csv(RESULTS_DIR / "flattened_network_edgelist.csv", index=False)
    print(
        f"Saved flattened network to {RESULTS_DIR / 'flattened_network_edgelist.csv'}"
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
                "flattened": flattened_stats["nodes"],
            },
            {
                "metric": "edges",
                "original": original_stats["total_edges"],
                "flattened": flattened_stats["edges"],
            },
            {"metric": "layers", "original": original_stats["layers"], "flattened": 1},
            {
                "metric": "density",
                "original": original_stats["avg_layer_density"],
                "flattened": flattened_stats["density"],
            },
            {
                "metric": "avg_degree",
                "original": original_stats["avg_degree_all_layers"],
                "flattened": flattened_stats["avg_degree"],
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

    print(f"All results saved to {RESULTS_DIR}]\n")

    print(f"Total actors: {original_stats['total_nodes']}")
    print(f"Total layers: {original_stats['layers']}")
    print(f"Total edges across all layers: {original_stats['total_edges']}")
    print(f"Flattened network edges: {flattened_stats['edges']}")
    print(f"Flattened network density: {flattened_stats['density']:.4f}")
    print(f"Average degree in flattened network: {flattened_stats['avg_degree']:.2f}\n")


def main():
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

    results, flattened_network, original_stats, flattened_stats, visit_counts = (
        compute_all_measures(layer_networks, nodes_df)
    )

    save_results(
        results,
        flattened_network,
        (original_stats, flattened_stats),
        visit_counts,
        nodes_df,
    )


if __name__ == "__main__":
    main()
