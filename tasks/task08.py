import random
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

DATA_DIR = Path("../data/CS-Aarhus_Multiplex_Social/CS-Aarhus_Multiplex_Social/Dataset")
RESULTS_DIR = Path("../results/task08")

# Create output directories
dirs = [
    RESULTS_DIR / name
    for name in ["association_rules", "all_layers", "comparison", "summary"]
]
RULES_DIR, ALL_LAYERS_DIR, COMPARISON_DIR, SUMMARY_DIR = dirs

for d in [RESULTS_DIR] + dirs:
    d.mkdir(parents=True, exist_ok=True)


def load_multilayer_data():
    layers_df = pd.read_csv(DATA_DIR / "CS-Aarhus_layers.txt", sep=" ")
    nodes_df = pd.read_csv(DATA_DIR / "CS-Aarhus_nodes.txt", sep=" ")
    edges_df = pd.read_csv(
        DATA_DIR / "CS-Aarhus_multiplex.edges",
        sep=" ",
        names=["layerID", "nodeID1", "nodeID2", "weight"],
    )

    layer_names = dict(zip(layers_df["layerID"], layers_df["layerLabel"]))
    print(
        f"Loaded {len(layer_names)} layers, {len(nodes_df)} nodes, {len(edges_df)} edges"
    )

    return layers_df, nodes_df, edges_df, layer_names


def build_layer_graphs(nodes_df, edges_df, layer_names):
    layer_networks = {}

    for layer_id, layer_name in layer_names.items():
        G = nx.Graph()
        G.add_nodes_from(nodes_df["nodeID"].tolist())

        layer_edges = edges_df[edges_df["layerID"] == layer_id]
        for _, row in layer_edges.iterrows():
            G.add_edge(int(row["nodeID1"]), int(row["nodeID2"]), weight=row["weight"])

        layer_networks[layer_name] = G

    return layer_networks


def get_node_pair_layers(edges_df, layer_names):
    pair_layers = defaultdict(set)

    for _, row in edges_df.iterrows():
        u, v = int(row["nodeID1"]), int(row["nodeID2"])
        pair = tuple(sorted([u, v]))
        layer = layer_names[row["layerID"]]
        pair_layers[pair].add(layer)

    return pair_layers


def compute_layer_support(pair_layers, layer_names):
    all_layers = list(layer_names.values())
    n_pairs = len(pair_layers)
    support = {}

    for combination_size in range(1, len(all_layers) + 1):
        for layer_combination in combinations(all_layers, combination_size):
            count = sum(
                1
                for layers in pair_layers.values()
                if all(layer in layers for layer in layer_combination)
            )
            support[frozenset(layer_combination)] = count / n_pairs

    return support


def mine_association_rules(support, min_support=0.05, min_confidence=0.3, min_lift=1.0):
    rules = []
    frequent_sets = {k: v for k, v in support.items() if v >= min_support}

    # Generate rules
    for layers in frequent_sets.keys():
        if len(layers) < 2:
            continue

        # Try all possible splits
        for size in range(1, len(layers)):
            for antecedent_tuple in combinations(layers, size):
                antecedent = frozenset(antecedent_tuple)
                consequent = layers - antecedent

                if len(consequent) == 0:
                    continue

                supp_union = support.get(layers, 0)
                supp_ante = support.get(antecedent, 0)
                supp_cons = support.get(consequent, 0)

                if supp_ante == 0:
                    continue

                confidence = supp_union / supp_ante

                if supp_cons > 0:
                    lift = supp_union / (supp_ante * supp_cons)
                else:
                    lift = 0

                if confidence >= min_confidence and lift >= min_lift:
                    rules.append(
                        {
                            "antecedent": list(antecedent),
                            "consequent": list(consequent),
                            "support": supp_union,
                            "confidence": confidence,
                            "lift": lift,
                            "antecedent_support": supp_ante,
                            "consequent_support": supp_cons,
                        }
                    )

    return rules


def visualize_association_rules(rules, layer_names):
    if not rules:
        return

    df_rules = pd.DataFrame(rules)
    df_rules["antecedent_str"] = df_rules["antecedent"].apply(
        lambda x: "+".join(sorted(x))
    )
    df_rules["consequent_str"] = df_rules["consequent"].apply(
        lambda x: "+".join(sorted(x))
    )

    df_rules_sorted = df_rules.sort_values("lift", ascending=False)
    df_rules_sorted.to_csv(RULES_DIR / "rules_table.csv", index=False)

    # 1. Lift heatmap (pairwise layer associations)
    all_layers = list(layer_names.values())
    lift_matrix = pd.DataFrame(1.0, index=all_layers, columns=all_layers)

    for _, rule in df_rules.iterrows():
        if len(rule["antecedent"]) == 1 and len(rule["consequent"]) == 1:
            ante = rule["antecedent"][0]
            cons = rule["consequent"][0]
            lift_matrix.loc[ante, cons] = rule["lift"]
            lift_matrix.loc[cons, ante] = rule["lift"]  # Symmetric

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        lift_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=1.0,
        vmin=0,
        vmax=2.5,
        cbar_kws={"label": "Lift"},
    )
    plt.title(
        "Layer Association Lift Matrix\n(Lift > 1: Positive association)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(RULES_DIR / "lift_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Top rules bar chart
    top_rules = df_rules_sorted.head(20).copy()
    top_rules["rule_str"] = (
        top_rules["antecedent_str"] + " → " + top_rules["consequent_str"]
    )
    top_rules = top_rules.sort_values("confidence", ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(
        range(len(top_rules)),
        top_rules["confidence"],
        color=cm.viridis(top_rules["lift"] / top_rules["lift"].max()),
    )
    plt.yticks(range(len(top_rules)), top_rules["rule_str"], fontsize=9)
    plt.xlabel("Confidence", fontsize=12, fontweight="bold")
    plt.title("Top 20 Association Rules by Confidence", fontsize=14, fontweight="bold")
    plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), label="Lift", ax=plt.gca())
    plt.tight_layout()
    plt.savefig(RULES_DIR / "top_rules_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# SIMILARITY FUNCTIONS (from Task 03)
# ============================================================================


def common_neighbors_score(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))


def jaccard_score(G, u, v):
    preds = list(nx.jaccard_coefficient(G, [(u, v)]))
    return preds[0][2]


def adamic_adar_score(G, u, v):
    preds = list(nx.adamic_adar_index(G, [(u, v)]))
    return preds[0][2]


def preferential_attachment_score(G, u, v):
    preds = list(nx.preferential_attachment(G, [(u, v)]))
    return preds[0][2]


def resource_allocation_score(G, u, v):
    preds = list(nx.resource_allocation_index(G, [(u, v)]))
    return preds[0][2]


def cosine_similarity_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    len_u = len(neighbors_u)
    len_v = len(neighbors_v)
    if len_u == 0 or len_v == 0:
        return 0.0
    return intersection / (np.sqrt(len_u * len_v))


def sorensen_index_score(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    total = len(neighbors_u) + len(neighbors_v)
    if total == 0:
        return 0.0
    return 2 * intersection / total


def car_based_common_neighbors_score(G, u, v):
    if not all("community" in G.nodes[n] for n in G.nodes()):
        for idx, node in enumerate(G.nodes()):
            G.nodes[node]["community"] = idx % 2
    preds = list(nx.cn_soundarajan_hopcroft(G, [(u, v)]))
    return preds[0][2]


def compute_cross_layer_features(
    node_pair, target_layer, rules, pair_layers, all_layers
):
    u, v = node_pair
    pair = tuple(sorted([u, v]))
    current_layers = pair_layers.get(pair, set())
    features = {}

    # Association score based on rules pointing to target layer
    association_score = 0.0
    for rule in rules:
        antecedent_set = set(rule["antecedent"])
        consequent_set = set(rule["consequent"])

        # Check if rule applies
        if target_layer in consequent_set and antecedent_set.issubset(current_layers):
            association_score += rule["confidence"] * rule["lift"]

    features["association_score"] = association_score

    features["n_layers_connected"] = len(current_layers)

    # Binary features for each layer
    for layer in all_layers:
        if layer != target_layer:
            features[f"connected_on_{layer}"] = 1 if layer in current_layers else 0

    # Compute lift-based features
    relevant_lifts = []
    lift_weighted_score = 0.0
    for rule in rules:
        if (
            len(rule["antecedent"]) == 1
            and len(rule["consequent"]) == 1
            and rule["consequent"][0] == target_layer
        ):
            ante_layer = rule["antecedent"][0]
            if ante_layer in current_layers:
                relevant_lifts.append(rule["lift"])
                lift_weighted_score += rule["lift"] * rule["confidence"]

    features["avg_lift_to_target"] = np.mean(relevant_lifts) if relevant_lifts else 0.0

    return features


def build_multilayer_dataset(
    target_layer,
    layer_networks,
    rules,
    pair_layers,
    similarity_func,
    use_cross_layer=True,
    n_negatives=None,
    normalize=True,
):
    G = layer_networks[target_layer]
    all_layers = list(layer_networks.keys())

    # Positive samples: existing edges
    positives = [(u, v, 1) for u, v in G.edges()]

    # Negative samples: non-edges
    non_edges = list(nx.non_edges(G))
    if n_negatives is None:
        n_negatives = len(positives)
    negatives = random.sample(non_edges, min(n_negatives, len(non_edges)))
    negatives = [(u, v, 0) for u, v in negatives]

    # Compute features
    dataset = []
    similarity_scores = []

    for u, v, label in positives + negatives:
        features = {}

        # Within-layer feature
        similarity_score = similarity_func(G, u, v)
        similarity_scores.append(similarity_score)
        features["similarity_raw"] = similarity_score

        # Cross-layer features
        if use_cross_layer:
            cross_features = compute_cross_layer_features(
                (u, v), target_layer, rules, pair_layers, all_layers
            )
            features.update(cross_features)

        features["label"] = label
        dataset.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Normalize ALL features if requested (comprehensive normalization)
    if normalize:
        feature_cols = [col for col in df.columns if col != "label"]

        # First handle similarity scores (like in Task 03)
        if len(similarity_scores) > 0:
            similarity_scores = np.array(similarity_scores)
            min_score = np.min(similarity_scores)
            max_score = np.max(similarity_scores)
            if max_score > min_score:
                df["similarity"] = (similarity_scores - min_score) / (
                    max_score - min_score
                )
            else:
                df["similarity"] = np.zeros_like(similarity_scores)
        else:
            df["similarity"] = df["similarity_raw"]

        # Then normalize all other features to [0,1] range
        for col in feature_cols:
            if col != "similarity":  # Skip similarity as it's already normalized
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.0  # If all values are the same, set to 0

    else:
        df["similarity"] = df["similarity_raw"]

    if "similarity_raw" in df.columns:
        df = df.drop("similarity_raw", axis=1)

    return df


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_single_feature(df, feature_col, k=5):
    X = df[feature_col].values
    y = df["label"].values

    thresholds = np.arange(0.0, 1.05, 0.05)
    results = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for threshold in thresholds:
        precisions, recalls, f1s = [], [], []

        for _, test_idx in kf.split(X):
            X_test, y_test = X[test_idx], y[test_idx]
            y_pred = (X_test >= threshold).astype(int)

            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))

        results.append(
            {
                "threshold": threshold,
                "precision": np.mean(precisions),
                "recall": np.mean(recalls),
                "f1": np.mean(f1s),
            }
        )

    return results


def evaluate_layer_with_methods(
    target_layer, layer_networks, rules, pair_layers, methods, k=5, normalize=True
):
    results_by_method = {}

    for method_name, similarity_func in methods.items():
        print(f"  {method_name}...", end=" ")

        # Baseline: within-layer only (normalized like Task 03)
        df_baseline = build_multilayer_dataset(
            target_layer,
            layer_networks,
            rules,
            pair_layers,
            similarity_func,
            use_cross_layer=False,
            normalize=normalize,
        )
        results_baseline = evaluate_single_feature(df_baseline, "similarity", k=k)

        # Enhanced: with cross-layer features
        df_enhanced = build_multilayer_dataset(
            target_layer,
            layer_networks,
            rules,
            pair_layers,
            similarity_func,
            use_cross_layer=True,
            normalize=normalize,
        )

        # Combine features: similarity + association_score
        if "association_score" in df_enhanced.columns:
            df_enhanced["combined"] = (
                0.7 * df_enhanced["similarity"] + 0.3 * df_enhanced["association_score"]
            )
        else:
            df_enhanced["combined"] = df_enhanced["similarity"]

        results_enhanced = evaluate_single_feature(df_enhanced, "combined", k=k)

        # Store results
        results_by_method[method_name] = {
            "baseline": results_baseline,
            "enhanced": results_enhanced,
        }

        # Best F1
        best_baseline = max(results_baseline, key=lambda x: x["f1"])
        best_enhanced = max(results_enhanced, key=lambda x: x["f1"])
        improvement = best_enhanced["f1"] - best_baseline["f1"]

        status = "✓" if improvement > 0 else "✗" if improvement < 0 else "="
        print(
            f"Best F1: {best_baseline['f1']:.3f} → {best_enhanced['f1']:.3f} "
            f"({improvement:+.3f}) {status}"
        )

    return results_by_method


def plot_all_layers_results(all_layer_results):
    for target_layer, results_by_method in all_layer_results.items():
        thresholds = [
            r["threshold"] for r in list(results_by_method.values())[0]["baseline"]
        ]
        colors = cm.tab10(np.linspace(0, 1, len(results_by_method)))
        safe_layer_name = target_layer.replace(" ", "_").replace("/", "_")

        # F1 comparison plot
        plt.figure(figsize=(14, 8))
        for i, (method_name, results) in enumerate(results_by_method.items()):
            f1_base = [r["f1"] for r in results["baseline"]]
            f1_enh = [r["f1"] for r in results["enhanced"]]
            color = colors[i]

            plt.plot(
                thresholds,
                f1_base,
                "--",
                color=color,
                alpha=0.7,
                linewidth=3,
                label=f"{method_name} (baseline)",
                marker="o",
                markersize=4,
            )
            plt.plot(
                thresholds,
                f1_enh,
                "-",
                color=color,
                linewidth=3,
                label=f"{method_name} (enhanced)",
                marker="s",
                markersize=4,
            )

        plt.title(
            f"{target_layer} - F1 Score Comparison", fontweight="bold", fontsize=16
        )
        plt.xlabel("Threshold", fontsize=14, fontweight="bold")
        plt.ylabel("F1 Score", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11, ncol=2, loc="best")
        plt.grid(True, alpha=0.4)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(
            ALL_LAYERS_DIR / f"{safe_layer_name}_f1_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Precision comparison plot
        plt.figure(figsize=(14, 8))
        for i, (method_name, results) in enumerate(results_by_method.items()):
            prec_base = [r["precision"] for r in results["baseline"]]
            prec_enh = [r["precision"] for r in results["enhanced"]]
            color = colors[i]

            plt.plot(
                thresholds,
                prec_base,
                "--",
                color=color,
                alpha=0.7,
                linewidth=3,
                label=f"{method_name} (baseline)",
                marker="o",
                markersize=4,
            )
            plt.plot(
                thresholds,
                prec_enh,
                "-",
                color=color,
                linewidth=3,
                label=f"{method_name} (enhanced)",
                marker="s",
                markersize=4,
            )

        plt.title(
            f"{target_layer} - Precision Comparison: Baseline vs Enhanced with Association Rules",
            fontweight="bold",
            fontsize=16,
        )
        plt.xlabel("Threshold", fontsize=14, fontweight="bold")
        plt.ylabel("Precision", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11, ncol=2, loc="best")
        plt.grid(True, alpha=0.4)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(
            ALL_LAYERS_DIR / f"{safe_layer_name}_precision_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 3: Recall Comparison (Baseline vs Enhanced)
        plt.figure(figsize=(14, 8))
        for i, (method_name, results) in enumerate(results_by_method.items()):
            rec_base = [r["recall"] for r in results["baseline"]]
            rec_enh = [r["recall"] for r in results["enhanced"]]
            color = colors[i]

            plt.plot(
                thresholds,
                rec_base,
                "--",
                color=color,
                alpha=0.7,
                linewidth=3,
                label=f"{method_name} (baseline)",
                marker="o",
                markersize=4,
            )
            plt.plot(
                thresholds,
                rec_enh,
                "-",
                color=color,
                linewidth=3,
                label=f"{method_name} (enhanced)",
                marker="s",
                markersize=4,
            )

        plt.title(f"{target_layer} - Recall Comparison", fontweight="bold", fontsize=16)
        plt.xlabel("Threshold", fontsize=14, fontweight="bold")
        plt.ylabel("Recall", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11, ncol=2, loc="best")
        plt.grid(True, alpha=0.4)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(
            ALL_LAYERS_DIR / f"{safe_layer_name}_recall_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def create_overall_comparison(summary_df):
    """Create overall comparison visualizations"""

    # 1. Improvement heatmap
    pivot = summary_df.pivot(index="method", columns="layer", values="improvement_f1")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "F1 Improvement"},
    )
    plt.title("F1 Improvement from Association Rules", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        COMPARISON_DIR / "improvement_heatmap.png", dpi=200, bbox_inches="tight"
    )
    plt.close()

    # 2. Best F1 by layer and method
    pivot_best = summary_df.pivot(index="method", columns="layer", values="enhanced_f1")

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_best,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Best F1 Score"},
    )
    plt.title(
        "Best F1 Scores (Enhanced with Association Rules)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "best_f1_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Bar chart: average improvement per method
    avg_improvement = (
        summary_df.groupby("method")["improvement_f1"].mean().sort_values()
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(avg_improvement)),
        avg_improvement.values,
        color=["red" if x < 0 else "green" for x in avg_improvement.values],
    )
    plt.yticks(range(len(avg_improvement)), avg_improvement.index)
    plt.xlabel("Average F1 Improvement", fontweight="bold")
    plt.title(
        "Average F1 Improvement Across All Layers", fontsize=14, fontweight="bold"
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        COMPARISON_DIR / "avg_improvement_by_method.png", dpi=200, bbox_inches="tight"
    )
    plt.close()


def main():
    _, nodes_df, edges_df, layer_names = load_multilayer_data()
    layer_networks = build_layer_graphs(nodes_df, edges_df, layer_names)

    pair_layers = get_node_pair_layers(edges_df, layer_names)
    support = compute_layer_support(pair_layers, layer_names)
    rules = mine_association_rules(
        support, min_support=0.02, min_confidence=0.2, min_lift=1.1
    )

    if rules:
        df_rules = pd.DataFrame(rules).sort_values("lift", ascending=False)
        print(f"\nMined {len(rules)} association rules")
        print("Top 5 rules by lift:")
        for i, row in df_rules.head(5).iterrows():
            ante_str = "+".join(sorted(row["antecedent"]))
            cons_str = "+".join(sorted(row["consequent"]))
            print(
                f"  {ante_str} → {cons_str}: conf={row['confidence']:.3f}, lift={row['lift']:.3f}"
            )

    visualize_association_rules(rules, layer_names)

    methods = {
        "Common Neighbors": common_neighbors_score,
        "Jaccard": jaccard_score,
        "Adamic-Adar": adamic_adar_score,
        "Preferential Attachment": preferential_attachment_score,
        "Resource Allocation": resource_allocation_score,
        "Cosine Similarity": cosine_similarity_score,
        "Sorensen Index": sorensen_index_score,
        "CAR-based CN": car_based_common_neighbors_score,
    }

    print("\nEvaluating link prediction per layer...")
    all_results = []
    all_layer_results = {}

    for target_layer in layer_networks.keys():
        print(f"\n{target_layer}:")
        results = evaluate_layer_with_methods(
            target_layer,
            layer_networks,
            rules,
            pair_layers,
            methods,
            k=5,
            normalize=True,
        )

        # Store results for plotting later
        all_layer_results[target_layer] = results

        # Extract summary statistics
        for method_name, method_results in results.items():
            best_baseline = max(method_results["baseline"], key=lambda x: x["f1"])
            best_enhanced = max(method_results["enhanced"], key=lambda x: x["f1"])

            all_results.append(
                {
                    "layer": target_layer,
                    "method": method_name,
                    "baseline_f1": best_baseline["f1"],
                    "baseline_precision": best_baseline["precision"],
                    "baseline_recall": best_baseline["recall"],
                    "baseline_threshold": best_baseline["threshold"],
                    "enhanced_f1": best_enhanced["f1"],
                    "enhanced_precision": best_enhanced["precision"],
                    "enhanced_recall": best_enhanced["recall"],
                    "enhanced_threshold": best_enhanced["threshold"],
                    "improvement_f1": best_enhanced["f1"] - best_baseline["f1"],
                    "improvement_precision": best_enhanced["precision"]
                    - best_baseline["precision"],
                    "improvement_recall": best_enhanced["recall"]
                    - best_baseline["recall"],
                    "improvement_pct": (
                        (best_enhanced["f1"] - best_baseline["f1"])
                        / (best_baseline["f1"] + 1e-10)
                    )
                    * 100,
                }
            )

    plot_all_layers_results(all_layer_results)

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values(
        ["layer", "enhanced_f1"], ascending=[True, False]
    )
    summary_df.to_csv(SUMMARY_DIR / "performance_summary.csv", index=False)

    # Print summary
    print("\nPERFORMANCE SUMMARY")
    print("-" * 60)
    for _, row in summary_df.iterrows():
        status = (
            "✓"
            if row["improvement_f1"] > 0
            else "✗" if row["improvement_f1"] < 0 else "="
        )
        print(
            f"{row['layer']:<12} | {row['method']:<18} | "
            f"{row['baseline_f1']:.3f} → {row['enhanced_f1']:.3f} "
            f"({row['improvement_f1']:+.3f}) {status}"
        )

    print("\nSTATISTICS")
    print("-" * 30)

    improvements_f1 = summary_df["improvement_f1"]
    improvements_prec = summary_df["improvement_precision"]
    improvements_rec = summary_df["improvement_recall"]

    print("F1 IMPROVEMENTS:")
    print(f"  Average: {improvements_f1.mean():.4f} ± {improvements_f1.std():.4f}")
    print(f"  Range: [{improvements_f1.min():.4f}, {improvements_f1.max():.4f}]")

    print("\nPRECISION IMPROVEMENTS:")
    print(f"  Average: {improvements_prec.mean():.4f} ± {improvements_prec.std():.4f}")
    print(f"  Range: [{improvements_prec.min():.4f}, {improvements_prec.max():.4f}]")

    print("\nRECALL IMPROVEMENTS:")
    print(f"  Average: {improvements_rec.mean():.4f} ± {improvements_rec.std():.4f}")
    print(f"  Range: [{improvements_rec.min():.4f}, {improvements_rec.max():.4f}]")

    improvements = summary_df[summary_df["improvement_f1"] > 0]
    degradations = summary_df[summary_df["improvement_f1"] < 0]

    print(
        f"Improvements: {len(improvements)}/{len(summary_df)} ({len(improvements)/len(summary_df)*100:.1f}%)"
    )
    if len(improvements) > 0:
        print(
            f"  Avg: {improvements['improvement_f1'].mean():.4f}, Best: {improvements['improvement_f1'].max():.4f}"
        )

    print(
        f"Degradations: {len(degradations)}/{len(summary_df)} ({len(degradations)/len(summary_df)*100:.1f}%)"
    )
    if len(degradations) > 0:
        print(
            f"  Avg: {degradations['improvement_f1'].mean():.4f}, Worst: {degradations['improvement_f1'].min():.4f}"
        )

    best_overall = summary_df.loc[summary_df["enhanced_f1"].idxmax()]
    best_improvement = summary_df.loc[summary_df["improvement_f1"].idxmax()]
    print(
        f"\nBest F1: {best_overall['layer']} + {best_overall['method']} ({best_overall['enhanced_f1']:.3f})"
    )
    print(
        f"Best Improvement: {best_improvement['layer']} + {best_improvement['method']} (+{best_improvement['improvement_f1']:.3f})"
    )

    # Create overall comparison visualizations
    create_overall_comparison(summary_df)

    # Layer statistics
    layer_stats = []
    for layer_name, G in layer_networks.items():
        layer_stats.append(
            {
                "layer": layer_name,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "avg_degree": np.mean([d for n, d in G.degree()]),
                "clustering": nx.average_clustering(G),
                "avg_f1_improvement": summary_df[summary_df["layer"] == layer_name][
                    "improvement_f1"
                ].mean(),
            }
        )

    layer_stats_df = pd.DataFrame(layer_stats)
    layer_stats_df.to_csv(SUMMARY_DIR / "layer_statistics.csv", index=False)

    with open(SUMMARY_DIR / "overall_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Layers: {', '.join(layer_networks.keys())}\n")
        f.write(f"Association rules: {len(rules)}\n")
        f.write(f"Methods: {len(methods)}\n\n")
        f.write(f"Avg F1 improvement: {summary_df['improvement_f1'].mean():.4f}\n")
        pos_count = (summary_df["improvement_f1"] > 0).sum()
        f.write(
            f"Improvements: {pos_count}/{len(summary_df)} ({pos_count/len(summary_df)*100:.1f}%)\n\n"
        )
        f.write("Top Rules (by lift):\n")
        for i, row in df_rules.head(5).iterrows():
            ante = "+".join(row["antecedent"])
            cons = "+".join(row["consequent"])
            f.write(f"{i+1}. {ante} → {cons} (lift: {row['lift']:.3f})\n")

    print(f"\nResults saved to {RESULTS_DIR}")
    print("Analysis complete!")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
