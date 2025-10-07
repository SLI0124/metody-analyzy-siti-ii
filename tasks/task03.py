import csv
import random
import networkx as nx
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import pandas as pd

BASE_RESULTS_DIR = "../results/task03"
NORMALIZED_DIR = os.path.join(BASE_RESULTS_DIR, "normalized")
UNNORMALIZED_DIR = os.path.join(BASE_RESULTS_DIR, "unnormalized")
COMPARISON_DIR = os.path.join(BASE_RESULTS_DIR, "comparison")


def load_graph(file_path):
    G = nx.Graph()
    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        return G
    with open(file_path, mode="r") as file:
        reader = csv.reader(file, delimiter=";")
        rows = []
        for row in reader:
            rows.append(row)
            if len(row) >= 2:
                u, v = row[0], row[1]
                G.add_edge(u, v)
    return G


# --------------------------
# Similarity scores
# --------------------------
def common_neighbors_score(G, u, v):
    return len(list(nx.common_neighbors(G, u, v)))


def jaccard_score(G, u, v):
    preds = list(nx.jaccard_coefficient(G, [(u, v)]))
    return preds[0][2]


def adar_index(G, u, v):
    preds = list(nx.adamic_adar_index(G, [(u, v)]))
    return preds[0][2]


def preferential_attachment_score(G, u, v):
    preds = list(nx.preferential_attachment(G, [(u, v)]))
    return preds[0][2]


def resource_allocation_index(G, u, v):
    preds = list(nx.resource_allocation_index(G, [(u, v)]))
    return preds[0][2]


def cosine_similarity(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    len_u = len(neighbors_u)
    len_v = len(neighbors_v)
    if len_u == 0 or len_v == 0:
        return 0.0
    return intersection / (np.sqrt(len_u * len_v))


def sorensen_index(G, u, v):
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    intersection = len(neighbors_u & neighbors_v)
    total = len(neighbors_u) + len(neighbors_v)
    if total == 0:
        return 0.0
    return 2 * intersection / total


def car_based_common_neighbors(G, u, v):
    # Add dummy community attribute if not present
    if not all("community" in G.nodes[n] for n in G.nodes()):
        for idx, node in enumerate(G.nodes()):
            G.nodes[node]["community"] = idx % 2
    preds = list(nx.cn_soundarajan_hopcroft(G, [(u, v)]))
    return preds[0][2]


def build_dataset(G, similarity_func, n_negatives=None, normalize=False):
    positives = [(u, v, 1) for u, v in G.edges()]
    non_edges = list(nx.non_edges(G))

    if n_negatives is None:
        n_negatives = len(positives)
    negatives = random.sample(non_edges, min(n_negatives, len(non_edges)))
    negatives = [(u, v, 0) for u, v in negatives]

    dataset = []
    scores = []
    labels = []

    for u, v, label in positives + negatives:
        score = similarity_func(G, u, v)
        scores.append(score)
        labels.append(label)

    # Normalize scores to [0, 1] range if requested
    if normalize:
        scores = np.array(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        else:
            scores = np.zeros_like(scores)

    for score, label in zip(scores, labels):
        dataset.append([score, label])

    return dataset


def evaluate_thresholds_kfold(dataset, k):
    X = np.array([d[0] for d in dataset])
    y = np.array([d[1] for d in dataset])
    thresholds = np.arange(0.0, 1.05, 0.05)
    f1s_by_threshold = np.zeros(len(thresholds))
    precisions_by_threshold = np.zeros(len(thresholds))
    recalls_by_threshold = np.zeros(len(thresholds))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_test, y_test = X[test_idx], y[test_idx]
        for i, t in enumerate(thresholds):
            y_pred = (X_test >= t).astype(int)
            precisions_by_threshold[i] += precision_score(
                y_test, y_pred, zero_division=0
            )
            recalls_by_threshold[i] += recall_score(y_test, y_pred, zero_division=0)
            f1s_by_threshold[i] += f1_score(y_test, y_pred, zero_division=0)

    results = []
    for i, t in enumerate(thresholds):
        results.append(
            {
                "threshold": t,
                "precision": precisions_by_threshold[i] / k,
                "recall": recalls_by_threshold[i] / k,
                "f1": f1s_by_threshold[i] / k,
            }
        )
    return results


def compute_summary_stats(results):
    """Compute min, max, mean, std for each metric across all thresholds"""
    f1_scores = [r["f1"] for r in results]
    precision_scores = [r["precision"] for r in results]
    recall_scores = [r["recall"] for r in results]

    # Find best threshold based on F1
    best_idx = np.argmax(f1_scores)
    best_threshold = results[best_idx]["threshold"]

    return {
        "f1_min": np.min(f1_scores),
        "f1_max": np.max(f1_scores),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "precision_min": np.min(precision_scores),
        "precision_max": np.max(precision_scores),
        "precision_mean": np.mean(precision_scores),
        "precision_std": np.std(precision_scores),
        "recall_min": np.min(recall_scores),
        "recall_max": np.max(recall_scores),
        "recall_mean": np.mean(recall_scores),
        "recall_std": np.std(recall_scores),
        "best_threshold": best_threshold,
        "best_f1": results[best_idx]["f1"],
        "best_precision": results[best_idx]["precision"],
        "best_recall": results[best_idx]["recall"],
    }


def run_analysis(datasets, methods, optimal_k, output_dir, normalize=False):
    """Run complete analysis and save results to output_dir"""

    mode_name = "NORMALIZED" if normalize else "UNNORMALIZED"
    print(f"\n{'#'*60}")
    print(f"# {mode_name} ANALYSIS")
    print(f"{'#'*60}")

    thresholds = np.arange(0.0, 1.05, 0.05)
    all_summary_stats = []

    for name, path in datasets.items():
        G = load_graph(path)
        print(
            f"\nDataset: {name} - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
        )

        f1_scores_by_method = {}
        precision_scores_by_method = {}
        recall_scores_by_method = {}

        for method_name, func in methods.items():
            dataset = build_dataset(G, func, normalize=normalize)
            if len(dataset) == 0:
                continue

            results = evaluate_thresholds_kfold(dataset, optimal_k)
            stats = compute_summary_stats(results)

            print(f"\n{method_name}:")
            print(
                f"  Best F1: {stats['best_f1']:.4f} at threshold {stats['best_threshold']:.2f}"
            )
            print(
                f"  F1  → Min: {stats['f1_min']:.4f}, Max: {stats['f1_max']:.4f}, "
                f"Mean: {stats['f1_mean']:.4f}, Std: {stats['f1_std']:.4f}"
            )
            print(
                f"  Prec→ Min: {stats['precision_min']:.4f}, Max: {stats['precision_max']:.4f}, "
                f"Mean: {stats['precision_mean']:.4f}, Std: {stats['precision_std']:.4f}"
            )
            print(
                f"  Rec → Min: {stats['recall_min']:.4f}, Max: {stats['recall_max']:.4f}, "
                f"Mean: {stats['recall_mean']:.4f}, Std: {stats['recall_std']:.4f}"
            )

            f1_scores_by_method[method_name] = [res["f1"] for res in results]
            precision_scores_by_method[method_name] = [
                res["precision"] for res in results
            ]
            recall_scores_by_method[method_name] = [res["recall"] for res in results]

            all_summary_stats.append(
                {
                    "Dataset": name,
                    "Method": method_name,
                    "Best_F1": stats["best_f1"],
                    "Best_Threshold": stats["best_threshold"],
                    "F1_Mean": stats["f1_mean"],
                    "F1_Std": stats["f1_std"],
                    "Best_Precision": stats["best_precision"],
                    "Best_Recall": stats["best_recall"],
                }
            )

        # Create dataset-specific subdirectory
        dataset_dir = os.path.join(output_dir, name.lower().replace(" ", "_"))
        os.makedirs(dataset_dir, exist_ok=True)

        # Plot F1 scores for optimal k
        plt.figure(figsize=(12, 7))
        for method_name, f1s in f1_scores_by_method.items():
            plt.plot(thresholds, f1s, marker="o", label=method_name, linewidth=2)
        plt.title(
            f"F1 Score vs Threshold - {name} (k={optimal_k}, {mode_name})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"f1_vs_threshold_k{optimal_k}.png"
        plt.savefig(os.path.join(dataset_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot Precision scores
        plt.figure(figsize=(12, 7))
        for method_name, prec in precision_scores_by_method.items():
            plt.plot(thresholds, prec, marker="o", label=method_name, linewidth=2)
        plt.title(
            f"Precision vs Threshold - {name} (k={optimal_k}, {mode_name})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"precision_vs_threshold_k{optimal_k}.png"
        plt.savefig(os.path.join(dataset_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

        # Plot Recall scores
        plt.figure(figsize=(12, 7))
        for method_name, rec in recall_scores_by_method.items():
            plt.plot(thresholds, rec, marker="o", label=method_name, linewidth=2)
        plt.title(
            f"Recall vs Threshold - {name} (k={optimal_k}, {mode_name})",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Recall", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"recall_vs_threshold_k{optimal_k}.png"
        plt.savefig(os.path.join(dataset_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

    # Create summary comparison table
    print(f"\n{'='*50}")
    print(f"SUMMARY ({mode_name})")
    print(f"{'='*50}")
    df = pd.DataFrame(all_summary_stats)
    df_sorted = df.sort_values(["Dataset", "Best_F1"], ascending=[True, False])
    print(df_sorted.to_string(index=False))

    csv_path = os.path.join(output_dir, f"summary_statistics.csv")
    df_sorted.to_csv(csv_path, index=False)

    return df_sorted


def create_comparison_plots(datasets, methods, optimal_k, comparison_dir):
    """Create side-by-side comparison plots of normalized vs unnormalized"""

    print(f"\n\n{'='*70}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*70}\n")

    thresholds = np.arange(0.0, 1.05, 0.05)

    for name, path in datasets.items():
        G = load_graph(path)

        print(f"Generating comparison for {name}...")

        for method_name, func in methods.items():
            dataset_norm = build_dataset(G, func, normalize=True)
            dataset_unnorm = build_dataset(G, func, normalize=False)

            if len(dataset_norm) == 0 or len(dataset_unnorm) == 0:
                continue

            results_norm = evaluate_thresholds_kfold(dataset_norm, optimal_k)
            results_unnorm = evaluate_thresholds_kfold(dataset_unnorm, optimal_k)

            f1_norm = [r["f1"] for r in results_norm]
            f1_unnorm = [r["f1"] for r in results_unnorm]

            plt.figure(figsize=(12, 7))

            plt.plot(
                thresholds,
                f1_unnorm,
                marker="o",
                linewidth=2,
                color="blue",
                label="Unnormalized",
                markersize=6,
            )
            plt.plot(
                thresholds,
                f1_norm,
                marker="s",
                linewidth=2,
                color="red",
                label="Normalized",
                markersize=6,
            )

            plt.title(
                f"{method_name} - Normalized vs Unnormalized\n{name} (k={optimal_k})",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Threshold", fontsize=12)
            plt.ylabel("F1 Score", fontsize=12)
            plt.legend(fontsize=11, loc="best")
            plt.grid(True, alpha=0.3)
            plt.ylim([-0.05, 1.05])
            plt.tight_layout()

            safe_method_name = method_name.lower().replace(" ", "_").replace("/", "_")
            safe_dataset_name = name.lower().replace(" ", "_")
            fname = f"{safe_dataset_name}_{safe_method_name}_comparison.png"
            plt.savefig(
                os.path.join(comparison_dir, fname), dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"Comparison plots for {name} saved to {comparison_dir}\n")


def main():
    # Create all necessary directories
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    os.makedirs(UNNORMALIZED_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    datasets = {
        "Karate": "../data/edges karate.csv",
        "Dolphins": "../data/edges dolphins.csv",
        "Les Misérables": "../data/edges lesmis.csv",
    }

    methods = {
        "Common Neighbors": common_neighbors_score,
        "Jaccard": jaccard_score,
        "Adamic-Adar": adar_index,
        "Preferential Attachment": preferential_attachment_score,
        "Resource Allocation": resource_allocation_index,
        "Cosine Similarity": cosine_similarity,
        "Sorensen Index": sorensen_index,
        "CAR-based CN": car_based_common_neighbors,
    }

    OPTIMAL_K = 5

    # Run UNNORMALIZED analysis first
    df_unnorm = run_analysis(
        datasets, methods, OPTIMAL_K, UNNORMALIZED_DIR, normalize=False
    )

    # Run NORMALIZED analysis next
    df_norm = run_analysis(datasets, methods, OPTIMAL_K, NORMALIZED_DIR, normalize=True)

    # Create comparison plots
    create_comparison_plots(datasets, methods, OPTIMAL_K, COMPARISON_DIR)

    # Create combined summary comparison
    print(f"\n\n{'='*70}")
    print("CREATING COMBINED SUMMARY COMPARISON")
    print(f"{'='*70}\n")

    df_unnorm["Mode"] = "Unnormalized"
    df_norm["Mode"] = "Normalized"
    df_combined = pd.concat([df_unnorm, df_norm])
    df_combined = df_combined.sort_values(["Dataset", "Method", "Mode"])

    combined_path = os.path.join(BASE_RESULTS_DIR, "combined_summary.csv")
    df_combined.to_csv(combined_path, index=False)
    print(f"Combined summary saved to: {combined_path}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
