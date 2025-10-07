import csv
import random
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "../results/task03"


# --------------------------
# Load graph from CSV
# --------------------------
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
    if not all("community" in G.nodes[n] for n in G.nodes()):
        for idx, node in enumerate(G.nodes()):
            G.nodes[node]["community"] = idx % 2
    preds = list(nx.cn_soundarajan_hopcroft(G, [(u, v)]))
    return preds[0][2]


# --------------------------
# Build dataset
# --------------------------
def build_dataset(G, similarity_func, n_negatives=None):
    positives = [(u, v, 1) for u, v in G.edges()]
    non_edges = list(nx.non_edges(G))

    if n_negatives is None:
        n_negatives = len(positives)
    negatives = random.sample(non_edges, min(n_negatives, len(non_edges)))
    negatives = [(u, v, 0) for u, v in negatives]

    dataset = []
    for u, v, label in positives + negatives:
        score = similarity_func(G, u, v)
        dataset.append([score, label])
    return dataset


# --------------------------
# Evaluate with k-fold
# --------------------------
def evaluate_thresholds_kfold(dataset, k):
    X = np.array([d[0] for d in dataset])
    y = np.array([d[1] for d in dataset])
    thresholds = np.arange(0.0, 1.05, 0.05)
    f1s_by_threshold = np.zeros(len(thresholds))
    precisions_by_threshold = np.zeros(len(thresholds))
    recalls_by_threshold = np.zeros(len(thresholds))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y):
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


# --------------------------
# Compute summary statistics
# --------------------------
def compute_summary_stats(results):
    """Compute min, max, mean, std for each metric across all thresholds"""
    f1_scores = [r['f1'] for r in results]
    precision_scores = [r['precision'] for r in results]
    recall_scores = [r['recall'] for r in results]
    
    # Find best threshold based on F1
    best_idx = np.argmax(f1_scores)
    best_threshold = results[best_idx]['threshold']
    
    return {
        'f1_min': np.min(f1_scores),
        'f1_max': np.max(f1_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'precision_min': np.min(precision_scores),
        'precision_max': np.max(precision_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'recall_min': np.min(recall_scores),
        'recall_max': np.max(recall_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'best_threshold': best_threshold,
        'best_f1': results[best_idx]['f1'],
        'best_precision': results[best_idx]['precision'],
        'best_recall': results[best_idx]['recall']
    }


# --------------------------
# Main
# --------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    # Use k=5 as the standard value for detailed analysis
    OPTIMAL_K = 5
    k_values = [2, 3, 4, 5, 10]
    thresholds = np.arange(0.0, 1.05, 0.05)
    
    # Store all results for comparison table
    all_summary_stats = []

    for name, path in datasets.items():
        G = load_graph(path)
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"{'='*70}")

        # Detailed analysis with optimal k
        print(f"\n--- DETAILED ANALYSIS (k={OPTIMAL_K}) ---")
        f1_scores_by_method = {}
        
        for method_name, func in methods.items():
            dataset = build_dataset(G, func)
            if len(dataset) == 0:
                print(f"\n{method_name}: WARNING - Empty dataset")
                continue
            
            results = evaluate_thresholds_kfold(dataset, OPTIMAL_K)
            stats = compute_summary_stats(results)
            
            print(f"\n{method_name}:")
            print(f"  Best F1: {stats['best_f1']:.4f} at threshold {stats['best_threshold']:.2f}")
            print(f"  F1  → Min: {stats['f1_min']:.4f}, Max: {stats['f1_max']:.4f}, "
                  f"Mean: {stats['f1_mean']:.4f}, Std: {stats['f1_std']:.4f}")
            print(f"  Prec→ Min: {stats['precision_min']:.4f}, Max: {stats['precision_max']:.4f}, "
                  f"Mean: {stats['precision_mean']:.4f}, Std: {stats['precision_std']:.4f}")
            print(f"  Rec → Min: {stats['recall_min']:.4f}, Max: {stats['recall_max']:.4f}, "
                  f"Mean: {stats['recall_mean']:.4f}, Std: {stats['recall_std']:.4f}")
            
            f1_scores_by_method[method_name] = [res["f1"] for res in results]
            
            # Store for summary table
            all_summary_stats.append({
                'Dataset': name,
                'Method': method_name,
                'Best_F1': stats['best_f1'],
                'Best_Threshold': stats['best_threshold'],
                'F1_Mean': stats['f1_mean'],
                'F1_Std': stats['f1_std']
            })
        
        # Plot F1 scores for optimal k
        plt.figure(figsize=(12, 7))
        for method_name, f1s in f1_scores_by_method.items():
            plt.plot(thresholds, f1s, marker='o', label=method_name, linewidth=2)
        plt.title(f"F1 Score vs Threshold for {name} (k={OPTIMAL_K})", fontsize=14, fontweight='bold')
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"{name.lower().replace(' ', '_')}_f1_vs_threshold_k{OPTIMAL_K}.png"
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate plots for all k values
        for k in k_values:
            if k == OPTIMAL_K:
                continue  # Already generated above
            f1_scores_by_method_k = {}
            for method_name, func in methods.items():
                dataset = build_dataset(G, func)
                if len(dataset) == 0:
                    continue
                results = evaluate_thresholds_kfold(dataset, k)
                f1_scores_by_method_k[method_name] = [res["f1"] for res in results]
            
            plt.figure(figsize=(10, 6))
            for method_name, f1s in f1_scores_by_method_k.items():
                plt.plot(thresholds, f1s, marker='o', label=method_name)
            plt.title(f"F1 Score vs Threshold for {name} (k={k})")
            plt.xlabel("Threshold")
            plt.ylabel("F1 Score")
            plt.legend()
            plt.grid(True)
            fname = f"{name.lower().replace(' ', '_')}_f1_vs_threshold_k{k}.png"
            plt.savefig(os.path.join(RESULTS_DIR, fname))
            plt.close()

    # Create summary comparison table
    print(f"\n\n{'='*70}")
    print("SUMMARY COMPARISON TABLE (k=5)")
    print(f"{'='*70}")
    df = pd.DataFrame(all_summary_stats)
    df_sorted = df.sort_values(['Dataset', 'Best_F1'], ascending=[True, False])
    print(df_sorted.to_string(index=False))
    
    # Save to CSV
    df_sorted.to_csv(os.path.join(RESULTS_DIR, 'summary_statistics.csv'), index=False)
    print(f"\nSummary saved to: {os.path.join(RESULTS_DIR, 'summary_statistics.csv')}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()