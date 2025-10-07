import csv
import random
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

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
            if len(row) >= 2:  # assume edge list: u,v
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
    # Get neighbors as sets
    neighbors_u = set(G.neighbors(u))
    neighbors_v = set(G.neighbors(v))
    # Compute intersection and lengths
    intersection = len(neighbors_u & neighbors_v)
    len_u = len(neighbors_u)
    len_v = len(neighbors_v)
    # Avoid division by zero
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
    # Assign a default community to each node if not present
    if not all("community" in G.nodes[n] for n in G.nodes()):
        for idx, node in enumerate(G.nodes()):
            G.nodes[node]["community"] = (
                idx % 2
            )  # Example: assign communities 0 and 1 alternately
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
    negatives = random.sample(non_edges, n_negatives)
    negatives = [(u, v, 0) for u, v in negatives]

    dataset = []
    for u, v, label in positives + negatives:
        score = similarity_func(G, u, v)
        dataset.append([score, label])
    return dataset


# --------------------------
# Evaluate with threshold search
# --------------------------
def evaluate_thresholds(dataset):
    X = np.array([d[0] for d in dataset])
    y = np.array([d[1] for d in dataset])

    # Split into train/test
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Only use uniform range of thresholds
    thresholds = np.arange(0.0, 1.05, 0.05)

    results = []
    for t in thresholds:
        y_pred = (X_test >= t).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return results


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
    # Average over folds
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
# Main
# --------------------------
def main():
    # Ensure results directory exists
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

    k_values = [2, 3, 4, 5, 10]
    thresholds = np.arange(0.0, 1.05, 0.05)
    threshold_labels = [f"{t:.2f}" for t in thresholds]

    for name, path in datasets.items():
        G = load_graph(path)
        print(
            f"\nDataset: {name} (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})"
        )

        for k in k_values:
            print(f"\n  K-Fold: k={k}")
            f1_scores_by_method = {}
            for method_name, func in methods.items():
                dataset = build_dataset(G, func)
                if len(dataset) == 0:
                    print(
                        f"    {method_name}: WARNING - No data to evaluate (empty dataset)."
                    )
                    continue
                print(f"    {method_name}:")
                results = evaluate_thresholds_kfold(dataset, k)
                for res in results:
                    print(f"      Threshold ({res['threshold']:.3f}):")
                    print(
                        f"        Precision={res['precision']:.3f}, Recall={res['recall']:.3f}, F1={res['f1']:.3f}"
                    )
                f1_scores_by_method[method_name] = [res["f1"] for res in results]
            # Plot F1 scores for all methods for this dataset and k
            plt.figure(figsize=(10, 6))
            for method_name, f1s in f1_scores_by_method.items():
                plt.plot(thresholds, f1s, marker="o", label=method_name)
            plt.title(f"F1 Score vs Threshold for {name} (k={k})")
            plt.xlabel("Threshold")
            plt.ylabel("F1 Score")
            plt.legend()
            plt.grid(True)
            fname = f"{name.lower().replace(' ', '_')}_f1_vs_threshold_k{k}.png"
            plt.savefig(os.path.join(RESULTS_DIR, fname))
            plt.close()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
