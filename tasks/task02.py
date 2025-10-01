from pathlib import Path
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

DIR_PATH = Path("../data/coauth-DBLP/")
RESULTS_DIR = Path("../results/task02")


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def parse_simplices(nverts_content, simplices_content, times_content):
    nverts = [int(x) for x in nverts_content.strip().splitlines()]
    nodes = [int(x) for x in simplices_content.strip().splitlines()]
    times = [int(x) for x in times_content.strip().splitlines()]
    simplices = []
    idx = 0
    for nv, t in zip(nverts, times):
        simplex_nodes = nodes[idx : idx + nv]
        simplices.append({"nodes": simplex_nodes, "time": t})
        idx += nv
    return simplices


def split_into_frames_by_timestamp(simplices):
    frames = defaultdict(list)  # Dictionary of keys: {timestamp: [simplices]}
    for s in simplices:
        frames[s["time"]].append(s)
    return frames


def build_edge_weights(simplices):
    edge_weights = defaultdict(int)
    for s in simplices:
        for u, v in itertools.combinations(sorted(s["nodes"]), 2):
            edge = tuple(sorted((u, v)))
            edge_weights[edge] += 1
    return edge_weights


def compute_avg_degree(edge_weights):
    node_degree = defaultdict(int)
    for (u, v), w in edge_weights.items():
        node_degree[u] += 1
        node_degree[v] += 1
    if not node_degree:
        return 0
    return sum(node_degree.values()) / len(node_degree)


def compute_avg_weighted_degree(edge_weights):
    node_wdegree = defaultdict(int)
    for (u, v), w in edge_weights.items():
        node_wdegree[u] += w
        node_wdegree[v] += w
    if not node_wdegree:
        return 0
    return sum(node_wdegree.values()) / len(node_wdegree)


def compute_avg_clustering_coefficient(simplices):
    # Build adjacency
    adj = defaultdict(set)
    for s in simplices:
        for u, v in itertools.combinations(s["nodes"], 2):
            adj[u].add(v)
            adj[v].add(u)
    clustering = []
    for node in adj:
        neighbors = adj[node]
        if len(neighbors) < 2:
            clustering.append(0)
            continue
        links = 0
        for u, v in itertools.combinations(neighbors, 2):
            if u in adj[v]:
                links += 1
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        clustering.append(links / possible if possible > 0 else 0)
    if not clustering:
        return 0
    return sum(clustering) / len(clustering)


def find_simplex_highest_avg_edge_weight(simplices, edge_weights):
    best_simplex = None
    best_avg = -1
    for s in simplices:
        edges = [
            tuple(sorted((u, v)))
            for u, v in itertools.combinations(sorted(s["nodes"]), 2)
        ]
        if not edges:
            continue
        avg_weight = sum(edge_weights[e] for e in edges) / len(edges)
        if avg_weight > best_avg:
            best_avg = avg_weight
            best_simplex = s
    return best_simplex, best_avg


def parse_node_labels(node_labels_content):
    # Assumes each line: <id> <name>
    node_labels = {}
    for line in node_labels_content.strip().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            node_id, name = parts
            node_labels[int(node_id)] = name
    return node_labels


def plot_statistics(
    years,
    avg_degrees,
    avg_weighted_degrees,
    avg_clusterings,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(years, avg_degrees, marker="o")
    plt.title("Average Degree Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Degree")

    plt.subplot(1, 3, 2)
    plt.plot(years, avg_weighted_degrees, marker="o", color="orange")
    plt.title("Average Weighted Degree Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Weighted Degree")

    plt.subplot(1, 3, 3)
    plt.plot(years, avg_clusterings, marker="o", color="green")
    plt.title("Average Clustering Coefficient Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Clustering Coefficient")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "network_stats_over_time.png")


def compute_cumulative_statistics_over_time(frames):
    years = []
    avg_degrees = []
    avg_weighted_degrees = []
    avg_clusterings = []
    all_simplices = []
    print("Calculating cumulative statistics:")
    for i, ts in enumerate(sorted(frames.keys()), 1):
        all_simplices.extend(frames[ts])
        edge_weights = build_edge_weights(all_simplices)
        avg_deg = compute_avg_degree(edge_weights)
        avg_wdeg = compute_avg_weighted_degree(edge_weights)
        avg_clust = compute_avg_clustering_coefficient(all_simplices)
        years.append(ts)
        avg_degrees.append(avg_deg)
        avg_weighted_degrees.append(avg_wdeg)
        avg_clusterings.append(avg_clust)
        print(
            f"{i}/{len(frames)} Up to {ts}: Avg degree={avg_deg:.2f}, "
            f"Avg weighted degree={avg_wdeg:.2f}, "
            f"Avg clustering={avg_clust:.4f}"
        )
    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "year": years,
            "avg_degree": avg_degrees,
            "avg_weighted_degree": avg_weighted_degrees,
            "avg_clustering": avg_clusterings,
        }
    )
    df.to_csv(RESULTS_DIR / "cumulative_stats.csv", index=False)
    return years, avg_degrees, avg_weighted_degrees, avg_clusterings


def load_cumulative_statistics_from_csv():
    csv_path = RESULTS_DIR / "cumulative_stats.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded cumulative statistics from {csv_path}")
        return (
            df["year"].tolist(),
            df["avg_degree"].tolist(),
            df["avg_weighted_degree"].tolist(),
            df["avg_clustering"].tolist(),
        )
    return None


def compute_yearly_statistics(frames):
    years = []
    avg_degrees = []
    avg_weighted_degrees = []
    avg_clusterings = []
    print("Calculating yearly statistics:")
    for i, ts in enumerate(sorted(frames.keys()), 1):
        frame = frames[ts]
        edge_weights = build_edge_weights(frame)
        avg_deg = compute_avg_degree(edge_weights)
        avg_wdeg = compute_avg_weighted_degree(edge_weights)
        avg_clust = compute_avg_clustering_coefficient(frame)
        years.append(ts)
        avg_degrees.append(avg_deg)
        avg_weighted_degrees.append(avg_wdeg)
        avg_clusterings.append(avg_clust)
        print(
            f"{i}/{len(frames)} Frame {ts}: Avg degree={avg_deg:.2f}, "
            f"Avg weighted degree={avg_wdeg:.2f}, "
            f"Avg clustering={avg_clust:.4f}"
        )
    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "year": years,
            "avg_degree": avg_degrees,
            "avg_weighted_degree": avg_weighted_degrees,
            "avg_clustering": avg_clusterings,
        }
    )
    df.to_csv(RESULTS_DIR / "yearly_stats.csv", index=False)
    return years, avg_degrees, avg_weighted_degrees, avg_clusterings


def load_yearly_statistics_from_csv():
    csv_path = RESULTS_DIR / "yearly_stats.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded yearly statistics from {csv_path}")
        return (
            df["year"].tolist(),
            df["avg_degree"].tolist(),
            df["avg_weighted_degree"].tolist(),
            df["avg_clustering"].tolist(),
        )
    return None


def main():
    node_labels_path = DIR_PATH / "coauth-DBLP-node-labels.txt"
    node_labels_content = load_file(node_labels_path)

    nverts_path = DIR_PATH / "coauth-DBLP-nverts.txt"
    nverts_content = load_file(nverts_path)

    simplices_path = DIR_PATH / "coauth-DBLP-simplices.txt"
    simplices_content = load_file(simplices_path)

    times_path = DIR_PATH / "coauth-DBLP-times.txt"
    times_content = load_file(times_path)

    node_labels = parse_node_labels(node_labels_content)

    simplices = parse_simplices(nverts_content, simplices_content, times_content)
    frames = split_into_frames_by_timestamp(simplices)
    print(f"Number of frames (unique timestamps): {len(frames)}")

    years = []
    avg_degrees = []
    avg_weighted_degrees = []
    avg_clusterings = []

    # Try to load yearly statistics from CSV, else compute and save
    loaded_yearly = load_yearly_statistics_from_csv()
    if loaded_yearly:
        years, avg_degrees, avg_weighted_degrees, avg_clusterings = loaded_yearly
    else:
        years, avg_degrees, avg_weighted_degrees, avg_clusterings = (
            compute_yearly_statistics(frames)
        )

    # Print per-year statistics
    print("\nYearly statistics for each frame:")
    for i, ts in enumerate(years):
        print(
            f"{i+1}/{len(years)} Frame {ts}: Avg degree={avg_degrees[i]:.2f}, "
            f"Avg weighted degree={avg_weighted_degrees[i]:.2f}, "
            f"Avg clustering={avg_clusterings[i]:.4f}"
        )

    # Find simplex with highest average edge weight (across all data)
    all_edge_weights = build_edge_weights(simplices)
    best_simplex, best_avg = find_simplex_highest_avg_edge_weight(
        simplices, all_edge_weights
    )
    if best_simplex:
        node_names = [node_labels.get(n, str(n)) for n in best_simplex["nodes"]]
        best_year = best_simplex["time"]
        print("Simplex with highest average edge weight (names):", node_names)
        print("Simplex with highest average edge weight (ids):", best_simplex)
        print("Highest average edge weight:", best_avg)
        print("Year of highest stats:", best_year)
    else:
        print("No simplex found.")

    plot_statistics(
        years,
        avg_degrees,
        avg_weighted_degrees,
        avg_clusterings,
    )

    # Try to load cumulative statistics from CSV, else compute and save
    loaded = load_cumulative_statistics_from_csv()
    if loaded:
        (
            cumulative_years,
            cumulative_avg_degrees,
            cumulative_avg_weighted_degrees,
            cumulative_avg_clusterings,
        ) = loaded
    else:
        (
            cumulative_years,
            cumulative_avg_degrees,
            cumulative_avg_weighted_degrees,
            cumulative_avg_clusterings,
        ) = compute_cumulative_statistics_over_time(frames)
    print("\nCumulative statistics up to each year:")
    for i, ts in enumerate(cumulative_years):
        print(
            f"Up to {ts}: Avg degree={cumulative_avg_degrees[i]:.2f}, "
            f"Avg weighted degree={cumulative_avg_weighted_degrees[i]:.2f}, "
            f"Avg clustering={cumulative_avg_clusterings[i]:.4f}"
        )


if __name__ == "__main__":
    main()
