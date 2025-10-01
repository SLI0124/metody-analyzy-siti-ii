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
    for (u, v), _ in edge_weights.items():
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
    adj = defaultdict(set)
    for s in simplices:
        for u, v in itertools.combinations(s["nodes"], 2):
            adj[u].add(v)
            adj[v].add(u)

    clustering = []
    for _, neighbors in adj.items():
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
    filename="network_stats_over_time.png",
    title_prefix="",
    max_avg_edge_weights=None,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_plots = 4 if max_avg_edge_weights is not None else 3
    plt.figure(figsize=(16, 4) if n_plots == 4 else (12, 4))

    plt.subplot(1, n_plots, 1)
    plt.plot(years, avg_degrees, marker="o")
    plt.title(f"{title_prefix}Average Degree")
    plt.xlabel("Year")
    plt.ylabel("Average Degree")

    plt.subplot(1, n_plots, 2)
    plt.plot(years, avg_weighted_degrees, marker="o", color="orange")
    plt.title(f"{title_prefix}Average Weighted Degree")
    plt.xlabel("Year")
    plt.ylabel("Average Weighted Degree")

    plt.subplot(1, n_plots, 3)
    plt.plot(years, avg_clusterings, marker="o", color="green")
    plt.title(f"{title_prefix}Average Clustering Coefficient")
    plt.xlabel("Year")
    plt.ylabel("Average Clustering Coefficient")

    if max_avg_edge_weights is not None:
        plt.subplot(1, n_plots, 4)
        plt.plot(years, max_avg_edge_weights, marker="o", color="red")
        plt.title(f"{title_prefix}Max Avg Edge Weight")
        plt.xlabel("Year")
        plt.ylabel("Max Avg Edge Weight")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename)
    plt.close()


def plot_max_avg_edge_weight(years, max_avg_edge_weights, filename, title_prefix=""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(years, max_avg_edge_weights, marker="o", color="red")
    plt.title(f"{title_prefix}Max Avg Edge Weight Over Time")
    plt.xlabel("Year")
    plt.ylabel("Max Avg Edge Weight")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename)
    plt.close()


def compute_max_avg_edge_weight_over_time(frames, cumulative=False):
    max_avg_weights = []
    years = []
    all_simplices = []
    for ts in sorted(frames.keys()):
        simplices = all_simplices + frames[ts] if cumulative else frames[ts]
        if cumulative:
            all_simplices.extend(frames[ts])
        edge_weights = build_edge_weights(simplices)
        max_avg = 0
        for s in simplices:
            edges = [
                tuple(sorted((u, v)))
                for u, v in itertools.combinations(sorted(s["nodes"]), 2)
            ]
            if not edges:
                continue
            avg_weight = sum(edge_weights[e] for e in edges) / len(edges)
            if avg_weight > max_avg:
                max_avg = avg_weight
        max_avg_weights.append(max_avg)
        years.append(ts)
    return years, max_avg_weights


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
    nverts_path = DIR_PATH / "coauth-DBLP-nverts.txt"
    simplices_path = DIR_PATH / "coauth-DBLP-simplices.txt"
    times_path = DIR_PATH / "coauth-DBLP-times.txt"

    node_labels_content = load_file(node_labels_path)
    nverts_content = load_file(nverts_path)
    simplices_content = load_file(simplices_path)
    times_content = load_file(times_path)

    node_labels = parse_node_labels(node_labels_content)
    simplices = parse_simplices(nverts_content, simplices_content, times_content)
    frames = split_into_frames_by_timestamp(simplices)
    print(f"Number of frames (unique timestamps): {len(frames)}")

    # Try to load yearly statistics from CSV, else compute and save
    loaded_yearly = load_yearly_statistics_from_csv()
    if loaded_yearly:
        years, avg_degrees, avg_weighted_degrees, avg_clusterings = loaded_yearly
    else:
        years, avg_degrees, avg_weighted_degrees, avg_clusterings = (
            compute_yearly_statistics(frames)
        )

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

    # Compute max avg edge weight per year (frame)
    years_max, max_avg_edge_weights_yearly = compute_max_avg_edge_weight_over_time(
        frames, cumulative=False
    )
    plot_max_avg_edge_weight(
        years_max,
        max_avg_edge_weights_yearly,
        filename="max_avg_edge_weight_yearly.png",
        title_prefix="Yearly ",
    )

    plot_statistics(
        years,
        avg_degrees,
        avg_weighted_degrees,
        avg_clusterings,
        filename="network_stats_yearly.png",
        title_prefix="Yearly ",
        max_avg_edge_weights=max_avg_edge_weights_yearly,
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

    # Compute max avg edge weight for cumulative years
    cumulative_years_max, max_avg_edge_weights_cumulative = (
        compute_max_avg_edge_weight_over_time(frames, cumulative=True)
    )
    plot_max_avg_edge_weight(
        cumulative_years_max,
        max_avg_edge_weights_cumulative,
        filename="max_avg_edge_weight_cumulative.png",
        title_prefix="Cumulative ",
    )

    plot_statistics(
        cumulative_years,
        cumulative_avg_degrees,
        cumulative_avg_weighted_degrees,
        cumulative_avg_clusterings,
        filename="network_stats_cumulative.png",
        title_prefix="Cumulative ",
        max_avg_edge_weights=max_avg_edge_weights_cumulative,
    )


if __name__ == "__main__":
    main()
