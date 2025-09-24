from pathlib import Path
import itertools
from collections import defaultdict

DIR_PATH = Path("../data/coauth-DBLP/")


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

    for i, ts in enumerate(sorted(frames.keys()), 1):
        frame = frames[ts]
        edge_weights = build_edge_weights(frame)
        avg_deg = compute_avg_degree(edge_weights)
        avg_wdeg = compute_avg_weighted_degree(edge_weights)
        avg_clust = compute_avg_clustering_coefficient(frame)
        print(
            f"{i}/{len(frames)} Frame {ts}: Avg degree={avg_deg:.2f}, Avg weighted degree={avg_wdeg:.2f}, Avg clustering={avg_clust:.4f}"
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


if __name__ == "__main__":
    main()
