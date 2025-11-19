import math
import multiprocessing as mp
import random
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "results" / "task09"
FACEBOOK_DIR = OUTPUT_DIR / "facebook"
SMALL_OUTPUT_DIR = OUTPUT_DIR / "small_networks"

CANDIDATE_POOL_SIZE = 1000
DEFAULT_ACTIVATION_PROB = 0.02
DEFAULT_TOTAL_SIMULATIONS = 1000
DEFAULT_RANDOM_SEED = 2025
DEFAULT_ACTIVATION_PROBS = [0.02, 0.05, 0.1, 0.2]

NETWORK_NAME = "Facebook Penn94"
SMALL_NETWORK_NAME = "Karate Club"
SMALL_NETWORK_FILE = DATA_DIR / "edges karate.csv"


STATE_COLOR_MAP = {
    "S": "#9ecae1",
    "I": "#d62728",
    "R": "#31a354",
    "E": "#ffbf78",
    "Active": "#9467bd",
    "Inactive": "#9edae5",
}

MAX_SNAPSHOT_SUBPLOTS = 12


def ensure_output_dir(path=None):
    """Create the given directory if needed and return it."""
    directory = path or OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_network_data(file_path):
    """Read an edge list file and return a list of edge tuples."""

    path = Path(file_path)
    edges = []
    is_mtx = path.suffix.lower() == ".mtx"
    skip_dimension_line = is_mtx
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            cleaned = stripped.replace(";", " ").replace(",", " ")
            parts = cleaned.split()
            if skip_dimension_line:
                skip_dimension_line = False
                continue
            if len(parts) < 2:
                continue
            u, v = (int(parts[0]), int(parts[1]))
            if u != v:
                edges.append((u, v))
    return edges


def create_adjacency_dict(edges):
    """Convert an edge list into a symmetric adjacency mapping."""
    adjacency = {}
    for u, v in edges:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return {node: tuple(sorted(neighbors)) for node, neighbors in adjacency.items()}


def calculate_degrees(adjacency):
    """Return a degree dictionary for the provided adjacency mapping."""

    return {node: len(neighbors) for node, neighbors in adjacency.items()}


def rank_nodes_by_degree(degrees):
    """Return nodes sorted by degree and node id descending."""

    return [
        node
        for node, _ in sorted(
            degrees.items(), key=lambda item: (item[1], item[0]), reverse=True
        )
    ]


def iter_progress(iterable, description):
    """Yield items while printing simple textual progress updates."""

    items = list(iterable)
    total = len(items)
    if description:
        print(f"{description}: {total} item(s).")
    for index, item in enumerate(items, start=1):
        if description:
            percent = int((index / total) * 100)
            print(f"{description}: {index}/{total} ({percent}%)")
        yield item


def get_highest_degree_nodes(adjacency, k):
    """Return the top-k nodes by degree."""
    degrees = calculate_degrees(adjacency)
    ranked = rank_nodes_by_degree(degrees)

    return ranked[:k]


def get_network_stats(adjacency):
    """Compute basic size and degree statistics for a graph."""

    node_count = len(adjacency)
    edge_count = sum((len(neighbors) for neighbors in adjacency.values())) // 2
    degree_values = [len(neighbors) for neighbors in adjacency.values()]
    average_degree = float(np.mean(degree_values)) if degree_values else 0.0
    max_degree = max(degree_values) if degree_values else 0
    return {
        "nodes": node_count,
        "edges": edge_count,
        "avg_degree": average_degree,
        "max_degree": max_degree,
    }


def bfs_shortest_path_lengths(adjacency, source):
    """Compute BFS distances from the source node."""

    visited = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        depth = visited[node]
        for neighbor in adjacency.get(node, ()):
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append(neighbor)
    return visited


def select_farthest_apart_nodes(adjacency, candidate_nodes, degrees, k):
    """Pick seed nodes that are far apart while respecting degree order."""

    selected = [candidate_nodes[0]]
    distance_cache = {selected[0]: bfs_shortest_path_lengths(adjacency, selected[0])}
    candidate_distances = {}
    for node in candidate_nodes[1:]:
        candidate_distances[node] = float(
            distance_cache[selected[0]].get(node, math.inf)
        )
    while len(selected) < k and candidate_distances:
        best_node = max(
            candidate_distances.items(),
            key=lambda item: (item[1], degrees.get(item[0], 0)),
        )[0]
        if math.isinf(candidate_distances[best_node]):
            best_node = max(candidate_distances, key=lambda node: degrees.get(node, 0))
        selected.append(best_node)
        distance_cache[best_node] = bfs_shortest_path_lengths(adjacency, best_node)
        candidate_distances.pop(best_node, None)
        for node in list(candidate_distances.keys()):
            distance = float(distance_cache[best_node].get(node, math.inf))
            if distance < candidate_distances[node]:
                candidate_distances[node] = distance
    if len(selected) < k:
        remaining = [node for node in candidate_nodes if node not in selected]
        selected.extend(remaining[: k - len(selected)])
    return selected


def independent_cascade(
    adjacency, seed_nodes, activation_prob, rng, return_active_history=False
):
    """Run the Independent Cascade diffusion process using BFS approach."""

    active_nodes = set(seed_nodes)
    newly_active = set(seed_nodes)
    active_counts = [len(active_nodes)]
    newly_active_counts = [len(newly_active)]
    active_history = None
    if return_active_history:
        active_history = [tuple(sorted(active_nodes))]

    while newly_active:
        next_wave = set()
        for node in newly_active:
            for neighbor in adjacency.get(node, ()):
                if neighbor in active_nodes:
                    continue
                if rng.random() <= activation_prob:
                    next_wave.add(neighbor)
        if not next_wave:
            break
        active_nodes.update(next_wave)
        newly_active = next_wave
        active_counts.append(len(active_nodes))
        newly_active_counts.append(len(newly_active))
        if active_history is not None:
            active_history.append(tuple(sorted(active_nodes)))
    result = {
        "total_steps": len(active_counts) - 1,
        "final_active_count": len(active_nodes),
        "active_counts": active_counts,
        "newly_active_counts": newly_active_counts,
    }
    if active_history is not None:
        result["active_history"] = [list(history) for history in active_history]
    return result


_GLOBAL_ADJACENCY = {}


def _pool_initializer(adjacency):
    """Store adjacency globally for child processes."""

    global _GLOBAL_ADJACENCY
    _GLOBAL_ADJACENCY = {
        node: tuple(neighbors) for node, neighbors in adjacency.items()
    }


def _ic_worker(sim_indices, seed_nodes, activation_prob, base_seed):
    """Execute IC simulations for a chunk of indices."""

    results = []
    for sim_id in sim_indices:
        rng = random.Random(base_seed + sim_id)
        simulation = independent_cascade(
            _GLOBAL_ADJACENCY, seed_nodes, activation_prob, rng
        )
        simulation["simulation_id"] = sim_id
        results.append(simulation)
    return results


def _chunk_simulation_indices(total_simulations, n_workers):
    """Split simulation indices across the requested workers."""

    if total_simulations <= 0:
        return []
    base = total_simulations // n_workers
    remainder = total_simulations % n_workers
    indices = []
    current = 0
    for worker in range(n_workers):
        chunk_size = base + (1 if worker < remainder else 0)
        if chunk_size <= 0:
            continue
        chunk = list(range(current, current + chunk_size))
        indices.append(chunk)
        current += chunk_size
    return indices


def run_parallel_ic_simulations(
    adjacency,
    seed_nodes,
    activation_prob,
    total_simulations,
    n_workers=None,
    base_seed=DEFAULT_RANDOM_SEED,
):
    """Run multiple IC simulations, distributing work across processes."""

    if total_simulations <= 0:
        return []  # No simulations to run
    if n_workers is None:
        n_workers = max(1, min(mp.cpu_count(), total_simulations))
    else:
        n_workers = max(1, min(n_workers, total_simulations))
    print(
        "Dispatching IC simulations:",
        f"{total_simulations} runs across {n_workers} worker(s).",
    )
    if n_workers == 1:
        results = []
        for sim_id in range(total_simulations):
            rng = random.Random(base_seed + sim_id)
            simulation = independent_cascade(
                adjacency, seed_nodes, activation_prob, rng
            )
            simulation["simulation_id"] = sim_id
            results.append(simulation)
        return results
    chunked_indices = _chunk_simulation_indices(total_simulations, n_workers)
    with mp.Pool(
        processes=n_workers, initializer=_pool_initializer, initargs=(adjacency,)
    ) as pool:
        async_results = [
            pool.apply_async(
                _ic_worker, (chunk, tuple(seed_nodes), activation_prob, base_seed)
            )
            for chunk in chunked_indices
        ]
        collected = []
        for async_result in async_results:
            collected.extend(async_result.get())
    collected.sort(key=lambda item: item.get("simulation_id", 0))
    return collected


def _pad_series(series_list):
    """Pad variable-length series by repeating their final value."""

    if not series_list:
        return np.empty((0, 0), dtype=float)
    max_len = max((len(series) for series in series_list))
    padded = np.zeros((len(series_list), max_len), dtype=float)
    for row_idx, series in enumerate(series_list):
        last_value = float(series[-1])
        for col_idx in range(max_len):
            padded[row_idx, col_idx] = (
                float(series[col_idx]) if col_idx < len(series) else last_value
            )
    return padded


def summarize_time_series(
    series_list, network, method, k, activation_prob, value_label
):
    """Aggregate time-series statistics for a given method and seed size."""

    matrix = _pad_series(series_list)
    if matrix.size == 0:
        return pd.DataFrame()
    time_steps = np.arange(matrix.shape[1])
    summary = pd.DataFrame(
        {
            "network": network,
            "method": method,
            "k": k,
            "p": activation_prob,
            "time_step": time_steps,
            f"mean_{value_label}": matrix.mean(axis=0),
            f"median_{value_label}": np.median(matrix, axis=0),
            f"q1_{value_label}": np.percentile(matrix, 25, axis=0),
            f"q3_{value_label}": np.percentile(matrix, 75, axis=0),
        }
    )
    return summary


def build_final_results_dataframe(simulations, network, method, k, activation_prob):
    """Collect final reach metrics for each simulation into a DataFrame."""

    records = []
    for simulation in simulations:
        records.append(
            {
                "network": network,
                "method": method,
                "k": k,
                "p": activation_prob,
                "simulation_id": simulation.get("simulation_id"),
                "final_active_count": simulation.get("final_active_count"),
                "total_steps": simulation.get("total_steps"),
            }
        )
    return pd.DataFrame(records)


def compute_descriptive_stats(final_df, total_nodes):
    """Summarize distribution statistics of final activation counts."""

    if final_df.empty:
        return pd.DataFrame()
    stats_rows = []
    group_cols = ["network", "method", "k", "p"]
    grouped = final_df.groupby(group_cols)
    for group_values, group in grouped:
        counts = group["final_active_count"].to_numpy(dtype=float)
        stats_rows.append(
            {
                "network": group_values[0],
                "method": group_values[1],
                "k": group_values[2],
                "p": group_values[3],
                "simulations": int(group.shape[0]),
                "mean_final": float(np.mean(counts)),
                "std_final": float(np.std(counts, ddof=1)) if counts.size > 1 else 0.0,
                "min_final": float(np.min(counts)),
                "max_final": float(np.max(counts)),
                "median_final": float(np.median(counts)),
                "reach_ratio_mean": (
                    float(np.mean(counts) / total_nodes) if total_nodes else 0.0
                ),
                "reach_ratio_median": (
                    float(np.median(counts) / total_nodes) if total_nodes else 0.0
                ),
            }
        )
    return pd.DataFrame(stats_rows)


def aggregate_overall_statistics(final_df):
    """Build rollups of final influence per method and seed size."""

    if final_df.empty:
        return pd.DataFrame()
    records = []
    final_counts = final_df["final_active_count"].to_numpy(dtype=float)

    def _append_stats(grouped_df, level, method_val, k_val, subset):
        if subset.size == 0:
            return
        records.append(
            {
                "level": level,
                "method": method_val,
                "k": k_val,
                "simulations": int(grouped_df.shape[0]),
                "mean_final": float(np.mean(subset)),
                "std_final": float(np.std(subset, ddof=1)) if subset.size > 1 else 0.0,
                "min_final": float(np.min(subset)),
                "max_final": float(np.max(subset)),
                "median_final": float(np.median(subset)),
            }
        )

    _append_stats(final_df, "overall", None, None, final_counts)
    for method, group in final_df.groupby("method"):
        _append_stats(
            group,
            "method",
            method,
            None,
            group["final_active_count"].to_numpy(dtype=float),
        )
    for k_value, group in final_df.groupby("k"):
        _append_stats(
            group,
            "k",
            None,
            int(k_value),
            group["final_active_count"].to_numpy(dtype=float),
        )
    for (method, k_value), group in final_df.groupby(["method", "k"]):
        _append_stats(
            group,
            "method_k",
            method,
            int(k_value),
            group["final_active_count"].to_numpy(dtype=float),
        )
    return pd.DataFrame(records)


def build_seed_dataframe(network, method, k, seed_nodes, degrees):
    """Record the chosen seed nodes along with their degrees."""

    records = []
    for rank, node in enumerate(seed_nodes, start=1):
        records.append(
            {
                "network": network,
                "method": method,
                "k": k,
                "seed_rank": rank,
                "node_id": node,
                "degree": degrees.get(node, 0),
            }
        )
    return pd.DataFrame(records)


def build_sample_trace_dataframe(simulations, network, method, k):
    """Extract a representative simulation trace for plotting."""

    if not simulations:
        return pd.DataFrame()
    sample = simulations[0]
    active_series = sample.get("active_counts", [])
    new_series = sample.get("newly_active_counts", [])
    records = []
    for time_step, total_active in enumerate(active_series):
        newly_active = new_series[time_step] if time_step < len(new_series) else np.nan
        records.append(
            {
                "network": network,
                "method": method,
                "k": k,
                "time_step": time_step,
                "total_active": total_active,
                "newly_active": newly_active,
                "simulation_id": sample.get("simulation_id"),
            }
        )
    return pd.DataFrame(records)


def save_dataframe(dataframe, path):
    """Persist a dataframe to disk when it holds data."""

    if dataframe.empty:
        return None
    ensure_output_dir(path.parent)
    dataframe.to_csv(path, index=False)
    return path


def _scale_columns(dataframe, columns, factor):
    """Scale selected dataframe columns by a factor."""

    if dataframe.empty or not factor:
        return dataframe.copy()
    scaled = dataframe.copy()
    for column in columns:
        if column in scaled.columns:
            scaled[column] = scaled[column].astype(float) * factor
    return scaled


def scale_time_summary(dataframe, value_label, factor):
    """Multiply summary statistics by the provided factor."""

    columns = [
        f"mean_{value_label}",
        f"median_{value_label}",
        f"q1_{value_label}",
        f"q3_{value_label}",
    ]
    return _scale_columns(dataframe, columns, factor)


def scale_trace_dataframe(dataframe, columns, factor):
    """Scale trace dataframe columns by the provided factor."""

    return _scale_columns(dataframe, columns, factor)


def _method_k_layout(dataframe, sharex=True, sharey=True, base_size=(4.5, 3.2)):
    if dataframe.empty or "method" not in dataframe or "k" not in dataframe:
        return None
    methods = list(dict.fromkeys(dataframe["method"].tolist()))
    ks = sorted(dataframe["k"].unique())
    if not methods or not ks:
        return None
    fig, axes = plt.subplots(
        len(methods),
        len(ks),
        figsize=(base_size[0] * len(ks), base_size[1] * len(methods)),
        sharex=sharex,
        sharey=sharey,
    )
    axes_array = np.atleast_2d(axes)
    if axes_array.shape != (len(methods), len(ks)):
        axes_array = axes_array.reshape(len(methods), len(ks))
    return fig, axes_array, methods, ks


def _method_k_iter(axes, methods, ks):
    for row_idx, method in enumerate(methods):
        for col_idx, k in enumerate(ks):
            yield axes[row_idx][col_idx], method, k


def _finalize_method_k_plot(
    fig,
    axes,
    xlabel,
    ylabel,
    title,
    legend=False,
    legend_cols=2,
    tighten_rect=(0, 0, 1, 0.95),
):
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for row in axes:
        row[0].set_ylabel(ylabel)
    if legend:
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=legend_cols)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=tighten_rect)


def plot_time_series_grid(
    time_summary,
    network,
    output_path,
    value_label,
    axis_label=None,
):
    """Render mean and IQR time-series grids for each method and k."""

    layout = _method_k_layout(time_summary, sharex=True, sharey=True)
    if not layout:
        return
    fig, axes, methods, ks = layout
    axis_label = axis_label or value_label.replace("_", " ").title()
    for ax, method, k in _method_k_iter(axes, methods, ks):
        subset = time_summary[
            (time_summary["method"] == method) & (time_summary["k"] == k)
        ]
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.plot(
            subset["time_step"],
            subset[f"mean_{value_label}"],
            color="#1f77b4",
            label="Mean",
            linewidth=1.8,
        )
        ax.fill_between(
            subset["time_step"],
            subset[f"q1_{value_label}"],
            subset[f"q3_{value_label}"],
            color="#1f77b4",
            alpha=0.2,
            label="IQR",
        )
        ax.set_title(f"{method.replace('_', ' ').title()} (k={k})", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    _finalize_method_k_plot(
        fig,
        axes,
        "Time Step",
        axis_label,
        f"{network} – {axis_label} Over Time",
        legend=True,
    )
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_reach_grid(
    final_df,
    network,
    output_path,
    value_column="final_active_count",
    axis_label="Final Active Nodes",
):
    """Display histograms of final reach for every method/k pair."""

    layout = _method_k_layout(final_df, sharex=True, sharey=True)
    if not layout:
        return
    fig, axes, methods, ks = layout
    for ax, method, k in _method_k_iter(axes, methods, ks):
        subset = final_df[(final_df["method"] == method) & (final_df["k"] == k)]
        if subset.empty:
            ax.set_visible(False)
            continue
        counts = subset[value_column].to_numpy(dtype=float)
        ax.hist(counts, bins=20, color="#ff7f0e", alpha=0.75)
        ax.axvline(
            np.mean(counts),
            color="#d62728",
            linewidth=1.4,
            linestyle="--",
            label="Mean",
        )
        ax.set_title(f"{method.replace('_', ' ').title()} (k={k})", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    _finalize_method_k_plot(
        fig,
        axes,
        axis_label,
        "Frequency",
        f"{network} – Final Influence Spread Distributions",
        legend=True,
    )
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_trace_grid(
    sample_traces,
    network,
    output_path,
    total_column="total_active",
    new_column="newly_active",
    axis_label="Nodes",
):
    """Plot representative cumulative and incremental activation traces."""

    layout = _method_k_layout(sample_traces, sharex=True, sharey=False)
    if not layout:
        return
    fig, axes, methods, ks = layout
    for ax, method, k in _method_k_iter(axes, methods, ks):
        subset = sample_traces[
            (sample_traces["method"] == method) & (sample_traces["k"] == k)
        ]
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.plot(
            subset["time_step"],
            subset[total_column],
            color="#2ca02c",
            linewidth=1.6,
            label="Total active",
        )
        ax.bar(
            subset["time_step"],
            subset[new_column],
            color="#98df8a",
            alpha=0.5,
            label="Newly active",
        )
        ax.set_title(f"{method.replace('_', ' ').title()} (k={k})", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    _finalize_method_k_plot(
        fig,
        axes,
        "Time Step",
        axis_label,
        f"{network} – Representative Simulation Snapshots",
        legend=True,
    )
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_reach_boxplot(
    final_df,
    network,
    output_path,
    value_column="final_active_count",
    axis_label="Final Active Nodes",
):
    """Generate boxplots of final influence split by method and k."""

    layout = _method_k_layout(final_df, sharex=True, sharey=True)
    if not layout:
        return
    fig, axes, methods, ks = layout
    for ax, method, k in _method_k_iter(axes, methods, ks):
        subset = final_df[(final_df["method"] == method) & (final_df["k"] == k)]
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.boxplot(
            subset[value_column],
            patch_artist=True,
            tick_labels=[""],
            boxprops={"facecolor": "#c5b0d5", "alpha": 0.7},
            medianprops={"color": "#7f7f7f", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        ax.set_title(f"{method.replace('_', ' ').title()} (k={k})", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    _finalize_method_k_plot(
        fig,
        axes,
        axis_label,
        "Distribution",
        f"{network} – Final Reach Boxplots",
        legend=False,
        tighten_rect=(0, 0, 1, 0.92),
    )
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_group_boxplots(
    dataframe,
    network,
    output_path,
    group_col,
    label_fn,
    value_column,
    axis_label,
    xlabel,
    title_suffix,
    cmap_name="Set2",
):
    if dataframe.empty or group_col not in dataframe:
        return
    groups = list(dict.fromkeys(dataframe[group_col].tolist()))
    if not groups:
        return
    data = [
        dataframe[dataframe[group_col] == group][value_column].to_numpy(dtype=float)
        for group in groups
    ]
    if not any((array.size for array in data)):
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    box = ax.boxplot(
        data,
        tick_labels=[label_fn(group) for group in groups],
        patch_artist=True,
    )
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(box["boxes"])))[..., :3]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(axis_label)
    ax.set_title(f"{network} – {title_suffix}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_boxplots(
    final_df,
    network,
    output_path,
    value_column="final_active_count",
    axis_label="Final Active Nodes",
):
    """Compare final reach distributions across seed selection methods."""

    _plot_group_boxplots(
        dataframe=final_df,
        network=network,
        output_path=output_path,
        group_col="method",
        label_fn=lambda method: method.replace("_", " ").title(),
        value_column=value_column,
        axis_label=axis_label,
        xlabel="Method",
        title_suffix="Final Reach by Method",
        cmap_name="Set2",
    )


def plot_k_boxplots(
    final_df,
    network,
    output_path,
    value_column="final_active_count",
    axis_label="Final Active Nodes",
):
    """Compare final reach distributions across different k values."""

    _plot_group_boxplots(
        dataframe=final_df,
        network=network,
        output_path=output_path,
        group_col="k",
        label_fn=lambda k_value: str(int(k_value)),
        value_column=value_column,
        axis_label=axis_label,
        xlabel="k (Seed Set Size)",
        title_suffix="Final Reach by Seed Set Size",
        cmap_name="Paired",
    )


def plot_mean_time_series_by_method(
    time_df,
    network,
    output_path,
    value_column="mean_active",
    axis_label=None,
):
    """Plot average time-series trends per method."""

    if time_df.empty or value_column not in time_df.columns:
        return
    subset = time_df.groupby(["method", "time_step"])[value_column].mean().reset_index()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method, group in subset.groupby("method"):
        ordered = group.sort_values("time_step")
        ax.plot(
            ordered["time_step"],
            ordered[value_column],
            linewidth=1.8,
            label=method.replace("_", " ").title(),
        )
    ax.set_xlabel("Time Step")
    label = axis_label or value_column.replace("_", " ").title()
    ax.set_ylabel(label)
    ax.set_title(f"{network} – Average {label} by Method")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best")
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_boxplots_by_method(
    time_df,
    network,
    output_path,
    value_column="mean_active",
    max_time_steps=12,
    axis_label=None,
):
    """Show boxplots over time steps for each method."""

    if time_df.empty or value_column not in time_df.columns:
        return
    methods = list(dict.fromkeys(time_df["method"].tolist()))
    if not methods:
        return
    axis_title = axis_label or value_column.replace("_", " ").title()
    cols = min(3, len(methods))
    rows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows), sharey=True)
    axes_array = np.atleast_2d(axes)
    for idx, method in enumerate(methods):
        row_idx, col_idx = divmod(idx, cols)
        ax = axes_array[row_idx][col_idx]
        subset = time_df[time_df["method"] == method]
        if subset.empty:
            ax.axis("off")
            continue
        time_steps = sorted(subset["time_step"].unique())
        if max_time_steps > 0:
            time_steps = time_steps[:max_time_steps]
        data = []
        labels = []
        for step in time_steps:
            values = subset[subset["time_step"] == step][value_column].to_numpy()
            if values.size == 0:
                continue
            data.append(values)
            labels.append(str(int(step)))
        if not data:
            ax.axis("off")
            continue
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
        ax.set_title(f"{method.replace('_', ' ').title()}")
        ax.set_xlabel("Time Step")
        if col_idx == 0:
            ax.set_ylabel(axis_title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    total_axes = rows * cols
    for idx in range(len(methods), total_axes):
        row_idx, col_idx = divmod(idx, cols)
        axes_array[row_idx][col_idx].axis("off")
    fig.suptitle(f"{network} – Time-Step Boxplots ({axis_title})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _copy_state_snapshot(states):
    """Duplicate the state mapping to freeze a time slice."""

    return dict(states)


def simulate_si_model(adjacency, initial_infected, beta, max_steps, rng):
    """Run a stochastic SI epidemic simulation."""

    states = {node: "S" for node in adjacency}
    for node in initial_infected:
        states[node] = "I"
    history = [_copy_state_snapshot(states)]
    for _ in range(max_steps):
        new_infections = set()
        for node, state in states.items():
            if state != "I":
                continue
            for neighbor in adjacency.get(node, ()):
                if states.get(neighbor, "S") == "S" and rng.random() <= beta:
                    new_infections.add(neighbor)
        if not new_infections:
            break
        for node in new_infections:
            states[node] = "I"
        history.append(_copy_state_snapshot(states))
    if history[-1] != states:
        history.append(_copy_state_snapshot(states))
    return history


def simulate_sir_model(adjacency, initial_infected, beta, gamma, max_steps, rng):
    """Run a stochastic SIR simulation with recovery."""

    states = {node: "S" for node in adjacency}
    for node in initial_infected:
        states[node] = "I"
    history = [_copy_state_snapshot(states)]
    for _ in range(max_steps):
        new_infections = set()
        recoveries = set()
        for node, state in states.items():
            if state == "I":
                for neighbor in adjacency.get(node, ()):
                    if states.get(neighbor, "S") == "S" and rng.random() <= beta:
                        new_infections.add(neighbor)
                if rng.random() <= gamma:
                    recoveries.add(node)
        if not new_infections and (not recoveries):
            break
        for node in new_infections:
            states[node] = "I"
        for node in recoveries:
            states[node] = "R"
        history.append(_copy_state_snapshot(states))
    if history[-1] != states:
        history.append(_copy_state_snapshot(states))
    return history


def simulate_sis_model(adjacency, initial_infected, beta, gamma, max_steps, rng):
    """Run a stochastic SIS simulation with reinfection."""

    states = {node: "S" for node in adjacency}
    for node in initial_infected:
        states[node] = "I"
    history = [_copy_state_snapshot(states)]
    for _ in range(max_steps):
        new_infections = set()
        recoveries = set()
        for node, state in states.items():
            if state == "I":
                for neighbor in adjacency.get(node, ()):
                    if states.get(neighbor, "S") == "S" and rng.random() <= beta:
                        new_infections.add(neighbor)
                if rng.random() <= gamma:
                    recoveries.add(node)
        if not new_infections and (not recoveries):
            break
        for node in new_infections:
            states[node] = "I"
        for node in recoveries:
            states[node] = "S"
        history.append(_copy_state_snapshot(states))
    if history[-1] != states:
        history.append(_copy_state_snapshot(states))
    return history


def simulate_ic_history(adjacency, seed_nodes, activation_prob, rng):
    """Run IC diffusion and return the per-step active state history."""

    simulation = independent_cascade(
        adjacency, seed_nodes, activation_prob, rng, return_active_history=True
    )
    active_history = simulation.get("active_history", [])
    if not active_history:
        return []
    history = []
    nodes = sorted(adjacency.keys())
    for active_nodes in active_history:
        active_set = set(active_nodes)
        state_snapshot = {
            node: "Active" if node in active_set else "Inactive" for node in nodes
        }
        history.append(state_snapshot)
    return history


def history_to_state_dataframe(history, network, model):
    """Convert node-level state histories into a DataFrame."""

    records = []
    for time_step, state_map in enumerate(history):
        for node, state in state_map.items():
            records.append(
                {
                    "network": network,
                    "model": model,
                    "time_step": time_step,
                    "node_id": node,
                    "state": state,
                }
            )
    return pd.DataFrame(records)


def history_to_counts_dataframe(history, network, model, states):
    """Summarize counts per state for each step."""

    records = []
    for time_step, state_map in enumerate(history):
        counts = Counter(state_map.values())
        for state in states:
            records.append(
                {
                    "network": network,
                    "model": model,
                    "time_step": time_step,
                    "state": state,
                    "count": counts.get(state, 0),
                }
            )
    return pd.DataFrame(records)


def plot_state_snapshots(
    history, graph, positions, model, output_path, state_sequence, title_prefix
):
    """Visualize selected time slices of a diffusion history."""

    if not history:
        return
    total_steps = len(history)
    if total_steps <= MAX_SNAPSHOT_SUBPLOTS:
        indices = list(range(total_steps))
    else:
        indices = sorted(
            set(np.linspace(0, total_steps - 1, MAX_SNAPSHOT_SUBPLOTS, dtype=int))
        )
    cols = min(4, max(1, math.ceil(math.sqrt(len(indices)))))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows))
    if rows == 1 and cols == 1:
        axes_array = np.array([axes])
    else:
        axes_array = np.atleast_2d(axes)
    nodes = sorted(graph.nodes())
    for idx, step in enumerate(indices):
        row_idx, col_idx = divmod(idx, cols)
        ax = axes_array[row_idx][col_idx]
        states = history[step]
        colors = [
            STATE_COLOR_MAP.get(states.get(node, "S"), "#cccccc") for node in nodes
        ]
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_color=colors, node_size=260, ax=ax
        )
        nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.4)
        ax.set_title(f"t={step}", fontsize=10)
        ax.axis("off")
    total_plots = rows * cols
    for idx in range(len(indices), total_plots):
        row_idx, col_idx = divmod(idx, cols)
        axes_array[row_idx][col_idx].axis("off")
    legend_handles = []
    for state in state_sequence:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=state,
                markerfacecolor=STATE_COLOR_MAP.get(state, "#cccccc"),
                markersize=10,
            )
        )
    fig.legend(
        legend_handles,
        list(state_sequence),
        loc="upper center",
        ncol=len(state_sequence),
    )
    fig.suptitle(f"{title_prefix} – {model.upper()} Snapshots", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_state_counts(counts_df, model, output_path, state_sequence, title_prefix):
    """Plot average state counts over time for small-network models."""

    if counts_df.empty:
        return
    pivot = (
        counts_df.pivot_table(
            index="time_step", columns="state", values="count", aggfunc="mean"
        )
        .reindex(columns=state_sequence)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for state in state_sequence:
        if state not in pivot:
            continue
        ax.plot(
            pivot.index,
            pivot[state],
            label=state,
            linewidth=1.8,
            color=STATE_COLOR_MAP.get(state, "#333333"),
        )
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Nodes")
    ax.set_title(f"{title_prefix} – {model.upper()} State Counts")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right")
    ensure_output_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_small_network_snapshots(
    data_path=None, output_dir=None, skip_existing=True, random_seed=DEFAULT_RANDOM_SEED
):
    """Simulate SI/SIR/SIS/IC models on the toy network and persist outputs."""

    target_dir = ensure_output_dir(output_dir or SMALL_OUTPUT_DIR)
    path = Path(data_path or SMALL_NETWORK_FILE)
    edges = load_network_data(path)
    adjacency = create_adjacency_dict(edges)
    if not adjacency:
        raise ValueError("Small network adjacency is empty.")
    graph = nx.Graph()
    graph.add_nodes_from(adjacency.keys())
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            if node <= neighbor:
                graph.add_edge(node, neighbor)
    positions = nx.spring_layout(graph, seed=random_seed)
    initial_infected = get_highest_degree_nodes(adjacency, 2)
    if not initial_infected:
        raise ValueError(
            "Unable to determine initial infected nodes for small network."
        )
    print(f"Using initial infected nodes {initial_infected} for {SMALL_NETWORK_NAME}.")
    stats = get_network_stats(adjacency)
    results = {"network_stats": dict(stats)}
    model_names = ["si", "sir", "sis", "ic"]
    state_sequences = {
        "si": ("S", "I"),
        "sir": ("S", "I", "R"),
        "sis": ("S", "I"),
        "ic": ("Inactive", "Active"),
    }
    for index, model_name in enumerate(model_names):
        state_sequence = state_sequences[model_name]
        filenames = {
            "states_csv": target_dir / f"karate_{model_name}_states.csv",
            "counts_csv": target_dir / f"karate_{model_name}_counts.csv",
            "snapshot_png": target_dir / f"karate_{model_name}_snapshots.png",
            "counts_png": target_dir / f"karate_{model_name}_state_counts.png",
        }
        if skip_existing and all((path.exists() for path in filenames.values())):
            message = (
                f"Skipping {model_name.upper()} for {SMALL_NETWORK_NAME} because "
                "the files are already saved."
            )
            print(message)
            results[model_name] = filenames
            continue
        rng = random.Random(random_seed + index)
        if model_name == "si":
            history = simulate_si_model(
                adjacency, initial_infected, beta=0.5, max_steps=40, rng=rng
            )
        elif model_name == "sir":
            history = simulate_sir_model(
                adjacency,
                initial_infected,
                beta=0.45,
                gamma=0.35,
                max_steps=45,
                rng=rng,
            )
        elif model_name == "sis":
            history = simulate_sis_model(
                adjacency, initial_infected, beta=0.4, gamma=0.3, max_steps=45, rng=rng
            )
        else:
            best_history = []
            attempts = 8
            for attempt in range(attempts):
                attempt_rng = random.Random(random_seed + index * 100 + attempt)
                trial = simulate_ic_history(
                    adjacency, initial_infected, activation_prob=0.15, rng=attempt_rng
                )
                if len(trial) > len(best_history):
                    best_history = trial
            history = best_history
        if not history:
            warning = (
                f"Warning: {model_name.upper()} on {SMALL_NETWORK_NAME} did not "
                "produce any steps, skipping outputs."
            )
            print(warning)
            results[model_name] = {key: None for key in filenames}
            continue
        states_df = history_to_state_dataframe(history, SMALL_NETWORK_NAME, model_name)
        counts_df = history_to_counts_dataframe(
            history, SMALL_NETWORK_NAME, model_name, state_sequence
        )
        state_path = save_dataframe(states_df, filenames["states_csv"])
        counts_path = save_dataframe(counts_df, filenames["counts_csv"])
        plot_state_snapshots(
            history,
            graph,
            positions,
            model_name,
            filenames["snapshot_png"],
            state_sequence,
            SMALL_NETWORK_NAME,
        )
        plot_state_counts(
            counts_df,
            model_name,
            filenames["counts_png"],
            state_sequence,
            SMALL_NETWORK_NAME,
        )
        summary = (
            f"Saved {model_name.upper()} snapshots with {len(history)} time steps "
            f"for {SMALL_NETWORK_NAME}."
        )
        print(summary)
        results[model_name] = {
            "states_csv": state_path,
            "counts_csv": counts_path,
            "snapshot_png": filenames["snapshot_png"],
            "counts_png": filenames["counts_png"],
        }
    return results


def run_facebook_influence_pipeline(
    data_path=None,
    k_values=(2, 4, 10),
    candidate_pool_size=CANDIDATE_POOL_SIZE,
    total_simulations=DEFAULT_TOTAL_SIMULATIONS,
    activation_prob=DEFAULT_ACTIVATION_PROB,
    n_workers=None,
    random_seed=DEFAULT_RANDOM_SEED,
    skip_existing=True,
    output_dir=None,
):
    """Execute the Facebook IC benchmarking suite and save artifacts."""

    facebook_dir = ensure_output_dir(output_dir or FACEBOOK_DIR)

    path = Path(data_path or DATA_DIR / "socfb-Penn94.mtx")
    edges = load_network_data(path)
    adjacency = create_adjacency_dict(edges)
    if not adjacency:
        raise ValueError("Facebook network adjacency is empty.")

    degrees = calculate_degrees(adjacency)
    stats = get_network_stats(adjacency)
    total_nodes = stats.get("nodes", 0)
    percent_factor = (100.0 / total_nodes) if total_nodes else 0.0

    data_files = {
        "final_results": facebook_dir / "ic_simulations.csv",
        "time_series": facebook_dir / "ic_time_series_summary.csv",
        "newly_series": facebook_dir / "ic_newly_summary.csv",
        "seed_sets": facebook_dir / "ic_seed_sets.csv",
        "summary_stats": facebook_dir / "ic_statistics.csv",
        "sample_trace": facebook_dir / "ic_sample_trace.csv",
        "overall_stats": facebook_dir / "ic_overall_stats.csv",
    }

    plot_files = {
        "active_timeseries": facebook_dir / "ic_active_timeseries.png",
        "newly_timeseries": facebook_dir / "ic_newly_timeseries.png",
        "histogram": facebook_dir / "ic_final_distribution.png",
        "sample_traces": facebook_dir / "ic_sample_traces.png",
        "final_boxplot": facebook_dir / "ic_final_boxplots.png",
        "method_boxplots": facebook_dir / "ic_method_boxplots.png",
        "k_boxplots": facebook_dir / "ic_k_boxplots.png",
        "time_method_active": facebook_dir / "ic_time_method_active.png",
        "time_boxplot_active": facebook_dir / "ic_time_boxplots_active.png",
        "time_method_newly": facebook_dir / "ic_time_method_newly.png",
        "time_boxplot_newly": facebook_dir / "ic_time_boxplots_newly.png",
    }

    def read_dataframe(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    skip_simulations = skip_existing and data_files["final_results"].exists()

    if skip_simulations:
        print("Skipping Facebook IC simulations because earlier results were found.")
        final_df = read_dataframe(data_files["final_results"])
        time_df = read_dataframe(data_files["time_series"])
        newly_df = read_dataframe(data_files["newly_series"])
        seeds_df = read_dataframe(data_files["seed_sets"])
        traces_df = read_dataframe(data_files["sample_trace"])
        summary_stats_df = read_dataframe(data_files["summary_stats"])
        if summary_stats_df.empty and not final_df.empty:
            summary_stats_df = compute_descriptive_stats(final_df, total_nodes)
    else:
        ranked_nodes = rank_nodes_by_degree(degrees)
        candidate_nodes = ranked_nodes[:candidate_pool_size]
        result_frames = []
        time_frames = []
        newly_frames = []
        seed_frames = []
        trace_frames = []

        methods = (
            ("highest_degree", lambda k: candidate_nodes[:k]),
            (
                "distance_greedy",
                lambda k: select_farthest_apart_nodes(
                    adjacency, candidate_nodes, degrees, k
                ),
            ),
        )
        for method_name, selector in iter_progress(methods, "Seed selection methods"):
            for k in iter_progress(k_values, f"{method_name} – k loop"):
                seed_nodes = selector(k)
                if not seed_nodes:
                    warning = (
                        "Warning: no seed nodes were generated for "
                        f"method={method_name}, k={k}. Skipping this combination."
                    )
                    print(warning)
                    continue
                if len(seed_nodes) < k:
                    shortage = (
                        f"Warning: method {method_name} returned only "
                        f"{len(seed_nodes)} seeds for k={k}."
                    )
                    print(shortage)
                print(
                    f"Running {total_simulations} simulations for "
                    f"method={method_name}, k={k}."
                )
                seed_frames.append(
                    build_seed_dataframe(
                        NETWORK_NAME, method_name, k, seed_nodes, degrees
                    )
                )
                simulations = run_parallel_ic_simulations(
                    adjacency=adjacency,
                    seed_nodes=seed_nodes,
                    activation_prob=activation_prob,
                    total_simulations=total_simulations,
                    n_workers=n_workers,
                    base_seed=random_seed,
                )
                result_frames.append(
                    build_final_results_dataframe(
                        simulations, NETWORK_NAME, method_name, k, activation_prob
                    )
                )
                active_series = [sim.get("active_counts", []) for sim in simulations]
                newly_series = [
                    sim.get("newly_active_counts", []) for sim in simulations
                ]
                time_frames.append(
                    summarize_time_series(
                        active_series,
                        NETWORK_NAME,
                        method_name,
                        k,
                        activation_prob,
                        "active",
                    )
                )
                newly_frames.append(
                    summarize_time_series(
                        newly_series,
                        NETWORK_NAME,
                        method_name,
                        k,
                        activation_prob,
                        "newly_active",
                    )
                )
                trace_frames.append(
                    build_sample_trace_dataframe(
                        simulations, NETWORK_NAME, method_name, k
                    )
                )
        final_df = (
            pd.concat(result_frames, ignore_index=True)
            if result_frames
            else pd.DataFrame()
        )
        time_df = (
            pd.concat(time_frames, ignore_index=True) if time_frames else pd.DataFrame()
        )
        newly_df = (
            pd.concat(newly_frames, ignore_index=True)
            if newly_frames
            else pd.DataFrame()
        )
        seeds_df = (
            pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
        )
        traces_df = (
            pd.concat(trace_frames, ignore_index=True)
            if trace_frames
            else pd.DataFrame()
        )
        summary_stats_df = compute_descriptive_stats(final_df, total_nodes)

    if final_df.empty:
        raise RuntimeError("No simulation results generated.")

    overall_stats_df = aggregate_overall_statistics(final_df)

    saved_data = {name: None for name in data_files}
    if skip_simulations:
        for name, path in data_files.items():
            if path.exists():
                saved_data[name] = path
        if saved_data["summary_stats"] is None and not summary_stats_df.empty:
            saved_data["summary_stats"] = save_dataframe(
                summary_stats_df, data_files["summary_stats"]
            )
        if saved_data["overall_stats"] is None and not overall_stats_df.empty:
            saved_data["overall_stats"] = save_dataframe(
                overall_stats_df, data_files["overall_stats"]
            )
    else:
        for name, dataframe in {
            "final_results": final_df,
            "time_series": time_df,
            "newly_series": newly_df,
            "seed_sets": seeds_df,
            "summary_stats": summary_stats_df,
            "sample_trace": traces_df,
            "overall_stats": overall_stats_df,
        }.items():
            saved_data[name] = save_dataframe(dataframe, data_files[name])

    def scale_series(series):
        values = series.astype(float)
        return values * percent_factor if percent_factor else values

    active_axis_label = "Active (% of nodes)" if percent_factor else "Active Nodes"
    newly_axis_label = (
        "Newly Active (% of nodes)" if percent_factor else "Newly Active Nodes"
    )
    final_axis_label = (
        "Final Active (% of nodes)" if percent_factor else "Final Active Nodes"
    )

    time_pct_df = scale_time_summary(time_df, "active", percent_factor)
    newly_pct_df = scale_time_summary(newly_df, "newly_active", percent_factor)
    traces_pct_df = scale_trace_dataframe(
        traces_df, ["total_active", "newly_active"], percent_factor
    )
    final_pct_df = final_df.copy()
    final_pct_df["final_active_pct"] = scale_series(final_df["final_active_count"])

    def already_rendered(path):
        return skip_existing and skip_simulations and path.exists()

    plot_results = {name: None for name in plot_files}

    plot_specs = [
        {
            "key": "active_timeseries",
            "path": plot_files["active_timeseries"],
            "func": plot_time_series_grid,
            "args": (
                time_pct_df,
                NETWORK_NAME,
                plot_files["active_timeseries"],
                "active",
            ),
            "kwargs": {"axis_label": active_axis_label},
            "empty": time_df.empty,
            "empty_message": (
                "Time-series summary is empty, so the active-node plot is being "
                "skipped."
            ),
            "skip_message": (
                "Active time-series plot already exists, skipping regeneration."
            ),
        },
        {
            "key": "newly_timeseries",
            "path": plot_files["newly_timeseries"],
            "func": plot_time_series_grid,
            "args": (
                newly_pct_df,
                NETWORK_NAME,
                plot_files["newly_timeseries"],
                "newly_active",
            ),
            "kwargs": {"axis_label": newly_axis_label},
            "empty": newly_df.empty,
            "empty_message": (
                "Newly active summary is empty, so that plot is being skipped."
            ),
            "skip_message": (
                "Newly active plot already exists, skipping regeneration."
            ),
        },
        {
            "key": "histogram",
            "path": plot_files["histogram"],
            "func": plot_final_reach_grid,
            "args": (
                final_pct_df,
                NETWORK_NAME,
                plot_files["histogram"],
            ),
            "kwargs": {
                "value_column": "final_active_pct",
                "axis_label": final_axis_label,
            },
            "empty": final_pct_df.empty,
            "skip_message": "Histogram plot already exists, skipping regeneration.",
        },
        {
            "key": "sample_traces",
            "path": plot_files["sample_traces"],
            "func": plot_sample_trace_grid,
            "args": (
                traces_pct_df,
                NETWORK_NAME,
                plot_files["sample_traces"],
            ),
            "kwargs": {"axis_label": active_axis_label},
            "empty": traces_df.empty,
            "empty_message": (
                "Sample trace summary is empty, so the plot is being skipped."
            ),
            "skip_message": (
                "Sample trace plot already exists, skipping regeneration."
            ),
        },
        {
            "key": "final_boxplot",
            "path": plot_files["final_boxplot"],
            "func": plot_final_reach_boxplot,
            "args": (
                final_pct_df,
                NETWORK_NAME,
                plot_files["final_boxplot"],
            ),
            "kwargs": {
                "value_column": "final_active_pct",
                "axis_label": final_axis_label,
            },
            "empty": final_pct_df.empty,
            "skip_message": (
                "Final reach boxplot already exists, so it is left untouched."
            ),
        },
        {
            "key": "method_boxplots",
            "path": plot_files["method_boxplots"],
            "func": plot_method_boxplots,
            "args": (
                final_pct_df,
                NETWORK_NAME,
                plot_files["method_boxplots"],
            ),
            "kwargs": {
                "value_column": "final_active_pct",
                "axis_label": final_axis_label,
            },
            "empty": final_pct_df.empty,
            "skip_message": (
                "Method summary boxplots already exist, skipping regeneration."
            ),
        },
        {
            "key": "k_boxplots",
            "path": plot_files["k_boxplots"],
            "func": plot_k_boxplots,
            "args": (
                final_pct_df,
                NETWORK_NAME,
                plot_files["k_boxplots"],
            ),
            "kwargs": {
                "value_column": "final_active_pct",
                "axis_label": final_axis_label,
            },
            "empty": final_pct_df.empty,
            "skip_message": (
                "K summary boxplots already exist, skipping regeneration."
            ),
        },
        {
            "key": "time_method_active",
            "path": plot_files["time_method_active"],
            "func": plot_mean_time_series_by_method,
            "args": (
                time_pct_df,
                NETWORK_NAME,
                plot_files["time_method_active"],
            ),
            "kwargs": {
                "value_column": "mean_active",
                "axis_label": active_axis_label,
            },
            "empty": time_df.empty,
            "skip_message": (
                "Method-by-time active trend already exists, skipping regeneration."
            ),
        },
        {
            "key": "time_boxplot_active",
            "path": plot_files["time_boxplot_active"],
            "func": plot_time_boxplots_by_method,
            "args": (
                time_pct_df,
                NETWORK_NAME,
                plot_files["time_boxplot_active"],
            ),
            "kwargs": {
                "value_column": "mean_active",
                "axis_label": active_axis_label,
            },
            "empty": time_df.empty,
            "skip_message": (
                "Time boxplots for active counts already exist, skipping "
                "regeneration."
            ),
        },
        {
            "key": "time_method_newly",
            "path": plot_files["time_method_newly"],
            "func": plot_mean_time_series_by_method,
            "args": (
                newly_pct_df,
                NETWORK_NAME,
                plot_files["time_method_newly"],
            ),
            "kwargs": {
                "value_column": "mean_newly_active",
                "axis_label": newly_axis_label,
            },
            "empty": newly_df.empty,
            "skip_message": (
                "Method-by-time newly active trend already exists, skipping "
                "regeneration."
            ),
        },
        {
            "key": "time_boxplot_newly",
            "path": plot_files["time_boxplot_newly"],
            "func": plot_time_boxplots_by_method,
            "args": (
                newly_pct_df,
                NETWORK_NAME,
                plot_files["time_boxplot_newly"],
            ),
            "kwargs": {
                "value_column": "mean_newly_active",
                "axis_label": newly_axis_label,
            },
            "empty": newly_df.empty,
            "skip_message": (
                "Time boxplots for newly active counts already exist, skipping "
                "regeneration."
            ),
        },
    ]

    for spec in plot_specs:
        key = spec["key"]
        path = spec["path"]

        if already_rendered(path):
            message = spec.get(
                "skip_message",
                f"{key.replace('_', ' ')} already exists, skipping.",
            )
            print(message)
            plot_results[key] = path
            continue

        if spec.get("empty"):
            message = spec.get("empty_message")
            if message:
                print(message)
            plot_results[key] = None
            continue

        spec["func"](*spec["args"], **spec.get("kwargs", {}))
        plot_results[key] = path if path.exists() else None

    return {
        "network_stats": stats,
        "data_files": saved_data,
        "plot_files": plot_results,
    }


def run_facebook_probability_scenarios(
    probabilities=None,
    base_output_dir=None,
    **pipeline_kwargs,
):
    """Run the Facebook IC pipeline for multiple activation probabilities."""

    probability_list = (
        list(probabilities)
        if probabilities is not None
        else list(DEFAULT_ACTIVATION_PROBS)
    )
    if not probability_list:
        raise ValueError("At least one activation probability is required.")
    pipeline_kwargs = dict(pipeline_kwargs)
    pipeline_kwargs.pop("output_dir", None)
    base_dir = ensure_output_dir(base_output_dir or FACEBOOK_DIR)
    scenario_results = {}
    for prob in iter_progress(probability_list, "Activation probabilities"):
        label = f"p_{prob:.3f}".rstrip("0").rstrip(".")
        scenario_dir = base_dir / label
        print(
            "Starting Facebook simulations for activation probability=" f"{prob:.3f}."
        )
        scenario_results[prob] = run_facebook_influence_pipeline(
            activation_prob=prob,
            output_dir=scenario_dir,
            **pipeline_kwargs,
        )
        print(
            "Finished Facebook simulations for activation probability=" f"{prob:.3f}."
        )
    return scenario_results


def main():
    """Entry point that runs both the toy and Facebook pipelines."""

    print("Starting small-network epidemic simulations...")
    run_small_network_snapshots()
    print("Small-network simulations finished.")
    print("Starting Facebook influence simulations...")
    run_facebook_probability_scenarios()
    print("Facebook influence simulations finished.")


if __name__ == "__main__":
    main()
