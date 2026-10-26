"""Find groups (cliques) of neurons with correlated firing activity.

Pipeline
--------
1. Binarize the activity of every neuron in [start_time_ms, end_time_ms] with
   bins of bin_size_ms (True = at least one spike in the bin).
2. Keep the "active" neurons, i.e. those firing in more than
   min_active_bins_fraction of the bins.
3. Compute the Hamming distance between the binary vectors of every pair of
   active neurons.
4. Two neurons are correlated if their distance is below
   max_different_bins_fraction * n_bins: this defines a correlation graph.
5. For every connected component of the graph larger than
   min_connected_component_size, repeatedly extract the largest maximal
   clique and discard the cliques overlapping it.
6. For the cliques larger than min_clique_size compute ISI distributions and
   save rasters / 3D positions plots.

Usage: python find_correlated_neuron_cliques.py [path/to/clique_configuration.json]
"""
import json
import os
import pickle
import sys

import h5py
import matplotlib

matplotlib.use("Agg")  # figures are only saved to file: no interactive (Tk) window
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clique_configuration.json")
PATH_KEYS = ["spikes_file", "positions_file", "results_dir"]

COLOR_PALETTE = [
    "orange", "red", "green", "blue", "pink", "yellow", "springgreen", "sienna",
    "linen", "magenta", "skyblue", "maroon", "moccasin", "olivedrab", "teal",
    "thistle", "tomato", "turquoise", "ivory", "khaki", "lavender",
    "lavenderblush", "lawngreen",
]
TRANSPARENT = "rgba(0,0,0,0)"
N_BACKGROUND_NEURONS = 5000  # neurons drawn at random (all populations) as background of the 3D plot


def clique_color(position):
    """Color of the clique at `position` in the list of selected cliques."""
    return COLOR_PALETTE[position % len(COLOR_PALETTE)]


# ---------------------------------------------------------------- configuration

def load_config(argv=None):
    """Load the JSON configuration (first command line argument, or the file next to this script).

    Every path in the configuration must be absolute.
    """
    argv = sys.argv if argv is None else argv
    config_file = os.path.abspath(argv[1]) if len(argv) > 1 else DEFAULT_CONFIG_FILE
    with open(config_file, "r") as f:
        config = json.load(f)

    for key in PATH_KEYS:
        if key not in config:
            raise KeyError("Missing path '%s' in %s" % (key, config_file))
        if not os.path.isabs(config[key]):
            raise ValueError("Path '%s' in %s must be absolute, got '%s'" % (key, config_file, config[key]))
        config[key] = os.path.normpath(config[key])

    print("configuration loaded from", config_file)
    return config


def results_folders(config):
    """Return (results_path, results_sub_path) and create them if needed.

    results_path depends on the time window and bin size, results_sub_path
    additionally on the activity and correlation thresholds.
    """
    results_path = os.path.join(
        config["results_dir"],
        "clique_interval_%s_%s_bin_%s" % (config["start_time_ms"], config["end_time_ms"], config["bin_size_ms"]),
    )
    results_sub_path = os.path.join(
        results_path, "active_%s_corr_%s" % (config["min_active_bins_fraction"], config["max_different_bins_fraction"]))
    os.makedirs(results_sub_path, exist_ok=True)
    return results_path, results_sub_path


# ---------------------------------------------------------------- input data

def load_spikes(spikes_file):
    """Read the (neuron_id, spike_time) array from the simulation activity file."""
    with h5py.File(spikes_file, "r") as f:
        return f["spikes"][:].astype("float64")


def load_positions(positions_file):
    """Positions of all the neurons (all populations), columns [neuron_id, x, y, z]."""
    with h5py.File(positions_file, "r") as f:
        return np.concatenate([f[key][:] for key in f.keys()])


class SpikeIndex:
    """Fast per-neuron access to the spike list.

    Spikes are sorted by neuron id and then by time, so the rows of the
    activity file can be in any order.
    """

    def __init__(self, spike_list):
        order = np.lexsort((spike_list[:, 1], spike_list[:, 0]))
        self.neuron_ids = spike_list[order, 0]
        self.spike_times = spike_list[order, 1]

    def spikes_of(self, neuron_id):
        lo = np.searchsorted(self.neuron_ids, neuron_id, side="left")
        hi = np.searchsorted(self.neuron_ids, neuron_id, side="right")
        return self.spike_times[lo:hi]


# ---------------------------------------------------------------- analysis

def compute_spiking_matrix(spike_list, t_start, bin_size, n_bins):
    """Bool array (n_rows, n_bins): True if the neuron fires at least once in the bin.

    Bin b covers [t_start + b * bin_size, t_start + (b + 1) * bin_size): a spike
    falling exactly on an edge belongs to the bin that starts there.
    Row = neuron id. n_rows = largest id of the neurons firing in the window + 1:
    neurons with a larger id never fire in the window, so they could never be active.
    """
    bin_index = np.floor((spike_list[:, 1] - t_start) / bin_size)
    inside = (bin_index >= 0) & (bin_index < n_bins)
    neuron_ids = spike_list[inside, 0].astype(int)
    n_rows = neuron_ids.max() + 1 if neuron_ids.size else 0
    is_spiking = np.zeros((n_rows, n_bins), dtype=bool)
    is_spiking[neuron_ids, bin_index[inside].astype(int)] = True
    return is_spiking


def row_blocks(n_rows, n_columns, max_elements=10_000_000):
    """Row ranges whose blocks hold at most ~max_elements entries of an (n_rows, n_columns) matrix."""
    rows_per_block = max(1, max_elements // max(n_columns, 1))
    for start in range(0, n_rows, rows_per_block):
        yield start, min(start + rows_per_block, n_rows)


def compute_hamming_distances(active_spiking):
    """Pairwise Hamming distances between the rows of the bool matrix active_spiking.

    distance(a, b) = |a| + |b| - 2 a.b, computed by row blocks with a matrix
    product (exact for n_bins < 2**24). The matrix uses the smallest unsigned
    integer type that holds n_bins (1 byte per pair if n_bins < 256).
    """
    n_active, n_bins = active_spiking.shape
    spiking = active_spiking.astype(np.float32)
    n_firing_bins = spiking.sum(1)
    hamming_distances = np.empty((n_active, n_active), dtype=np.uint8 if n_bins < 256 else np.uint16)
    for start, stop in row_blocks(n_active, n_active):
        common_bins = spiking[start:stop] @ spiking.T
        distances = n_firing_bins[start:stop, None] + n_firing_bins[None, :] - 2 * common_bins
        hamming_distances[start:stop] = np.rint(distances)
        print("hamming distances: %d / %d rows" % (stop, n_active))
    return hamming_distances


def hamming_distance_counts(hamming_distances):
    """counts[d] = number of ordered neuron pairs at distance d (computed by row blocks)."""
    counts = np.zeros(int(hamming_distances.max()) + 1, dtype=np.int64)
    for start, stop in row_blocks(*hamming_distances.shape):
        counts += np.bincount(hamming_distances[start:stop].ravel(), minlength=counts.shape[0])
    return counts


def build_correlation_graph(hamming_distances, correlation_threshold):
    """Sparse adjacency matrix: i != j connected if hamming_distances[i, j] < correlation_threshold."""
    n_active = hamming_distances.shape[0]
    rows, columns = [np.array([], dtype=np.int32)], [np.array([], dtype=np.int32)]
    for start, stop in row_blocks(n_active, n_active):
        block_rows, block_columns = np.nonzero(hamming_distances[start:stop] < correlation_threshold)
        block_rows += start
        not_self = block_rows != block_columns
        rows.append(block_rows[not_self].astype(np.int32))
        columns.append(block_columns[not_self].astype(np.int32))
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    return csr_matrix((np.ones(rows.shape[0], dtype=bool), (rows, columns)), shape=(n_active, n_active))


def find_cliques_per_component(correlation_graph, min_component_size):
    """Connected components larger than min_component_size and their maximal cliques.

    A networkx graph is built only for each of these components, never for the whole network.
    """
    _, component_labels = connected_components(csgraph=correlation_graph, directed=False, return_labels=True)

    connected_component_ids = []
    cliques_per_component = []
    for label in np.unique(component_labels):
        component_neurons = np.where(component_labels == label)[0]
        if component_neurons.shape[0] > min_component_size:
            connected_component_ids.append(label)
            component_matrix = correlation_graph[component_neurons][:, component_neurons]
            print("connected component", label, "size", component_neurons.shape[0], "edges", component_matrix.nnz // 2)
            cliques_per_component.append(list(nx.find_cliques(nx.from_scipy_sparse_array(component_matrix))))
    return connected_component_ids, cliques_per_component, component_labels


def extract_disjoint_cliques(active_neurons, component_labels, connected_component_ids, cliques_per_component):
    """Greedy selection of disjoint cliques inside each connected component.

    The largest clique is taken, all cliques sharing a neuron with it are
    discarded, and the procedure is repeated until no clique is left.
    Returns a flat list with the neuron ids of every selected clique.
    """
    clique_neuron_ids = []
    for component_id, cliques in zip(connected_component_ids, cliques_per_component):
        component_neurons = active_neurons[component_labels == component_id]
        remaining = list(cliques)
        while remaining:
            largest = remaining[int(np.argmax([len(c) for c in remaining]))]
            clique_neuron_ids.append(component_neurons[largest])
            largest_set = set(largest)
            remaining = [c for c in remaining if largest_set.isdisjoint(c)]
    return clique_neuron_ids


def build_raster(neuron_groups, spike_index, t_start, t_end):
    """Raster of several groups of neurons.

    Every neuron gets its own row (numbered consecutively across groups, also
    for neurons without spikes). Only spikes with t_start <= t < t_end are kept.
    Returns raster (2, n_spikes) = [spike times, rows] and the group index of every spike.
    """
    times, rows, labels = [np.array([])], [np.array([])], [np.array([])]
    row = 0
    for group_index, neuron_ids in enumerate(neuron_groups):
        for neuron_id in np.atleast_1d(neuron_ids):
            spikes = spike_index.spikes_of(neuron_id)
            spikes = spikes[(spikes >= t_start) & (spikes < t_end)]
            times.append(spikes)
            rows.append(np.full(spikes.shape[0], row, dtype=float))
            labels.append(np.full(spikes.shape[0], group_index, dtype=float))
            row += 1
    return np.vstack((np.concatenate(times), np.concatenate(rows))), np.concatenate(labels)


def compute_isi(spike_index, neuron_ids):
    """Inter-spike intervals (whole simulation) of all the neurons in neuron_ids."""
    return np.concatenate([np.array([])] + [np.diff(spike_index.spikes_of(n)) for n in np.atleast_1d(neuron_ids)])


# ---------------------------------------------------------------- figures

def save_png(file_path, title, xlabel, ylabel, legend=False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_hamming_distance_histogram(hamming_distances, correlation_threshold, results_sub_path):
    distance_counts = hamming_distance_counts(hamming_distances)
    plt.figure()
    plt.bar(np.arange(distance_counts.shape[0]), distance_counts, width=1.0)
    plt.axvline(correlation_threshold, color="red", linestyle="--", label="correlation threshold")
    save_png(os.path.join(results_sub_path, "hamming_distance_histogram.png"),
             "Hamming distances between active neurons",
             "Hamming distance (number of differing bins)", "number of neuron pairs", legend=True)


def plot_clique_raster(clique_spikes, clique_spike_labels, selected_clique_indices, results_sub_path):
    """Raster of the selected cliques, one legend entry (and color) per clique."""
    selected = np.isin(clique_spike_labels, selected_clique_indices)
    # consecutive y coordinate for every neuron with spikes, cliques in ascending order
    _, y_raster = np.unique(clique_spikes[1, selected], return_inverse=True)
    spike_times = clique_spikes[0, selected]
    spike_labels = clique_spike_labels[selected]

    fig = go.Figure()
    for k, clique_index in enumerate(selected_clique_indices):
        in_clique = spike_labels == clique_index
        fig.add_scatter(x=spike_times[in_clique], y=y_raster[in_clique], mode="markers",
                        name="clique_" + str(clique_index),
                        marker=dict(symbol=142, size=2, color=clique_color(k)))
    fig.update_layout(title="Raster of the clique neurons",
                      xaxis_title="time (ms)", yaxis_title="neuron (grouped by clique)",
                      paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT)
    fig.write_html(os.path.join(results_sub_path, "clique_raster.html"))


def plot_clique_positions(config, clique_neuron_ids, selected_clique_indices, results_sub_path):
    """3D positions of the clique neurons over a gray background of randomly chosen network neurons.

    The background has N_BACKGROUND_NEURONS neurons drawn without replacement from all the
    populations (all the neurons if the network is smaller). The seed is fixed, so the
    background is the same at every run.
    """
    all_neuron_positions = load_positions(config["positions_file"])
    n_background = min(N_BACKGROUND_NEURONS, all_neuron_positions.shape[0])
    background_rows = np.random.default_rng(0).choice(all_neuron_positions.shape[0], n_background, replace=False)
    background = all_neuron_positions[background_rows]

    fig = go.Figure(go.Scatter3d(x=background[:, 1], y=background[:, 2], z=background[:, 3],
                                 name="network (%d random neurons)" % n_background, mode="markers",
                                 marker=dict(size=1, color="gray", opacity=0.3)))
    for k, clique_index in enumerate(selected_clique_indices):
        in_clique = np.isin(all_neuron_positions[:, 0], clique_neuron_ids[clique_index])
        fig.add_scatter3d(x=all_neuron_positions[in_clique, 1], y=all_neuron_positions[in_clique, 2],
                          z=all_neuron_positions[in_clique, 3], mode="markers",
                          name="clique_" + str(clique_index),
                          text=all_neuron_positions[in_clique, 0].astype(int).astype(str),
                          marker=dict(size=5, color=clique_color(k)))

    axis = dict(backgroundcolor=TRANSPARENT)
    fig.update_layout(
        title="Positions of the clique neurons",
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(eye=dict(x=2, y=0, z=0)),
        scene=dict(xaxis=dict(title="x", **axis), yaxis=dict(title="y", **axis), zaxis=dict(title="z", **axis)),
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
    )
    fig.write_html(os.path.join(results_sub_path, "clique_neuron_positions_3d.html"))


def plot_isi_distribution(clique_isis, selected_clique_indices, results_sub_path):
    """Normalized distribution of all the ISI of each clique, 1 ms bins [n, n+1).

    The range is common to all cliques: from 0 to the largest ISI among all the selected cliques.
    """
    max_isi = max((isi.max() for isi in clique_isis if isi.size), default=0)
    bin_edges = np.arange(0, np.floor(max_isi) + 2)  # last bin [floor(max), floor(max)+1) contains max_isi
    fig = go.Figure()
    for k, isi in enumerate(clique_isis):
        hist_value, _ = np.histogram(isi, bin_edges)
        fig.add_scatter(x=bin_edges[:-1] + 0.5, y=hist_value / max(hist_value.sum(), 1),
                        name="clique_" + str(selected_clique_indices[k]), marker=dict(size=2, color=clique_color(k)))
    fig.update_layout(title="ISI distribution of the cliques (all ISI, 1 ms bins)",
                      xaxis_title="ISI (ms)", yaxis_title="fraction of ISI")
    fig.write_html(os.path.join(results_sub_path, "isi_distribution.html"))


# ---------------------------------------------------------------- main

def main():
    config = load_config()
    t_start, t_end, bin_size = config["start_time_ms"], config["end_time_ms"], config["bin_size_ms"]
    n_bins = int((t_end - t_start) / bin_size)
    activity_threshold = config["min_active_bins_fraction"] * n_bins
    correlation_threshold = config["max_different_bins_fraction"] * n_bins
    results_path, results_sub_path = results_folders(config)

    spike_list = load_spikes(config["spikes_file"])
    spike_index = SpikeIndex(spike_list)

    # 1-2. binarized activity and active neurons (uniformly subsampled if there are too many)
    is_spiking = compute_spiking_matrix(spike_list, t_start, bin_size, n_bins)
    active_neurons = np.where(is_spiking.sum(1) > activity_threshold)[0]
    if active_neurons.shape[0] > config["max_active_neurons"]:
        active_neurons = active_neurons[::int(active_neurons.shape[0] / config["max_active_neurons"]) + 1]
    print("number of active neurons =", active_neurons.shape[0])
    if active_neurons.shape[0] == 0:
        print("no active neuron: no neuron fires in more than %s of the %d bins of [%s, %s] ms. "
              "Check the time window and min_active_bins_fraction." % (
                  config["min_active_bins_fraction"], n_bins, t_start, t_end))
        return

    # 3. Hamming distances (cached: the most expensive step)
    distances_file = os.path.join(results_path, "hamming_distances_active_%s_n_%s.pkl"
                                  % (config["min_active_bins_fraction"], active_neurons.shape[0]))
    if os.path.exists(distances_file):
        with open(distances_file, "rb") as f:
            active_neurons, hamming_distances = pickle.load(f)
        print("hamming distances loaded")
    else:
        hamming_distances = compute_hamming_distances(is_spiking[active_neurons])
        with open(distances_file, "wb") as f:
            pickle.dump([active_neurons, hamming_distances], f)
    plot_hamming_distance_histogram(hamming_distances, correlation_threshold, results_sub_path)

    # 4-5. correlation graph, connected components and maximal cliques (cached)
    components_file = os.path.join(results_sub_path, "correlation_graph_components.pkl")
    if os.path.exists(components_file):
        with open(components_file, "rb") as f:
            correlation_graph, connected_component_ids, cliques_per_component, component_labels = pickle.load(f)
        print("connected components loaded")
    else:
        correlation_graph = build_correlation_graph(hamming_distances, correlation_threshold)
        connected_component_ids, cliques_per_component, component_labels = find_cliques_per_component(
            correlation_graph, config["min_connected_component_size"])
        with open(components_file, "wb") as f:
            pickle.dump([correlation_graph, connected_component_ids, cliques_per_component, component_labels], f)
    del hamming_distances

    # 6. disjoint cliques, raster and ISI
    clique_neuron_ids = extract_disjoint_cliques(
        active_neurons, component_labels, connected_component_ids, cliques_per_component)
    clique_spikes, clique_spike_labels = build_raster(clique_neuron_ids, spike_index, t_start, t_end)
    selected_clique_indices = [i for i, ids in enumerate(clique_neuron_ids) if len(ids) > config["min_clique_size"]]
    clique_isis = [compute_isi(spike_index, clique_neuron_ids[i]) for i in selected_clique_indices]

    with open(os.path.join(results_sub_path, "clique_info.pkl"), "wb") as f:
        pickle.dump([clique_neuron_ids, selected_clique_indices, clique_spikes, clique_spike_labels, clique_isis], f)
    pd.DataFrame(np.transpose(np.vstack((clique_spikes, clique_spike_labels))),
                 columns=["spike_time_ms", "neuron_row", "clique_index"]).to_csv(
        os.path.join(results_sub_path, "clique_raster_spikes.csv"), index=False)

    print(len(clique_neuron_ids), "cliques found,", len(selected_clique_indices),
          "with more than", config["min_clique_size"], "neurons")
    if not selected_clique_indices:
        print("no clique larger than min_clique_size: nothing to plot")
        return

    plot_clique_raster(clique_spikes, clique_spike_labels, selected_clique_indices, results_sub_path)
    plot_clique_positions(config, clique_neuron_ids, selected_clique_indices, results_sub_path)
    plot_isi_distribution(clique_isis, selected_clique_indices, results_sub_path)


if __name__ == "__main__":
    main()
