**Code authorship**: *Emiliano Spera*

**Reference scientific Poster**: *Spera, E., Tribuzzi, C., Marasco, A., & Migliore, M. (2024). Analysis of a full-scale hippocampal CA1 computational model network [Graphic]. Zenodo. https://doi.org/10.5281/zenodo.23527085*


# Clique analysis of correlated neurons

This folder contains a single self-contained script,
`find_correlated_neuron_cliques.py`. It finds groups of neurons of a spiking
neural network simulation whose firing activity is strongly correlated
(cliques of the correlation graph). It then characterises them with rasters,
ISI distributions and 3D positions.

## Contents

| File | Description |
|---|---|
| `find_correlated_neuron_cliques.py` | The whole analysis |
| `clique_configuration.json` | Parameters and **absolute** input/output paths. The paths are placeholders to replace |
| `test/` | Synthetic test network and activity (see [Test data](#test-data)) |

## Requirements

Python 3.9+ with:

```bash
pip install numpy scipy h5py "networkx>=2.7" matplotlib plotly pandas
```

The script needs networkx 2.7 or later, because it uses
`networkx.from_scipy_sparse_array`.

## Usage

1. In `clique_configuration.json`, replace the three placeholder paths
   (`my_absolute_path/...`, `results_absolute_path`) with **absolute** paths
   to your files (see [Paths](#paths)). If you leave them unchanged, the
   script stops with `Path '...' must be absolute`.
2. Set the analysis parameters (see [Analysis parameters](#analysis-parameters)).
3. Run:

```bash
python find_correlated_neuron_cliques.py
```

The script reads `clique_configuration.json` from its own folder, so you can
run it from any working directory (also from PyCharm). To use another
configuration file, pass its path as the first argument:

```bash
python find_correlated_neuron_cliques.py C:/path/to/my_configuration.json
```

Figures are only saved to file.

To check your installation first, run the script on the
[test data](#test-data).

## Input data

The script reads two HDF5 files:

* **Spikes** (`spikes_file`): one dataset named `spikes`, a 2-column array
  with one row per spike: `[neuron_id, spike_time_ms]`. Spike times are in
  **ms**. The row order does not matter.
* **Positions** (`positions_file`): one dataset per population (any name),
  each a 4-column array with one row per neuron: `[neuron_id, x, y, z]`.

Neuron ids must be **non-negative integers**, because they are used directly
as row indices. The ids in the positions file must be the same as in the
spikes file. Ids do not need to be contiguous, but the binary activity matrix
has `max id + 1` rows, so very large ids waste memory.

## Configuration (`clique_configuration.json`)

### Paths

Every path must be **absolute**. The script stops with an error if one is
relative. Forward slashes work on Windows too.

| Key | Description |
|---|---|
| `spikes_file` | Spikes file (see [Input data](#input-data)) |
| `positions_file` | Positions file (see [Input data](#input-data)) |
| `results_dir` | Folder for the results. It is created if missing |

> **Use a different `results_dir` for every simulation.** Cached results are
> looked up by a name that depends only on the time window, bin size and
> thresholds, not on the input files. If you analyse a different spikes file
> with the same `results_dir` and parameters, the script reuses the cached
> distances and cliques of the previous simulation, without any warning.

Example (Windows):

```json
{
	"spikes_file": "C:/data/sim_01/activity_network.hdf5",
	"positions_file": "C:/data/network/positions.hdf5",
	"results_dir": "C:/data/sim_01/clique_results",
	"start_time_ms": 1000,
	"end_time_ms": 9000,
	"bin_size_ms": 10,
	"min_active_bins_fraction": 0.1,
	"max_different_bins_fraction": 0.15,
	"max_active_neurons": 10000,
	"min_connected_component_size": 10,
	"min_clique_size": 5
}
```

### Analysis parameters

| Key | Description |
|---|---|
| `start_time_ms`, `end_time_ms` | Time window of the simulation to analyse (ms) |
| `bin_size_ms` | Bin size used to binarize the activity of each neuron (firing / not firing). `n_bins = (end − start) / bin_size` |
| `min_active_bins_fraction` | A neuron is *active* if it fires in more than `min_active_bins_fraction · n_bins` bins |
| `max_different_bins_fraction` | Two active neurons are *correlated* if their Hamming distance is below `max_different_bins_fraction · n_bins` |
| `max_active_neurons` | Maximum number of active neurons. If there are more, they are uniformly subsampled. Lower it if memory is not enough: the distance matrix has `n_active²` entries of 1–2 bytes |
| `min_connected_component_size` | Connected components with this many neurons or fewer are ignored |
| `min_clique_size` | Only cliques with more than this many neurons are analysed and plotted |

If no neuron is active in the window, the script prints a message and stops.
In that case, check the time window and `min_active_bins_fraction`.

## Method

1. **Binarization.** The window is split into `n_bins` bins of `bin_size_ms`.
   Bin `b` covers `start + b·bin_size ≤ t < start + (b+1)·bin_size`, so a spike
   falling exactly on an edge belongs to the bin that starts there.
   `is_spiking[n, b]` is `True` if neuron `n` fires at least once in bin `b`.
   The row index is the neuron id. The matrix has as many rows as the largest
   id among the neurons firing in the window, plus 1. Neurons with a larger id
   never fire in the window and could never be active.
2. **Active neurons.** These are the neurons that fire in more than
   `min_active_bins_fraction · n_bins` bins.
3. **Hamming distances.** For each pair of active neurons `(a, b)`, the
   distance is the number of bins where exactly one of the two fires. It is
   computed as `|a| + |b| − 2·a·b` with matrix products, in blocks of rows.
4. **Correlation graph.** Two neurons are connected if their distance is below
   `max_different_bins_fraction · n_bins`. The script computes the connected
   components of the graph. For each component larger than
   `min_connected_component_size`, it lists all maximal cliques with
   `networkx.find_cliques`.
5. **Disjoint cliques.** In each component the largest clique is selected.
   Every clique that shares a neuron with it is then discarded, and this repeats
   until no clique is left.
6. **Characterisation** of the cliques with more than `min_clique_size`
   neurons:
   * raster of the clique neurons in the analysis window
     (`start_time_ms ≤ t < end_time_ms`);
   * inter-spike intervals (ISI) of the clique neurons, computed over the
     **whole** simulation, not only the analysis window;
   * 3D positions of the clique neurons in the network.

## Output

```
<results_dir>/
└── clique_interval_<start>_<end>_bin_<bin_size>/                      = results_path
    ├── hamming_distances_active_<min_active>_n_<n_active>.pkl
    └── active_<min_active>_corr_<max_diff>/                            = results_sub_path
        └── (all the other files)
```

### Figures (in `results_sub_path`)

| File | Content | x axis | y axis |
|---|---|---|---|
| `hamming_distance_histogram.png` | Number of neuron pairs at each Hamming distance. The red dashed line is the correlation threshold | Hamming distance (number of differing bins) | number of neuron pairs |
| `clique_raster.html` | Raster of the selected cliques, one color / legend entry per clique (transparent background) | time (ms) | neuron (grouped by clique) |
| `clique_neuron_positions_3d.html` | 3D positions of the clique neurons. Hover shows the neuron id. The gray background shows 5000 neurons drawn at random from all the populations (all of them if the network has fewer). The seed is fixed, so the background is the same at every run. To change the number, edit `N_BACKGROUND_NEURONS` at the top of the script | x | y, z |
| `isi_distribution.html` | Distribution of all ISIs per clique, normalised to 1 (each curve sums to 1). It uses 1 ms bins, one per integer ISI value `[n, n+1)`. The x range, common to all cliques, goes from 0 to the largest ISI among all the selected cliques. ISIs are computed over the whole simulation | ISI (ms) | fraction of ISI |

### Data files

* `results_path/hamming_distances_active_<min_active>_n_<n_active>.pkl`: `[active_neurons, hamming_distances]`.
  This is a cache, reused by later runs with the same window, bin size and activity threshold.
  * `active_neurons`: ids of the active neurons.
  * `hamming_distances[i, j]`: distance between `active_neurons[i]` and `active_neurons[j]` (`uint8` or `uint16`).
* `correlation_graph_components.pkl`: `[correlation_graph, connected_component_ids, cliques_per_component, component_labels]`.
  This is a cache, reused by later runs with the same thresholds.
  * `correlation_graph`: sparse `scipy.sparse.csr_matrix` `(n_active, n_active)`. Entry `(i, j)` is `True` if `i != j` and the two neurons are correlated. Use `correlation_graph.nonzero()` to get the correlated pairs.
  * `connected_component_ids`: labels of the components larger than `min_connected_component_size`.
  * `cliques_per_component[c]`: maximal cliques of component `c`, as indices local to that component.
  * `component_labels[i]`: connected component of `active_neurons[i]`.
* `clique_info.pkl`: `[clique_neuron_ids, selected_clique_indices, clique_spikes, clique_spike_labels, clique_isis]`.
  It is rewritten at every run.
  * `clique_neuron_ids[i]`: neuron ids of the i-th disjoint clique (flat list over all components).
  * `selected_clique_indices`: indices `i` of the cliques with more than `min_clique_size` neurons.
  * `clique_spikes`: array `(2, n_spikes)`: spike time and raster row. The row is a consecutive index, not the neuron id.
  * `clique_spike_labels[s]`: clique index `i` of spike `s`.
  * `clique_isis[k]`: ISIs of the k-th selected clique.
* `clique_raster_spikes.csv`: the raster of all cliques, with columns `spike_time_ms, neuron_row, clique_index`.

To force a recomputation, delete the corresponding `.pkl` cache.

## Test data

The `test/` folder contains a small synthetic dataset, in the format described
in [Input data](#input-data). You can use it to check your installation, or to
try parameters quickly.

| File | Description |
|---|---|
| `positions.hdf5` | 10 000 neurons in 3 populations (`SP_PC`, `SP_PVBC`, `SO_OLM`), positions in µm |
| `activity_network.hdf5` | 10 s of activity (0.1 ms resolution, about 245 000 spikes) |
