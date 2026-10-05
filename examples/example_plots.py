"""
Colors, plot style, and plotting helpers for the pgam_jax notebooks.

The notebooks call these so that each cell shows model code and not matplotlib
code. Every function that draws opens its own figure and returns it.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from example_data import EVENT_WINDOW_SIZE, spatial_effect
from scipy.stats import binned_statistic, sem

BLUE = "#2C6EAA"
ORANGE = "#E07A3F"
GREEN = "#4B8B6B"
PURPLE = "#7A5AA6"
GRAY = "#606A73"
LIGHT_GRAY = "#D9DEE3"

# Background tint of the input panels in the recording figure. It groups the
# measured inputs and separates them from the spike counts below.
PANEL_FILL = "#F2F4F6"

# Fill color of the design-matrix entries that have no value. The first rows
# of a convolutional block are NaN, because the convolution has no past yet.
MISSING_FILL = "#F0C9A6"


def set_plot_style():
    """Apply the shared plotting style."""
    sns.set_theme(style="ticks", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 72,
            "savefig.dpi": 220,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _format_warning_without_file_path(message, category, filename, lineno, line=None):
    """Format a warning as its category and its message only."""
    return f"{category.__name__}: {message}\n"


def hide_file_paths_in_warnings():
    """
    Print warnings without the file path and the line number of their source.

    The default format starts each warning with the path of the file that
    raised it. That path is local to the machine that ran the notebook, so it
    does not belong in the saved outputs.
    """
    warnings.formatwarning = _format_warning_without_file_path


# ---------------------------------------------------------------------------
# 01_fit_one_smooth
# ---------------------------------------------------------------------------


def plot_counts_and_truth(data):
    """
    Draw the observed counts and the true tuning curve on a new figure.

    Return the figure and its axis. Pass the axis to ``add_rate_curve`` to add
    a fitted curve on top.
    """
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.scatter(
        data.position,
        data.spikes,
        s=22,
        alpha=0.30,
        color=GRAY,
        label="Observed counts",
    )
    ax.plot(
        data.position_grid,
        data.true_rate_grid,
        color="black",
        linewidth=3,
        linestyle="--",
        label="True tuning curve",
    )
    ax.set(
        xlabel="Position",
        ylabel="Expected count",
        xlim=(-0.02, 1.02),
        ylim=(-0.4, data.spikes.max() + 0.8),
    )
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    return fig, ax


def add_rate_curve(ax, x, rate, label, color, alpha=1.0):
    """Draw one fitted rate curve on an existing axis and refresh the legend."""
    ax.plot(x, rate, color=color, linewidth=4, alpha=alpha, label=label)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    return ax


def plot_design_recorded_and_sorted(design, position):
    """
    Draw the design matrix twice, in recorded order and sorted by position.

    Sorting the rows shows the band structure of a B-spline basis. Each row
    touches only the few basis functions that cover its position.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)
    sort_index = np.argsort(position)

    for ax, rows, title in [
        (axes[0], design, "Recorded order"),
        (axes[1], design[sort_index], "Sorted by position"),
    ]:
        image = ax.imshow(
            rows,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
        )
        ax.set(xlabel="Basis function", title=title)

    axes[0].set_ylabel("Observation")
    fig.colorbar(image, ax=axes, label="Feature value")
    return fig


# ---------------------------------------------------------------------------
# 02_fit_position_and_event_smooths
# ---------------------------------------------------------------------------


def plot_inputs_and_counts_over_time(data, n_bins=2000):
    """
    Draw the three inputs and the spike counts over the first ``n_bins``.

    The three input panels sit on a tinted background. The spike panel sits
    below a gap, on a white background. The split separates what the
    experiment measured from what the neuron did.
    """
    window = slice(0, n_bins)
    time_axis = np.arange(data.y.size)[window] * data.time_bin

    fig = plt.figure(figsize=(12.5, 6.8), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.2], hspace=0.10)
    input_grid = outer[0].subgridspec(3, 1, height_ratios=[1.0, 1.0, 0.5], hspace=0.06)

    # Share the time axis across all panels. The event panel draws with
    # eventplot, which otherwise sets its own limits from the event times.
    input_axes = [fig.add_subplot(input_grid[0])]
    input_axes += [
        fig.add_subplot(input_grid[row], sharex=input_axes[0]) for row in range(1, 3)
    ]
    spike_ax = fig.add_subplot(outer[1], sharex=input_axes[0])
    axes = input_axes + [spike_ax]

    input_axes[0].plot(time_axis, data.position[window], color=BLUE, linewidth=0.9)
    input_axes[0].set(ylabel="Position")

    input_axes[1].plot(time_axis, data.nuisance[window], color=GREEN, linewidth=0.9)
    input_axes[1].set(ylabel="Nuisance")

    event_times = time_axis[data.events[window] > 0]
    input_axes[2].eventplot(
        event_times,
        colors=PURPLE,
        lineoffsets=0.5,
        linelengths=0.8,
    )
    input_axes[2].set(ylabel="Events", yticks=[])

    spike_ax.plot(time_axis, data.y[window], color=GRAY, linewidth=0.9)
    spike_ax.set(ylabel="Spike count", xlabel="Time (s)")

    for ax in input_axes:
        ax.set_facecolor(PANEL_FILL)
        ax.tick_params(labelbottom=False)

    input_axes[0].set_title("Inputs: what the experiment measured", fontsize=14)
    spike_ax.set_title("Output: what the neuron did", fontsize=14)

    fig.align_ylabels(axes)
    return fig


def _mean_count_by_value(counts, values, n_value_bins):
    """
    Average the spike counts inside bins of one input value.

    Return the bin centers, the mean count per bin, and its standard error.
    """
    binning = dict(bins=n_value_bins, range=(-2.0, 2.0))
    mean_count, edges, _ = binned_statistic(values, counts, "mean", **binning)
    standard_error, _, _ = binned_statistic(values, counts, sem, **binning)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mean_count, standard_error


def _event_triggered_average(data, before_ms, after_ms, plot_bin_ms):
    """
    Cut the spike counts around each event and average the pieces.

    Return the bin centers in ms from the event, the mean count per bin, its
    standard error, and the number of events used. An event needs a full
    window, so an event too close to the end of the recording is left out.
    """
    data_bin_ms = data.time_bin * 1000
    n_before = int(round(before_ms / data_bin_ms))
    n_after = int(round(after_ms / data_bin_ms))
    data_bins_per_plot_bin = int(round(plot_bin_ms / data_bin_ms))

    offsets = np.arange(-n_before, n_after)
    n_plot_bins = offsets.size // data_bins_per_plot_bin
    offsets = offsets[: n_plot_bins * data_bins_per_plot_bin]

    event_index = np.flatnonzero(data.events)
    complete = (event_index + offsets[0] >= 0) & (
        event_index + offsets[-1] < data.y.size
    )
    event_index = event_index[complete]

    rows = data.y[event_index[:, None] + offsets[None, :]]
    rebinned = rows.reshape(-1, n_plot_bins, data_bins_per_plot_bin).mean(axis=2)
    centers = (
        offsets.reshape(n_plot_bins, data_bins_per_plot_bin).mean(axis=1) * data_bin_ms
    )
    mean_count = rebinned.mean(axis=0)
    standard_error = sem(rebinned, axis=0)
    return centers, mean_count, standard_error, event_index.size


def plot_binned_tuning(
    data,
    n_value_bins=24,
    before_ms=300.0,
    after_ms=1500.0,
    plot_bin_ms=50.0,
):
    """
    Draw binned tuning curves of the three inputs, computed with no model.

    The layout repeats ``plot_true_effects``. The top row holds the two inputs
    that act through their current value. Each panel shows the mean count per
    bin of input value. The bottom row holds the event input, which acts over
    time. It shows the event-triggered average: the mean count per bin of time
    since the event.

    Each input is binned on its own, so correlated inputs mix their effects.
    Position and nuisance are correlated, so the nuisance panel shows part of
    the position tuning, although the nuisance input has no effect. Events
    that fall close together share part of their window, so their responses
    add up in the bottom panel. The model separates both. This figure does not.
    """
    fig = plt.figure(figsize=(13.0, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    spatial_ax = fig.add_subplot(grid[0, 0])
    nuisance_ax = fig.add_subplot(grid[0, 1])
    event_ax = fig.add_subplot(grid[1, :])

    value_panels = [
        (spatial_ax, data.position, BLUE, "Position"),
        (nuisance_ax, data.nuisance, GREEN, "Nuisance input"),
    ]
    for ax, values, color, name in value_panels:
        centers, mean_count, standard_error = _mean_count_by_value(
            data.y, values, n_value_bins
        )
        ax.errorbar(
            centers,
            mean_count,
            yerr=standard_error,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2.0,
            capsize=3,
        )
        ax.set(xlabel=name, title=f"Mean count per {name.lower()} bin")

    spatial_ax.set_ylabel("Mean spike count")
    # Share the vertical scale, so the two panels compare directly. The
    # nuisance panel then shows how much of the position tuning leaks into it.
    shared_limits = (
        min(spatial_ax.get_ylim()[0], nuisance_ax.get_ylim()[0]),
        max(spatial_ax.get_ylim()[1], nuisance_ax.get_ylim()[1]),
    )
    spatial_ax.set_ylim(shared_limits)
    nuisance_ax.set_ylim(shared_limits)

    centers, mean_count, standard_error, n_events = _event_triggered_average(
        data, before_ms, after_ms, plot_bin_ms
    )
    event_ax.axhline(
        data.y.mean(),
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Mean over the whole recording",
    )
    event_ax.axvline(0.0, color=PURPLE, linewidth=1.4, linestyle=":")
    event_ax.errorbar(
        centers,
        mean_count,
        yerr=standard_error,
        color=PURPLE,
        marker="o",
        markersize=5,
        linewidth=2.0,
        capsize=3,
        label=f"Mean over {n_events} events",
    )
    event_ax.set(
        xlabel="Time from event (ms)",
        ylabel="Mean spike count",
        title=f"Event-triggered average, {plot_bin_ms:.0f} ms bins",
    )
    event_ax.legend(frameon=False, fontsize=12)

    return fig


def plot_design_blocks(basis, data, n_rows=400):
    """
    Draw the first ``n_rows`` of the design matrix, with one block per smooth.

    Each block of columns holds the features of one smooth. The first
    ``EVENT_WINDOW_SIZE`` rows of the event block are NaN, drawn in orange.
    A convolution needs a full window of past samples, so those rows have no
    value.
    """
    design = np.asarray(
        basis.compute_features(data.position, data.nuisance, data.events)
    )[:n_rows]

    color_map = plt.get_cmap("Blues").copy()
    color_map.set_bad(MISSING_FILL)

    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_invalid(design),
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
    )
    ax.set(xlabel="Feature", ylabel="Time bin", title="Design matrix")

    centers = []
    labels = []
    start = 0
    for component in basis:
        width = component.n_basis_funcs
        if start > 0:
            ax.axvline(start - 0.5, color="black", linewidth=1.8)
        centers.append(start + 0.5 * width - 0.5)
        labels.append(component.label)
        start += width

    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks(centers)
    top_axis.set_xticklabels(labels, fontsize=12)
    top_axis.tick_params(length=0)

    fig.colorbar(image, ax=ax, label="Feature value")
    return fig


def _find_isolated_events(data, n_events=3, length=600):
    """
    Find a stretch of the recording whose only events are ``n_events`` events.

    The stretch must also start with a full response window that holds no
    event, so that no earlier response leaks into it.
    """
    event_index = np.flatnonzero(data.events)
    for start in range(EVENT_WINDOW_SIZE, data.y.size - length):
        inside = (event_index >= start - EVENT_WINDOW_SIZE) & (
            event_index < start + length
        )
        chosen = event_index[inside]
        if chosen.size == n_events and chosen.min() >= start + 30:
            return start, chosen
    raise RuntimeError(f"No stretch of the recording holds exactly {n_events} events.")


def plot_true_effects(data, value_grid):
    """
    Draw the three true effects that built the recording.

    The top row holds the two effects that act through the current value of an
    input. The bottom row holds the event effect, which acts over time. It
    shows three events, one copy of the true response under each, and their
    sum. ``plot_binned_tuning`` repeats this layout with summaries of the data.
    """
    true_position = spatial_effect(value_grid)

    start, chosen_events = _find_isolated_events(data)
    # Sum only the chosen responses, not every event in the recording. Run the
    # time axis on until the last response has decayed.
    end = chosen_events.max() + EVENT_WINDOW_SIZE
    chosen_train = np.zeros(end - start)
    chosen_train[chosen_events - start] = 1.0
    event_sum = np.convolve(chosen_train, data.event_kernel, mode="full")[
        : chosen_train.size
    ]
    demo_time = np.arange(start, end) * data.time_bin
    kernel_time = np.arange(EVENT_WINDOW_SIZE) * data.time_bin

    fig = plt.figure(figsize=(13.0, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    spatial_ax = fig.add_subplot(grid[0, 0])
    nuisance_ax = fig.add_subplot(grid[0, 1])
    event_ax = fig.add_subplot(grid[1, :])

    spatial_ax.plot(value_grid, true_position, color=BLUE, linewidth=2.7)
    spatial_ax.set(
        xlabel="Position",
        ylabel="Contribution to log-rate",
        title="True spatial effect",
    )

    nuisance_ax.axhline(0.0, color=GREEN, linewidth=2.7)
    # axhline carries no x data, so set both limits from the spatial panel. The
    # flat line then reads as flat, not as a line with its own small scale.
    nuisance_ax.set(xlabel="Nuisance input", title="True nuisance effect: none")
    nuisance_ax.set_xlim(value_grid.min(), value_grid.max())
    nuisance_ax.set_ylim(spatial_ax.get_ylim())

    for time_of_event in chosen_events * data.time_bin:
        event_ax.axvline(time_of_event, color=PURPLE, linewidth=1.4, linestyle=":")
        event_ax.plot(
            time_of_event + kernel_time,
            data.event_kernel,
            color=LIGHT_GRAY,
            linewidth=6.5,
            label="One event response",
        )
    event_ax.plot(
        demo_time,
        event_sum,
        color=PURPLE,
        linewidth=2.4,
        label="Sum of the three",
    )
    event_ax.set(
        xlabel="Time (s)",
        ylabel="Contribution to log-rate",
        title="True event response over time",
    )
    # The loop adds one labeled handle per event. Keep the last two only.
    handles, labels = event_ax.get_legend_handles_labels()
    event_ax.legend(handles[-2:], labels[-2:], frameon=False, fontsize=12)

    return fig


def plot_recovered_smooths(
    data,
    value_grid,
    position_smooth,
    nuisance_smooth,
    event_smooth,
    impulse,
    event_time_ms,
):
    """
    Draw each fitted smooth against its truth.

    Each ``*_smooth`` argument is the ``(mean, low, high)`` triple that
    ``GAM.smooth_compute`` returns.
    """
    position_mean, position_low, position_high = position_smooth
    nuisance_mean, nuisance_low, nuisance_high = nuisance_smooth
    event_mean, event_low, event_high = (np.asarray(part) for part in event_smooth)

    true_position = spatial_effect(value_grid)
    true_position -= np.mean(true_position)

    # A convolution needs a full window of past samples. The first
    # EVENT_WINDOW_SIZE rows have no defined value, so drop them.
    settled = np.arange(event_mean.size) >= EVENT_WINDOW_SIZE

    # smooth_compute centers the smooth on its input, here the impulse. Before
    # the event the window holds no event, so every basis feature is zero and
    # the centered curve sits at a constant offset. Subtract the pre-event mean
    # to remove that offset. The fit then shows the uncentered response, which
    # is zero before the event, like the truth. Use only settled rows here.
    # The unsettled rows would drag the baseline toward zero.
    baseline_rows = settled & (event_time_ms < 0)
    true_event = np.convolve(impulse, data.event_kernel, mode="full")[: impulse.size]
    true_event = true_event - true_event[baseline_rows].mean()
    event_baseline = event_mean[baseline_rows].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)

    axes[0].plot(
        value_grid,
        true_position,
        color="black",
        linestyle="--",
        linewidth=2.4,
        label="Truth",
    )
    axes[0].plot(value_grid, position_mean, color=BLUE, linewidth=2.7, label="PGAM")
    axes[0].fill_between(value_grid, position_low, position_high, color=BLUE, alpha=0.2)
    axes[0].set(
        xlabel="Position",
        ylabel="Contribution to log-rate",
        title="Spatial smooth",
    )

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=2.0, label="Truth")
    axes[1].plot(value_grid, nuisance_mean, color=GREEN, linewidth=2.7, label="PGAM")
    axes[1].fill_between(
        value_grid, nuisance_low, nuisance_high, color=GREEN, alpha=0.2
    )
    axes[1].set(
        xlabel="Nuisance input",
        title="Nuisance smooth",
    )
    # Share the vertical scale with the spatial panel. On its own scale the flat
    # smooth looks uncertain, but its band is small next to a real effect.
    axes[1].set_ylim(axes[0].get_ylim())

    valid = settled & np.isfinite(event_mean)
    axes[2].plot(
        event_time_ms,
        true_event,
        color="black",
        linestyle="--",
        linewidth=2.4,
        label="Truth",
    )
    axes[2].plot(
        event_time_ms[valid],
        event_mean[valid] - event_baseline,
        color=PURPLE,
        linewidth=2.7,
        label="PGAM",
    )
    axes[2].fill_between(
        event_time_ms[valid],
        event_low[valid] - event_baseline,
        event_high[valid] - event_baseline,
        color=PURPLE,
        alpha=0.2,
    )
    axes[2].set(
        xlabel="Time from event (ms)",
        ylabel="Change in log-rate",
        title="Temporal smooth",
        xlim=(-150, EVENT_WINDOW_SIZE * data.time_bin * 1000 + 150),
    )

    for ax in axes:
        ax.legend(frameon=False, fontsize=12)

    return fig


def plot_predicted_and_true_rate(data, predicted_rate, n_bins=300):
    """
    Draw the predicted and the true firing rate over the first ``n_bins``.

    ``predicted_rate`` is in spikes per second, with one value per time bin of
    the recording. Dashed vertical lines mark the events.
    """
    window = slice(0, n_bins)
    time_axis = np.arange(data.y.size)[window] * data.time_bin
    event_times = time_axis[data.events[window] > 0]
    # data.true_rate is the expected count per bin. Convert it to spikes/s.
    true_rate = data.true_rate / data.time_bin

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.vlines(
        event_times,
        0,
        1,
        transform=ax.get_xaxis_transform(),
        color=GREEN,
        linestyle="--",
        label="Event",
    )
    ax.plot(
        time_axis,
        true_rate[window],
        color=GRAY,
        linewidth=3,
        alpha=0.6,
        label="True rate",
    )
    ax.plot(
        time_axis,
        np.asarray(predicted_rate)[window],
        color=ORANGE,
        linewidth=1.2,
        label="PGAM prediction",
    )
    ax.set(xlabel="Time (s)", ylabel="Rate (spikes/s)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    return fig
