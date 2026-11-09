"""
Shared simulation code for the pgam_jax example notebooks.

The example notebooks import from this module.

1. ``01_fit_one_smooth`` fits a PGAM to the tuning-curve dataset.
2. ``02_fit_position_and_event_smooths`` fits the spatial, nuisance, and
   event model.
3. ``03_circular_variables`` fits head direction cells with a cyclic basis.

Every dataset has its own seed. A notebook can therefore run on its own and
produce the same numbers on every run.
"""

from dataclasses import dataclass

import numpy as np
import scipy.stats as sts

SEED_TUNING = 20260925
SEED_TEMPORAL = 20260926
# The jump of a non-cyclic fit at the wrap point varies a lot from seed to
# seed. With this seed, its size is close to the median over seeds.
SEED_HEAD_DIRECTION = 20260930

TIME_BIN = 0.01

# Length of the event-response window, in time bins. 120 bins is 1200 ms. The
# window must be long enough for the true response to decay back to zero inside
# it. A shorter window cuts the response off and leaves a step at the edge.
EVENT_WINDOW_SIZE = 120


@dataclass
class PositionTunedData:
    """Counts from a neuron with a non-monotonic position tuning curve."""

    position: np.ndarray
    spikes: np.ndarray
    # Expected spike count at each sampled position.
    true_rate: np.ndarray
    position_grid: np.ndarray
    true_rate_grid: np.ndarray
    true_log_rate_grid: np.ndarray


@dataclass
class TemporalDataset:
    """One neuron driven by position, an event train, and nothing else."""

    position: np.ndarray
    nuisance: np.ndarray
    events: np.ndarray
    y: np.ndarray
    # Expected spike count per bin, exp(log_rate).
    true_rate: np.ndarray
    event_kernel: np.ndarray
    time_bin: float


def true_tuning_log_rate(position):
    """Ground-truth log-rate of the teaching neuron at a given position."""
    central_field = 2.25 * np.exp(-0.5 * ((position - 0.57) / 0.13) ** 2)
    shoulder = 0.65 * np.exp(-0.5 * ((position - 0.20) / 0.075) ** 2)
    return -1.25 + central_field + shoulder


def make_position_tuned_data(n_samples=320, seed=SEED_TUNING):
    """
    Build counts from a neuron with a main field and a smaller shoulder.

    This is the dataset of the ``01_fit_one_smooth`` notebook. Pass another
    ``seed`` to draw a new sample from the same model, for example as test
    data.
    """
    rng = np.random.default_rng(seed)
    position = rng.uniform(0.0, 1.0, n_samples)
    true_rate = np.exp(true_tuning_log_rate(position))
    spikes = rng.poisson(true_rate)

    position_grid = np.linspace(0.0, 1.0, 500)
    true_log_rate_grid = true_tuning_log_rate(position_grid)
    return PositionTunedData(
        position=position,
        spikes=spikes,
        true_rate=true_rate,
        position_grid=position_grid,
        true_rate_grid=np.exp(true_log_rate_grid),
        true_log_rate_grid=true_log_rate_grid,
    )


def spatial_effect(position):
    """Ground-truth spatial contribution to the log-rate of the temporal neuron."""
    return 1.6 * np.exp(-0.5 * (position / 0.55) ** 2)


# Baseline log-rate of the temporal neuron, before any input contributes.
TEMPORAL_BASELINE_LOG_RATE = -2.7

# Peak height of the true event response, in log-rate units. A much larger
# response makes the approximate significance test numerically unstable,
# because its test statistic lands far out in the tail of the null distribution.
EVENT_PEAK_LOG_RATE = 0.70


def make_event_kernel():
    """
    Ground-truth response following one event, sampled at the time bin.

    The response peaks about 260 ms after the event. It then decays to less
    than 2 percent of its peak before the window ends. A much sharper peak sits
    below the resolution of an eight-function B-spline basis over this window.
    """
    kernel = sts.gamma.pdf(np.linspace(0.0, 9.0, EVENT_WINDOW_SIZE), a=3.0)
    kernel /= kernel.max()
    return EVENT_PEAK_LOG_RATE * kernel


def make_event_impulse():
    """
    Build a single event with enough padding to show the whole response.

    Return the impulse and the time of each sample in ms, measured from the
    event. The array holds three windows of EVENT_WINDOW_SIZE samples. The
    input is zero everywhere except at the event.

    1. Padding. The convolution looks back over one full window. For these
       rows, part of that window is before the start of the array, so the
       output is not defined.
    2. Baseline. The look-back window is complete and holds no event. These
       rows give the level of the smooth before the event.
    3. Response. The event is the first sample. The response starts one
       sample later, so these rows show all but its last sample.
    """
    impulse = np.zeros(3 * EVENT_WINDOW_SIZE)
    event_index = 2 * EVENT_WINDOW_SIZE
    impulse[event_index] = 1.0
    time_ms = (np.arange(impulse.size) - event_index) * TIME_BIN * 1000
    return impulse, time_ms


def make_temporal_dataset(
    n_time_points=12_000,
    n_events=180,
    spatial_nuisance_correlation=0.9,
    seed=SEED_TEMPORAL,
):
    """
    Build a neuron that responds to position and to events, but not to noise.

    The nuisance input has no effect on the firing rate. A correct model must
    flatten its smooth and report a large p-value for it. Pass another ``seed``
    to draw a new recording from the same model, for example as test data.
    """
    rng = np.random.default_rng(seed)

    r = spatial_nuisance_correlation
    x = rng.multivariate_normal(
        mean=np.array([0, 0]), cov=np.array([[1, r], [r, 1]]), size=n_time_points
    )
    xn = (
        4 * (x - np.min(x, axis=0, keepdims=True)) / np.ptp(x, axis=0, keepdims=True)
        - 2
    )

    position = xn[:, 0]
    nuisance = xn[:, 1]

    events = np.zeros(n_time_points)
    event_indices = rng.choice(
        np.arange(100, n_time_points - 100),
        size=n_events,
        replace=False,
    )
    events[event_indices] = 1.0

    event_kernel = make_event_kernel()
    event_effect = np.convolve(events, event_kernel, mode="full")[:n_time_points]
    log_rate = TEMPORAL_BASELINE_LOG_RATE + spatial_effect(position) + event_effect
    true_rate = np.exp(log_rate)
    y = rng.poisson(true_rate)

    return TemporalDataset(
        position=position,
        nuisance=nuisance,
        events=events,
        y=y,
        true_rate=true_rate,
        event_kernel=event_kernel,
        time_bin=TIME_BIN,
    )


# Expected spike count of every head direction cell far from its preferred
# direction, and the height of its peak above that baseline.
HEAD_DIRECTION_BASELINE = 0.05
HEAD_DIRECTION_PEAK = 0.6


@dataclass
class HeadDirectionCell:
    """
    Tuning parameters of one head direction cell.

    A larger concentration gives a narrower peak.
    """

    name: str
    preferred_direction_deg: float
    concentration: float


HEAD_DIRECTION_CELLS = (
    HeadDirectionCell("sharp", preferred_direction_deg=90.0, concentration=8.0),
    HeadDirectionCell("wide", preferred_direction_deg=160.0, concentration=1.5),
)


@dataclass
class HeadDirectionData:
    """Counts from head direction cells, one column per cell."""

    cells: tuple
    # Head direction in radians, in [-pi, pi).
    angle: np.ndarray
    # Spike counts, shape (n_samples, n_cells).
    spikes: np.ndarray
    # Expected spike count, shape (n_samples, n_cells).
    true_count: np.ndarray
    angle_grid: np.ndarray
    # Expected spike count on the grid, shape (n_grid, n_cells).
    true_count_grid: np.ndarray


def head_direction_rate(angle, cell):
    """
    Ground-truth expected spike count of a head direction cell.

    The peak has a von Mises shape and sits on top of a constant baseline. The
    baseline makes the log-rate differ from a pure cosine, so a GLM with only
    cos and sin features cannot fit it exactly.
    """
    preferred_direction = np.deg2rad(cell.preferred_direction_deg)
    bump = np.exp(cell.concentration * (np.cos(angle - preferred_direction) - 1))
    return HEAD_DIRECTION_BASELINE + HEAD_DIRECTION_PEAK * bump


def make_head_direction_data(
    n_samples=1000,
    cells=HEAD_DIRECTION_CELLS,
    seed=SEED_HEAD_DIRECTION,
):
    """
    Build counts from head direction cells, all recorded at the same angles.

    This is the dataset of the ``03_circular_variables`` notebook. Each sample
    has an angle drawn uniformly around the circle.
    """
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, n_samples)
    true_count = np.stack([head_direction_rate(angle, cell) for cell in cells], axis=1)
    spikes = rng.poisson(true_count)

    angle_grid = np.linspace(-np.pi, np.pi, 361)
    true_count_grid = np.stack(
        [head_direction_rate(angle_grid, cell) for cell in cells], axis=1
    )
    return HeadDirectionData(
        cells=cells,
        angle=angle,
        spikes=spikes,
        true_count=true_count,
        angle_grid=angle_grid,
        true_count_grid=true_count_grid,
    )
