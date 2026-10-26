# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trials with pynapple

# %% [markdown]
# Many experiments are organized into trials separated by gaps, and we often want to fit a model using only the data within those trials. When the model includes event history, this raises a question about how to handle trial boundaries. Simply extracting the trial data and joining them into one array makes the end of one trial and the start of the next appear to be consecutive samples. Convolution then carries event history from one trial into the next, ignoring the time that elapsed between them.
#
# To preserve those boundaries, we can use [`pynapple`](https://pynapple.org) objects, which both `nemos` and `pgam_jax` accept in place of arrays. Along with timestamps and values, a pynapple time series stores its `time_support`: the intervals over which observations are valid. Setting this support to the trial intervals allows convolution to run separately within each trial. If the inputs are also recorded during the gaps, another option is to convolve over the continuous recording while using only the trial responses to fit the model.
#
# In this notebook, we will start with raw arrays from a simulated recording and demonstrate what goes wrong when trials are concatenated before convolution. We will then use pynapple to implement both solutions and compare how much trial data each retains. For both fits, we will use the position and event model from [notebook 02](02_fit_position_and_event_smooths.ipynb), leaving out the nuisance input:
#
# $$\log \lambda_t = \beta_0 + f_{\mathrm{pos}}(x_t) + \sum_{\tau \ge 0} k(\tau)\, e_{t-\tau}$$
#
# $$y_t \sim \mathrm{Poisson}(\lambda_t)$$
#
# Here, the spike count $y_t$ in each time bin depends on position $x_t$ through the smooth $f_{\mathrm{pos}}$, and on past event counts $e_t$ through the response kernel $k$. The model gives the expected count per bin, $\lambda_t$, which we divide by the bin width to obtain a firing rate in spikes/s.
#
# We simulate events only within trials and leave gaps longer than the event response, so no response carries over into the next trial in the underlying recording. This makes any carryover introduced by concatenating the trials easy to distinguish from the true event history.

# %%
import jax
import nemos as nmo
import numpy as np
import pynapple as nap
from example_data import (
    EVENT_WINDOW_SIZE,
    TIME_BIN,
    make_event_impulse,
    make_trial_recording,
)
from example_plots import (
    hide_file_paths_in_warnings,
    plot_design_at_trial_border,
    plot_predicted_and_true_rate_in_trials,
    plot_recovered_smooths,
    plot_trials,
    set_plot_style,
)

from pgam_jax import GAM

jax.config.update("jax_enable_x64", True)
set_plot_style()
hide_file_paths_in_warnings()

# %% [markdown]
# ## The raw data
#
# The simulated recording lasts about 160 seconds and contains 20 trials. Trial durations range from 4 to 8 seconds, with a 2 second gap after each trial.
#
# The data arrive as plain arrays:
#
# * position values and their timestamps, sampled every 10 ms
# * event timestamps
# * spike timestamps
# * the start and end of each trial
#
# All timestamps are in seconds. Position and spikes are recorded throughout, including the gaps. Event recording also covers the whole recording, but no events occur in the gaps.

# %%
data = make_trial_recording()

for name in [
    "position_time",
    "position",
    "event_times",
    "spike_times",
    "trial_starts",
    "trial_ends",
]:
    label = f"Shape of data.{name}:"
    print(f"{label:<29}{getattr(data, name).shape}")

# %% [markdown]
# ## Convert the raw data into pynapple objects
#
# * The trials become a `nap.IntervalSet`: a set of time intervals, each with a start and an end.
# * Position becomes a `nap.Tsd`: a time series with one value at each timestamp.
# * Events and spikes become `nap.Ts` objects: timestamps with no values.

# %%
trials = nap.IntervalSet(start=data.trial_starts, end=data.trial_ends)
position = nap.Tsd(t=data.position_time, d=data.position)
events = nap.Ts(t=data.event_times)
spikes = nap.Ts(t=data.spike_times)

trials

# %% [markdown]
# The shaded bands show the first 3 trials. Position and spikes continue in the gaps, but the events happen only inside the trials.

# %%
fig, ax = plot_trials(trials, position, events, spikes)

# %% [markdown]
# ## Build the model inputs
#
# The model needs position, event counts, and spike counts on the same time bins. We bin all three series with a bin size of `TIME_BIN` seconds. Passing `ep=trials` limits the bins to the trial intervals.
#
# * `bin_average` averages the position samples in each bin.
# * `count` counts the events or spikes in each bin.
#
# Each result is a `nap.Tsd`, with timestamps at the bin centers and the trials as its `time_support`. We check that all three results have the same timestamps.

# %%
position_binned = position.bin_average(TIME_BIN, ep=trials)
event_counts = events.count(TIME_BIN, ep=trials)
spike_counts = spikes.count(TIME_BIN, ep=trials)

print(f"Time bins in the trials: {len(spike_counts)}")
print(
    "Same time bins: "
    f"{np.array_equal(position_binned.t, event_counts.t)}, "
    f"{np.array_equal(event_counts.t, spike_counts.t)}"
)

# %% [markdown]
# ## Build the basis
#
# The same bases as in notebook 02, without the nuisance input.

# %%
position_basis = nmo.basis.BSplineEval(
    11,
    bounds=(-2.0, 2.0),
    label="position",
)
event_basis = nmo.basis.BSplineConv(
    8,
    window_size=EVENT_WINDOW_SIZE,  # in bins
    label="event history",
)

full_basis = position_basis + event_basis

# %% [markdown]
# ## Compare convolution with and without trial boundaries
#
# We build the same design matrix from two versions of the binned data:
#
# * Plain arrays carry no trial information. Convolution treats the end of one trial and the start of the next as consecutive samples.
# * Pynapple objects preserve the trial intervals, so convolution runs separately within each trial.

# %%
# `.values` gives an array holding data from all trials concatenated
design_joined = full_basis.compute_features(position_binned.values, event_counts.values)
design_split = full_basis.compute_features(position_binned, event_counts)

# given pynapple objects, nemos bases return pynapple objects preserving time info
print(type(design_split))
print(f"Design matrix shape: {design_split.shape}")

# %% [markdown]
# The plot shows the design around one trial boundary, with time on the vertical axis. The hatched band marks the gap, where neither design has rows. An event near the end of the first trial makes the difference visible:
#
# * With arrays, its history carries into the next trial, even though its response ended during the gap in the actual recording.
# * With pynapple, the first `EVENT_WINDOW_SIZE` rows of the next trial have NaN event features, shown in orange. The position features remain defined, but the event features lack a full window of history within that trial.
#
# `GAM` offers two ways to handle these NaNs. The default, `nan_handling="zero"`, replaces missing feature values with zero and keeps the rows. With `nan_handling="drop"`, any row containing a NaN feature is excluded from the fit.

# %%
fig, ax = plot_design_at_trial_border(full_basis, design_joined, design_split)

# %% [markdown]
# ## Fit the GAM
#
# We can pass the pynapple inputs and spike counts directly to `fit`.
#
# We use `nan_handling="drop"` because zero filling would set the modeled contribution of events early in the trial to zero, even though those events already affect the recorded neuron's firing.
# Dropping rows with incomplete event history avoids this mismatch, but discards the first 1.2 seconds of every trial. Later, we show how most of these bins can be recovered.

# %%
gam = GAM(
    full_basis,
    use_scipy=True,  # can be faster on CPU
    nan_handling="drop",
)
gam.fit((position_binned, event_counts), spike_counts)

# %% [markdown]
# ## Compare the fitted smooths with the truth
#
# We evaluate the position smooth on a position grid and the event smooth using a single event impulse, as in notebook 02. The shaded bands are the default 95% confidence bands returned by `smooth_compute`.

# %%
value_grid = np.linspace(-2.0, 2.0, 400)
position_smooth = gam.smooth_compute(value_grid, "position")

# evaluate the temporal smooth in response to a single event
impulse, event_time_ms = make_event_impulse()
event_smooth = gam.smooth_compute(impulse, "event history")

fig, ax = plot_recovered_smooths(
    value_grid,
    position_smooth=position_smooth,
    nuisance_smooth=None,
    event_smooth=event_smooth,
    impulse=impulse,
    event_time_ms=event_time_ms,
)

# %% [markdown]
# ## Predict the firing rate within trials
#
# `gam.predict` accepts pynapple inputs and returns a `Tsd` with their timestamps and time support.
#
# Prediction preserves all input rows. With `nan_handling="drop"`, it returns NaN where the event features are undefined, leaving the first 1.2 seconds of each trial blank in the plot.
#
# We compare fitted and true firing rates for the first two training trials.

# %%
# Divide by the bin width to convert the count per bin to spikes/s.
predicted_rate = gam.predict((position_binned, event_counts)) / TIME_BIN
true_rate = nap.Tsd(t=data.position_time, d=data.true_rate)

fig, ax = plot_predicted_and_true_rate_in_trials(
    trials, events, true_rate, predicted_rate
)

# %% [markdown]
# ## Use continuous history while fitting only the trials
#
# So far, each trial has been treated as a separate observation interval. In this recording, however, the inputs are also known during the gaps. We can use that history to construct event features at the start of each trial.
#
# We bin the whole recording and convolve over that continuous interval. To keep the fit limited to trial responses, we set spike counts outside the trials to NaN. `GAM.fit` excludes those response rows while still using the input history from the gaps.
#
# Note that this approach requires knowing when events occurred between trials. If events were not recorded during the gaps, we cannot assume that none occurred.
#
# The choice between the two approaches depends on the available history. When inputs outside trials are unavailable, keep trials separate and exclude bins without a full history window. When continuous inputs are known, use them to build the history and mask the responses to select the bins used for fitting.

# %%
# One interval for the whole recording gives all inputs the same bins.
# The position samples are at the bin centers.
recording = nap.IntervalSet(start=0.0, end=data.position_time[-1] + TIME_BIN / 2)

position_rec = position.bin_average(TIME_BIN, ep=recording)
event_counts_rec = events.count(TIME_BIN, ep=recording)
spike_counts_rec = spikes.count(TIME_BIN, ep=recording)

# in_interval gives NaN for the bins outside the trials
in_trial_mask = ~np.isnan(trials.in_interval(spike_counts_rec))
# set spike counts outside trials to NaN
nan_masked_counts = np.where(in_trial_mask, spike_counts_rec.values, np.nan)
# create a nap.Tsd with this data
spike_counts_in_trials = nap.Tsd(
    t=spike_counts_rec.t,
    d=nan_masked_counts,
    time_support=spike_counts_rec.time_support,  # y must have the same time support as the inputs
)

# %% [markdown]
# We build the design explicitly to compare it with the earlier designs at the same trial boundary.
#
# The third panel includes rows in the gap. The earlier event's response ends there, leaving no contribution from that event at the next trial's start. New events early in the next trial have defined features, where the separate-trial design has NaNs.

# %%
design_rec = full_basis.compute_features(position_rec, event_counts_rec)

fig, ax = plot_design_at_trial_border(
    full_basis, design_joined, design_split, design_rec
)

# %% [markdown]
# Fit using the continuous inputs and the masked spike counts. History comes from the whole recording, but because `gam.fit` drops rows where `spike_counts_in_trials` is NaN, only responses inside trials contribute to the fit.

# %%
gam_rec = GAM(
    full_basis,
    use_scipy=True,
    nan_handling="drop",
)
gam_rec.fit((position_rec, event_counts_rec), spike_counts_in_trials)

# %% [markdown]
# We predict over the continuous recording before restricting the result to trials, so the convolution still uses the history in the gaps.

# %%
# Predict with continuous history, then select the trial intervals.
predicted_rate_full = gam_rec.predict((position_rec, event_counts_rec)) / TIME_BIN
predicted_rate_rec = predicted_rate_full.restrict(trials)

# %%
fig, ax = plot_predicted_and_true_rate_in_trials(
    trials, events, true_rate, predicted_rate_rec
)

# %% [markdown]
# Here, only the first trial still loses its first 1.2 seconds, because no recorded history precedes it.
#
# In this data every bin in every trial has a valid spike count, so the number of finite predictions within trials also gives the number of bins used for fitting.

# %%
rows_split = np.isfinite(predicted_rate.values).sum()
rows_rec = np.isfinite(predicted_rate_rec.values).sum()
print(f"Rows in the fit, trials convolved separately: {rows_split}")
print(f"Rows in the fit, whole recording convolved:   {rows_rec}")

# %% [markdown]
# Convolving each trial separately retained 9,500 bins. Using continuous history increases this to 11,780 bins, recovering 2,280 bins, or 22.8 seconds of data within trials.

# %% [markdown]
# ### Compare the smooths after retaining more trial data
#
# We use the same grids and plotting conventions to inspect the second fit.

# %%
fig, ax = plot_recovered_smooths(
    value_grid,
    position_smooth=gam_rec.smooth_compute(value_grid, "position"),
    nuisance_smooth=None,
    event_smooth=gam_rec.smooth_compute(impulse, "event history"),
    impulse=impulse,
    event_time_ms=event_time_ms,
)
