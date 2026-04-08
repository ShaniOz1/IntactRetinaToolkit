"""
IntactRetinaToolkit.dataobj.rhs_loader
========================================
Loads a single Intan .rhs file into a RetinalRecording.
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pyintan.pyintan as pyintan

from dataobj.recording import RetinalRecording
from dataobj.channel_utils import intan_get_locations, _RHS_INDEX_TO_REAL_CH


def load_rhs(
    file_path: str,
    stim_threshold: float | None = None,
) -> RetinalRecording:
    """
    Load an Intan .rhs file into a RetinalRecording.

    Parameters
    ----------
    file_path : str
        Path to the .rhs file.
    stim_threshold : float | None
        If provided and no stim signal is found in the file, this value is
        used directly as the detection threshold (skipping the interactive
        figure).  Peaks in max(|recording_data|) that exceed
        abs(stim_threshold) and are at least 10 ms apart are set as
        stim_indices.

    Returns
    -------
    RetinalRecording

    Raises
    ------
    FileNotFoundError
    ValueError
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith('.rhs'):
        raise ValueError(f"Expected a .rhs file, got: {file_path}")

    file = pyintan.File(file_path)

    sample_rate    = int(file.sample_rate)
    file_name      = file.fname
    start_time     = file.datetime
    recording_data = file.analog_signals[0].signal          # (n_ch, n_samples)
    raw_channel_names = list(file.analog_signals[0].channel_names[0])
    channel_names = [str(_RHS_INDEX_TO_REAL_CH[i]) for i in range(len(raw_channel_names))]
    channel_locations = intan_get_locations(channel_names)

    # --- Stimulation ---
    stim_indices = None
    stim_data = None
    stim_current = None
    stim_channel_name = None

    try:
        stim_obj          = file.stimulation[0]
        stim_data         = stim_obj.signal
        stim_ch_idx       = stim_obj.channels
        stim_channel_name = (channel_names[stim_ch_idx]
                             if stim_ch_idx < len(channel_names) else None)
        stim_current          = float(stim_obj.current_levels.max())
        stim_indices          = _detect_stim_indices(stim_data)
    except Exception:
        warnings.warn(
            "No stimulation data detected in the .rhs file — "
            "launching interactive threshold picker.",
            UserWarning, stacklevel=2,
        )

    if stim_indices is None:
        if stim_threshold is not None:
            threshold = stim_threshold
            print(f"[load_rhs] Using provided stim_threshold = {threshold:.4f}")
        else:
            threshold = _interactive_stim_threshold(recording_data, channel_names, sample_rate)

        if threshold is not None:
            stim_indices = _detect_stim_from_threshold(recording_data, threshold, sample_rate)
            print(f"[load_rhs] Threshold = {threshold:.4f}  →  "
                  f"{len(stim_indices)} stim pulses detected.")
        else:
            print("[load_rhs] No threshold selected — stim_indices remains None.")

    return RetinalRecording(
        source='rhs',
        file_path=file_path,
        file_name=file_name,
        sample_rate=sample_rate,
        recording_data=recording_data,
        channel_names=channel_names,
        channel_locations=channel_locations,
        stim_indices=stim_indices,
        stim_data=stim_data,
        stim_current=stim_current,
        stim_channel_name=stim_channel_name,
        metadata={
            'start_time': start_time,
            'n_channels': recording_data.shape[0],
            'n_samples':  recording_data.shape[1],
        },
    )


# ── internal ────────────────────────────────────────────────────────────────

def _detect_stim_indices(stim_signal: np.ndarray) -> np.ndarray:
    """
    Detect biphasic stimulation pulse onsets from the Intan stim signal.

    Uses negative-going zero-crossings (threshold = -0.01), keeping every
    other crossing to account for the biphasic pulse shape.
    """
    crossings = np.where(np.diff(stim_signal < -0.01))[0] + 1
    return crossings[::2]


def _interactive_stim_threshold(
    recording_data: np.ndarray,
    channel_names: list[str],
    sample_rate: int,
) -> float | None:
    """
    Show all recording channels stacked in a compact figure and let the user
    click to set a threshold.

    The clicked y-value is marked as a horizontal dashed line (pastel red)
    across every subplot.  Clicking again updates the line.  Returns the
    last selected threshold when the figure is closed, or None if the user
    closed without clicking.
    """
    import matplotlib.pyplot as plt

    n_ch, n_samples = recording_data.shape
    times = np.arange(n_samples) / sample_rate

    # Y limits: true min/max across all channels with a 5 % margin
    global_min = float(recording_data.min())
    global_max = float(recording_data.max())
    margin = (global_max - global_min) * 0.05
    ylim = (global_min - margin, global_max + margin)

    fig_height = max(4.0, n_ch * 0.7 + 0.6)
    fig, axes = plt.subplots(
        n_ch, 1,
        figsize=(14, fig_height),
        sharex=True, sharey=True,
    )
    if n_ch == 1:
        axes = [axes]

    plt.subplots_adjust(hspace=0.02, left=0.08, right=0.98,
                        top=0.94, bottom=0.07)

    for i, ax in enumerate(axes):
        ax.plot(times, recording_data[i], color='black', linewidth=0.3)
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', labelsize=6)
        if i < n_ch - 1:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        else:
            ax.spines['bottom'].set_visible(True)
            ax.set_xlabel('Time (s)', fontsize=8)

    fig.suptitle(
        'Click to set threshold — click again to update — close when done',
        fontsize=8,
    )

    state: dict = {'threshold': None, 'hlines': []}

    def _on_click(event):
        if event.inaxes is None or event.ydata is None:
            return
        # Remove previous lines
        for line in state['hlines']:
            try:
                line.remove()
            except ValueError:
                pass
        state['hlines'].clear()
        # Draw new line on every subplot
        for ax in axes:
            line = ax.axhline(event.ydata, color='#f4a9a8', linewidth=1.2,
                              linestyle='--', zorder=4)
            state['hlines'].append(line)
        state['threshold'] = event.ydata
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.show(block=True)

    thr = state['threshold']
    if thr is not None:
        print(f"[load_rhs] Selected threshold: {thr:.4f}")
    return thr


def _detect_stim_from_threshold(
    recording_data: np.ndarray,
    threshold: float,
    sample_rate: int,
) -> np.ndarray:
    """
    Detect stim pulse indices from raw recording data using a user threshold.

    Takes the max absolute value across all channels at each sample, then
    finds peaks that exceed abs(threshold) and are at least 10 ms apart.
    """
    from scipy.signal import find_peaks

    combined     = np.max(np.abs(recording_data), axis=0)
    min_distance = int(0.010 * sample_rate)          # 10 ms minimal
    peaks, _     = find_peaks(combined,
                               height=abs(threshold),
                               distance=min_distance)
    return peaks