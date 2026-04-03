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
) -> RetinalRecording:
    """
    Load an Intan .rhs file into a RetinalRecording.

    Parameters
    ----------
    file_path : str
        Path to the .rhs file.

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
            "No stimulation data detected in the .rhs file. "
            "stim_indices, stim_data, stim_current, and stim_pulse_duration "
            "will be None.",
            UserWarning, stacklevel=2,
        )

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