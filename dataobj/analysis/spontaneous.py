"""
IntactRetinaToolkit.dataobj.analysis.spontaneous
=================================================
Spontaneous activity wave detection.

Detects sustained negative deflections (waves) in each channel using a
noise-based threshold. A wave is defined as a contiguous region where the
signal stays below the threshold for at least min_duration_ms.

Called via rec.detect_spontaneous().
Results stored in rec.spontaneous as a DataFrame.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_spontaneous(
    rec,
    min_duration_ms: float = 5.0,
    threshold_std: float = 4.0,
) -> pd.DataFrame:
    """
    Detect spontaneous waves and return a DataFrame.

    Parameters
    ----------
    rec : RetinalRecording
    min_duration_ms : float
        Minimum duration of a wave to be counted, in ms. Default: 5.
    threshold_std : float
        Threshold = threshold_std × per-channel std.
        Signal must go below -threshold to count as a wave. Default: 4.

    Returns
    -------
    pd.DataFrame
        Columns: channel, start_index, stop_index.
    """
    data        = _get_data(rec)
    min_samples = int(min_duration_ms / 1000 * rec.sample_rate)

    # If stimulation data is present, mask it out before estimating noise
    # so stim artefacts don't inflate the threshold
    quiet_mask = _build_quiet_mask(rec, data.shape[1])

    rows = []
    for ch_idx, ch_name in enumerate(rec.channel_names):
        ch_data   = data[ch_idx, :]
        noise_std = float(np.std(ch_data[quiet_mask]))
        threshold = -threshold_std * noise_std   # negative threshold

        waves = _find_waves(ch_data, threshold, min_samples)
        for start, stop in waves:
            rows.append({
                'channel':     ch_name,
                'start_index': int(start),
                'stop_index':  int(stop),
            })

    return pd.DataFrame(rows) if rows else _empty_spontaneous_df()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_data(rec) -> np.ndarray:
    """Return the best available data array (filtered > blanked > raw)."""
    if rec.filtered_data is not None:
        return rec.filtered_data
    if rec.blanked_data is not None:
        return rec.blanked_data
    return rec.recording_data


def _build_quiet_mask(rec, n_samples: int) -> np.ndarray:
    """
    Boolean mask: True = away from stim pulses (for noise estimation).
    Falls back to all-True if no stim_indices are available.
    """
    mask = np.ones(n_samples, dtype=bool)
    if rec.stim_indices is None or len(rec.stim_indices) == 0:
        return mask

    pre    = int(0.005  * rec.sample_rate)   # 5 ms before
    post   = int(0.015  * rec.sample_rate)   # 15 ms after
    for idx in rec.stim_indices:
        start = int(max(0, idx - pre))
        end   = int(min(n_samples, idx + post + 1))
        mask[start:end] = False
    return mask


def _find_waves(
    ch_data: np.ndarray,
    threshold: float,
    min_samples: int,
) -> list[tuple[int, int]]:
    """
    Find contiguous regions where ch_data < threshold for at least
    min_samples samples.

    Returns list of (start_index, stop_index) tuples.
    """
    below = ch_data < threshold
    waves = []
    in_wave = False
    start   = 0

    for i, b in enumerate(below):
        if b and not in_wave:
            in_wave = True
            start   = i
        elif not b and in_wave:
            in_wave = False
            if (i - start) >= min_samples:
                waves.append((start, i - 1))

    # Handle wave that runs to end of array
    if in_wave and (len(ch_data) - start) >= min_samples:
        waves.append((start, len(ch_data) - 1))

    return waves


def _empty_spontaneous_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['channel', 'start_index', 'stop_index'])