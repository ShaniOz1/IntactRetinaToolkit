"""
IntactRetinaToolkit.dataobj.analysis.indirect
==============================================
Indirect (network-driven) response detection.

Blanks windows around each stim pulse, estimates per-channel noise,
then detects negative-going spikes above a noise threshold.

Called via rec.detect_indirect_response().
Results stored in rec.indirect_response as a DataFrame.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import scipy.signal


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_indirect_response(
    rec,
    blanking_ms: float = 15.0,
    threshold_std: float = 4.0,
) -> pd.DataFrame:
    """
    Detect indirect responses and return a DataFrame.

    Parameters
    ----------
    rec : RetinalRecording
    blanking_ms : float
        Window around each stim pulse excluded from noise estimation
        and detection, in ms. Default: 15.
    threshold_std : float
        Threshold = threshold_std × per-channel std of quiet segments.
        Default: 4.

    Returns
    -------
    pd.DataFrame
        Columns: channel, spike_index, amplitude (µV), latency (ms),
        width (ms).
    """
    if rec.stim_indices is None or len(rec.stim_indices) == 0:
        warnings.warn("[indirect] stim_indices is None — returning empty DataFrame.",
                      UserWarning)
        return _empty_indirect_df()

    data          = _get_data(rec)
    blank_samples = int(blanking_ms / 1000 * rec.sample_rate)
    min_distance  = int(rec.sample_rate / 1000)   # 1 ms minimum between spikes

    # Build a mask that excludes windows around stim pulses
    mask = _build_quiet_mask(rec.stim_indices, data.shape[1],
                             blank_samples, rec.sample_rate)

    # Blank the data for detection (keep quiet segments only for noise est.)
    blanked = data.copy()
    blanked[:, ~mask] = 0.0

    stim_ch_idx = (rec.get_channel_index(rec.stim_channel_name)
                   if rec.stim_channel_name else None)

    rows = []
    for ch_idx, ch_name in enumerate(rec.channel_names):
        if stim_ch_idx is not None and ch_idx == stim_ch_idx:
            continue

        ch_data   = data[ch_idx, :]
        ch_blank  = blanked[ch_idx, :]
        noise_std = float(np.std(ch_data[mask]))
        threshold = threshold_std * noise_std

        peaks, _ = scipy.signal.find_peaks(
            -ch_blank,
            height=threshold,
            distance=min_distance,
        )

        for spike_idx in peaks:
            amp    = float(ch_data[spike_idx])
            lat_ms = _compute_latency(spike_idx, rec.stim_indices,
                                      rec.sample_rate)
            wid_ms = _estimate_width(ch_data, spike_idx, rec.sample_rate)

            rows.append({
                'channel':     ch_name,
                'spike_index': int(spike_idx),
                'amplitude':   amp,
                'latency':     lat_ms,
                'width':       wid_ms,
            })

    return pd.DataFrame(rows) if rows else _empty_indirect_df()


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


def _build_quiet_mask(
    stim_indices: np.ndarray,
    n_samples: int,
    blank_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Boolean mask of shape (n_samples,): True = quiet (away from stim pulses).
    """
    mask = np.ones(n_samples, dtype=bool)
    pre  = int(0.005 * sample_rate)    # 5 ms before stim onset
    for idx in stim_indices:
        start = int(max(0, idx - pre))
        end   = int(min(n_samples, idx + blank_samples + 1))
        mask[start:end] = False
    return mask


def _compute_latency(
    spike_idx: int,
    stim_indices: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Latency from the most recent preceding stim pulse to this spike, in ms.
    Returns NaN if no preceding stim found.
    """
    preceding = stim_indices[stim_indices < spike_idx]
    if len(preceding) == 0:
        return float('nan')
    nearest_stim = preceding[-1]
    return float((spike_idx - nearest_stim) / sample_rate * 1000)


def _estimate_width(
    ch_data: np.ndarray,
    peak_idx: int,
    sample_rate: int,
) -> float:
    """Estimate spike width at half-maximum in ms."""
    try:
        inv    = -ch_data
        widths = scipy.signal.peak_widths(inv, [peak_idx], rel_height=0.5)
        return float((widths[3][0] - widths[2][0]) / sample_rate * 1000)
    except Exception:
        return float('nan')


def _empty_indirect_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'channel', 'spike_index', 'amplitude', 'latency', 'width'
    ])