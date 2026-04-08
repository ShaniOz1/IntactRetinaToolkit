"""
IntactRetinaToolkit.dataobj.analysis.direct
=============================================
Direct (stimulus-evoked) response detection.

- RHS: ICA-based artifact removal → spike shape validation
- EDF: threshold-based detection on blanked+filtered signal

Called via rec.detect_direct_response().
Results stored in rec.direct_response as a DataFrame.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import scipy.signal
from sklearn.decomposition import FastICA


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_direct_response(
    rec,
    data_type: str = 'filtered',
    win_size_ms: float = 15.0,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Detect direct responses and return a DataFrame.

    Dispatches to the RHS or EDF pipeline based on rec.source.

    Parameters
    ----------
    rec : RetinalRecording
    data_type : str
        Which data array to use for detection.
        RHS accepts 'filtered', 'blanked', or 'raw' (default 'filtered').
        EDF always uses 'blanked' regardless of this argument.
    win_size_ms : float
        Window extracted around each stim pulse, in ms.
    blank_ms : float
        Samples zeroed after each stim onset before detection, in ms.
    threshold : float or None
        If provided, use this value directly as the detection threshold
        (EDF pipeline) instead of computing it from the data.

    Returns
    -------
    pd.DataFrame
        Columns: channel, pulse_index, amplitude (µV), latency (ms),
        width (ms), amplitude_decay (exponential decay constant k, one
        value per channel).
    """
    if rec.stim_indices is None or len(rec.stim_indices) == 0:
        warnings.warn("[direct] stim_indices is None — returning empty DataFrame.",
                      UserWarning)
        return _empty_direct_df()

    if rec.source == 'rhs':
        df = _run_rhs(rec, data_type, win_size_ms)
        # RHS amplitudes are in µV — convert to mV to match EDF
        if not df.empty:
            df['amplitude_mV'] = df['amplitude_mV'] / 1000.0
    elif rec.source == 'edf':
        df = _run_edf(rec, win_size_ms, threshold)
    else:
        raise ValueError(
            f"Unsupported recording source: {rec.source!r}. "
            "Expected 'rhs' or 'edf'."
        )

    return add_amplitude_decay(df)


# ─────────────────────────────────────────────────────────────
# Shared: threshold-based detection on blanked data
# (used by both EDF and RHS when data_type='blanked')
# ─────────────────────────────────────────────────────────────

def _run_threshold_detection(
    rec,
    win_size_ms: float,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Threshold-based direct-response detection on rec.blanked_data.

    For each channel and each stim pulse, extracts a window of
    ``win_size_ms`` starting at the stim onset, finds the negative peak,
    and records it if it crosses the threshold.

    Parameters
    ----------
    rec : RetinalRecording
        Must have blanked_data and (for auto-threshold) recording_data.
    win_size_ms : float
        Window length per pulse in ms.
    threshold : float or None
        Fixed detection threshold (absolute µV). If None, computed
        per-channel from the raw signal.
    """
    window_samples = int(win_size_ms / 1000 * rec.sample_rate)

    data     = _resolve_data(rec, 'blanked')
    raw_data = _resolve_data(rec, 'raw')
    stim_ch_name = rec.stim_channel_name
    rows = []

    for ch_idx, ch_name in enumerate(rec.channel_names):
        if stim_ch_name and ch_name == stim_ch_name:
            continue

        ch_data = data[ch_idx, :]
        ch_threshold = (threshold if threshold is not None
                        else _compute_threshold(
                            raw_data[ch_idx, :], rec.stim_indices, rec.sample_rate
                        ))

        for pulse_idx, stim_idx in enumerate(rec.stim_indices):
            start = int(stim_idx)
            end   = int(min(stim_idx + window_samples, len(ch_data)))
            if start >= end:
                continue
            window = ch_data[start:end]
            if abs(window.min()) < ch_threshold:
                continue

            peak_local = int(np.argmin(window))
            amp        = float(window[peak_local])
            lat_ms     = peak_local / rec.sample_rate * 1000
            wid_ms     = _estimate_width(window, peak_local, rec.sample_rate)

            rows.append({
                'channel':      ch_name,
                'pulse_index':  pulse_idx,
                'amplitude_mV': amp,
                'latency_ms':   lat_ms,
                'width_ms':     wid_ms,
            })

    return pd.DataFrame(rows) if rows else _empty_direct_df()


# ─────────────────────────────────────────────────────────────
# EDF pipeline
# ─────────────────────────────────────────────────────────────

def _run_edf(rec, win_size_ms: float, threshold: float | None = None) -> pd.DataFrame:
    return _run_threshold_detection(rec, win_size_ms, threshold)


# ─────────────────────────────────────────────────────────────
# RHS pipeline
# ─────────────────────────────────────────────────────────────

def _run_rhs(rec, data_type: str, win_size_ms: float) -> pd.DataFrame:
    """
    Dispatch to the appropriate RHS detection method based on data_type.

    Parameters
    ----------
    data_type : {'blanked', 'filtered', 'raw'}
        - 'blanked'  : threshold-based detection (same as EDF pipeline).
        - 'filtered' : ICA artifact removal followed by spike detection.
        - 'raw'      : alternative detection method (not yet implemented).
    """
    if data_type == 'blanked':
        return _run_threshold_detection(rec, win_size_ms)
    elif data_type == 'filtered':
        return _run_rhs_ica(rec, win_size_ms)
    elif data_type == 'raw':
        return _run_rhs_raw(rec, win_size_ms)
    else:
        raise ValueError(
            f"data_type must be 'blanked', 'filtered', or 'raw'; got {data_type!r}"
        )


def _run_rhs_ica(rec, win_size_ms: float) -> pd.DataFrame:
    """RHS detection via ICA artifact removal → spike shape validation."""
    win_samples = int(win_size_ms / 1000 * rec.sample_rate)

    data   = _resolve_data(rec, 'filtered')
    pulses = _extract_pulses(rec.stim_indices, data, win_samples)
    # pulses: (n_pulses, n_channels, n_samples_per_pulse)

    stim_ch_idx = (rec.get_channel_index(rec.stim_channel_name)
                   if rec.stim_channel_name else None)

    rows = []
    for ch_idx, ch_name in enumerate(rec.channel_names):
        if stim_ch_idx is not None and ch_idx == stim_ch_idx:
            continue

        ch_pulses = pulses[:, ch_idx, :]           # (n_pulses, n_samples)
        cleaned   = _ica_remove_artifact(ch_pulses)

        for pulse_idx, pulse in enumerate(cleaned):
            spike = _detect_spike(pulse, rec.sample_rate)
            if spike is None:
                continue

            amp, lat_ms, wid_ms = spike
            rows.append({
                'channel':      ch_name,
                'pulse_index':  pulse_idx,
                'amplitude_mV': amp,
                'latency_ms':   lat_ms,
                'width_ms':     wid_ms,
            })

    return pd.DataFrame(rows) if rows else _empty_direct_df()


def _run_rhs_raw(rec, win_size_ms: float) -> pd.DataFrame:
    """RHS detection on raw data — implementation coming soon."""
    raise NotImplementedError(
        "_run_rhs_raw: detection on raw RHS data is not yet implemented."
    )


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

_DATA_ATTRS = {
    'filtered': 'filtered_data',
    'blanked':  'blanked_data',
    'raw':      'recording_data',
}

def _resolve_data(rec, data_type: str) -> np.ndarray:
    """Return the requested data array from rec.

    Parameters
    ----------
    data_type : {'filtered', 'blanked', 'raw'}
    """
    attr = _DATA_ATTRS.get(data_type)
    if attr is None:
        raise ValueError(
            f"data_type must be 'filtered', 'blanked', or 'raw'; got {data_type!r}"
        )
    data = getattr(rec, attr, None)
    if data is None:
        raise ValueError(
            f"rec.{attr} is None — cannot use data_type={data_type!r}"
        )
    return data


def _extract_pulses(
    stim_indices: np.ndarray,
    data: np.ndarray,
    win_samples: int,
) -> np.ndarray:
    """
    Extract fixed-length windows starting at each stim onset.

    Returns
    -------
    np.ndarray
        Shape (n_pulses, n_channels, win_samples).
    """
    result = []
    for idx in stim_indices:
        start = int(max(0, idx))
        end   = int(min(data.shape[1], idx + win_samples))
        seg   = data[:, start:end]
        # Pad if the last window is shorter
        if seg.shape[1] < win_samples:
            pad = np.zeros((data.shape[0], win_samples - seg.shape[1]))
            seg = np.hstack([seg, pad])
        result.append(seg)
    return np.array(result)   # (n_pulses, n_channels, win_samples)


def _ica_remove_artifact(ch_pulses: np.ndarray) -> np.ndarray:
    """
    Remove the stimulation artifact from a (n_pulses, n_samples) matrix
    using FastICA.

    The component with the largest peak-to-peak amplitude (the artifact)
    is zeroed out and the signal is reconstructed.

    Returns cleaned (n_pulses, n_samples).
    """
    n_pulses, n_samples = ch_pulses.shape
    if n_pulses < 4 or n_samples < 4:
        return ch_pulses

    try:
        n_components = min(n_pulses, n_samples, 4)
        ica = FastICA(n_components=n_components, max_iter=500, random_state=0)
        sources  = ica.fit_transform(ch_pulses)          # (n_pulses, n_comp)
        mixing   = ica.mixing_                           # (n_samples, n_comp)

        # Identify artifact component: largest peak-to-peak across pulses
        ptp = np.ptp(sources, axis=0)
        artifact_idx = int(np.argmax(ptp))

        # Zero it out and reconstruct
        sources_clean = sources.copy()
        sources_clean[:, artifact_idx] = 0.0
        cleaned = sources_clean @ mixing.T + ica.mean_
        return cleaned

    except Exception:
        return ch_pulses


def _detect_spike(
    pulse: np.ndarray,
    sample_rate: int,
) -> tuple[float, float, float] | None:
    """
    Validate whether a single pulse trace contains a spike and return
    (amplitude µV, latency ms, width ms) or None.

    Criteria (from spikes_analysis.py):
    - Hard amplitude threshold: min < -50 µV
    - Spike width at half-maximum: 0.4–2.0 ms
    - No more than 2 peaks above 70% of the largest peak
    """
    if pulse.min() > -50:
        return None

    inv = -pulse
    peaks, _ = scipy.signal.find_peaks(inv)
    if len(peaks) == 0:
        return None

    largest = peaks[int(np.argmax(inv[peaks]))]

    # Width check
    try:
        widths = scipy.signal.peak_widths(inv, [largest], rel_height=0.5)
        width_ms = (widths[3][0] - widths[2][0]) / sample_rate * 1000
        if not (0.4 <= width_ms <= 2.0):
            return None
    except Exception:
        return None

    # Peak count check
    high_peaks, _ = scipy.signal.find_peaks(inv,
                                             height=0.7 * inv[largest])
    if len(high_peaks) > 2:
        return None

    amp     = float(pulse[largest])
    lat_ms  = float(largest / sample_rate * 1000)
    return amp, lat_ms, width_ms


def _compute_threshold(
    ch_data: np.ndarray,
    stim_indices: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Compute detection threshold from raw data.

    For each stimulus, take the 1 ms window starting 10 ms after onset,
    average all windows, then return mean - 0.15 as the threshold.
    """
    window_size = int(0.001 * sample_rate)
    windows = []
    for idx in stim_indices:
        start = int(idx + 0.01 * sample_rate)
        end   = start + window_size
        if end <= len(ch_data):
            windows.append(ch_data[start:end])

    if len(windows) == 0:
        return 1.0 - 0.15

    avg_window = np.mean(np.array(windows), axis=0)
    return float(np.mean(avg_window)) - 0.15


def _estimate_width(
    window: np.ndarray,
    peak_local: int,
    sample_rate: int,
) -> float:
    """Estimate spike width at half-maximum in ms."""
    try:
        inv = -window
        widths = scipy.signal.peak_widths(inv, [peak_local], rel_height=0.5)
        return float((widths[3][0] - widths[2][0]) / sample_rate * 1000)
    except Exception:
        return float('nan')


def _empty_direct_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'channel', 'pulse_index', 'amplitude_mV', 'latency_ms', 'width_ms',
        'amplitude_decay_pct',
    ])


# ─────────────────────────────────────────────────────────────
# Amplitude decay enrichment
# ─────────────────────────────────────────────────────────────

def add_amplitude_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an ``amplitude_decay`` column to a direct-response DataFrame.

    For each channel, percent decay is computed as::

        (1 - mean(|amplitude| of last 3 pulses) / mean(|amplitude| of first 3 pulses)) * 100

    The result is stored in ``amplitude_decay`` — the same value for every
    row belonging to that channel.  Channels with fewer than 3 detected
    pulses receive ``NaN``.

    Parameters
    ----------
    df : pd.DataFrame
        Direct response DataFrame with columns: channel, pulse_index,
        amplitude.

    Returns
    -------
    pd.DataFrame
        New DataFrame with an ``amplitude_decay`` column added.
    """
    df = df.copy()
    df['amplitude_decay_pct'] = np.nan

    for ch, group in df.groupby('channel'):
        sorted_group = group.sort_values('pulse_index')
        y = np.abs(sorted_group['amplitude_mV'].values)

        if len(y) < 3:
            continue

        mean_first = np.mean(y[:3])
        if mean_first == 0:
            continue

        mean_last = np.mean(y[-3:])
        pct_decay = (1 - mean_last / mean_first) * 100
        df.loc[sorted_group.index, 'amplitude_decay_pct'] = pct_decay

    return df