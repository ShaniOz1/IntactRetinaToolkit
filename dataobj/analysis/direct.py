"""
IntactRetinaToolkit.dataobj.analysis.direct
=============================================
Direct (stimulus-evoked) response detection.

- RHS raw  : SALPA artifact suppression → threshold-based detection
- RHS filt : ICA-based artifact removal → spike shape validation
- EDF      : threshold-based detection on blanked+filtered signal

Called via rec.detect_direct_response().
Results stored in rec.direct_response as a DataFrame.
"""

from __future__ import annotations
import warnings
from math import comb
import numpy as np
import pandas as pd
import scipy.signal
from scipy.optimize import curve_fit
from sklearn.decomposition import FastICA


# ─────────────────────────────────────────────────────────────
# SALPA: Subtraction of Artifacts by Local Polynomial Approximation
# Wagenaar & Potter, J. Neurosci. Methods 120 (2002) 113-120
# ─────────────────────────────────────────────────────────────

def salpa(
    V,
    N: int = 60,
    d: int = 5,
    b_squared: float = 5.0,
    sat_low: float = -2048,
    sat_high: float = 2047,
) -> np.ndarray:
    """
    Remove stimulation artifacts via local cubic polynomial subtraction.

    Parameters
    ----------
    V         : 1-D array of raw voltages (ADC counts or µV)
    N         : window half-length in samples (paper default 75 → 3 ms at 25 kHz)
    d         : deviation-check window width in samples (paper default d = N/10)
    b_squared : noise-colour correction factor b² (paper reports b² ≈ 5)
    sat_low   : ADC value indicating negative saturation (default: 12-bit)
    sat_high  : ADC value indicating positive saturation (default: 12-bit)

    Returns
    -------
    np.ndarray  Cleaned signal, same length as V. Saturated samples → 0.
    """
    V = np.asarray(V, dtype=float)
    n = len(V)
    v = np.zeros(n)

    # ── S^{-1}: maps W_l → cubic polynomial coefficients a_k ──────────
    def _build_S_inv(N):
        j = np.arange(-N, N + 1, dtype=float)
        T = np.array([np.sum(j ** k) for k in range(7)])
        S = np.array([[T[k + l] for l in range(4)] for k in range(4)])
        return np.linalg.inv(S)

    S_inv = _build_S_inv(N)

    # ── σ_V² via MAD (robust against large artifact excursions) ────────
    mad = np.median(np.abs(V - np.median(V)))
    sigma_V2 = (mad / 0.6745) ** 2
    D2_threshold = 9.0 * b_squared * d * sigma_V2

    # ── Helpers ─────────────────────────────────────────────────────────
    def _compute_W_full(nc):
        j = np.arange(-N, N + 1, dtype=float)
        seg = V[nc - N: nc + N + 1]
        return np.array([np.dot(j ** l, seg) for l in range(4)])

    def _poly_val(a, offset):
        return sum(a[k] * (offset ** k) for k in range(4))

    def _advance_W(W, nc):
        W_new = np.zeros(4)
        for k in range(4):
            acc = sum((-1) ** (k - l) * comb(k, l) * W[l] for l in range(k + 1))
            acc += (N ** k) * V[nc + N + 1]
            acc -= ((-N - 1) ** k) * V[nc - N]
            W_new[k] = acc
        return W_new

    def _deviation(a, nc):
        D = 0.0
        for pos in range(nc + N - (d - 1), nc + N + 1):
            if 0 <= pos < n:
                D += V[pos] - _poly_val(a, pos - nc)
        return D

    # ── Main loop ───────────────────────────────────────────────────────
    i = 0
    while i < n:

        # Saturation blanking
        if V[i] == sat_low or V[i] == sat_high:
            v[i] = 0
            i += 1
            continue

        # Depeg phase — find first polynomial fit that passes deviation check
        depeg_start = i
        if depeg_start + N >= n:
            v[depeg_start:] = V[depeg_start:]
            break

        nc = depeg_start + N
        accepted = False

        while nc + N < n:
            W = _compute_W_full(nc)
            a = S_inv @ W
            D = _deviation(a, nc)

            if D * D <= D2_threshold:
                for pos in range(depeg_start, nc + 1):
                    v[pos] = V[pos] - _poly_val(a, pos - nc)
                i = nc + 1
                accepted = True
                break

            nc += 1

        if not accepted:
            v[depeg_start:] = V[depeg_start:]
            break

        # Bulk phase — O(1) recursive window update per sample
        W = _compute_W_full(nc)
        bulk_nc = nc

        while i < n:
            if V[i] == sat_low or V[i] == sat_high:
                break

            if i - 1 + N + 1 >= n or i - 1 - N < 0:
                v[i] = V[i]
                i += 1
                bulk_nc = i - 1
                continue

            W = _advance_W(W, bulk_nc)
            bulk_nc = i
            a = S_inv @ W
            v[i] = V[i] - a[0]
            i += 1

    return v


# ─────────────────────────────────────────────────────────────
# Artifact suppression: threshold + polynomial breakaway
# ─────────────────────────────────────────────────────────────

def suppress_stim_artifact(
    segment: np.ndarray,
    mv_threshold: float = 3000.0,
    N_fit: int = 5,
    breakaway_thr: float = 500.0,
    plot: bool = False,
) -> np.ndarray:
    """
    Detect and blank the stimulation artifact in a single pulse segment.

    Algorithm
    ---------
    1. Find ``last_cross``: the last sample where |signal| >= mv_threshold.
       Up to this point the signal is unambiguously artifact.
    2. Fit a cubic polynomial to the ``N_fit`` samples that start at
       ``last_cross``, modelling the smooth sub-threshold artifact tail.
    3. Scan forward from ``last_cross``: the first sample whose residual
       |signal − fit| exceeds ``breakaway_thr`` marks the end of the
       artifact — the signal has broken away from the smooth decay.
    4. Blank [0, artifact_end) with zeros and return the cleaned segment.

    Parameters
    ----------
    segment       : 1-D raw signal for one pulse (µV for RHS data)
    mv_threshold  : amplitude above which a sample is definitely artifact
                    (µV; 5000.0 = 5 mV, appropriate for RHS ±6389 µV range)
    N_fit         : samples used to fit the polynomial to the artifact tail
    breakaway_thr : fixed residual (µV) above which the signal is considered
                    to have broken away from the artifact curve
    plot          : if True, show a diagnostic figure with two subplots:
                    (top) raw segment + tail region + fitted curve;
                    (bottom) residual vs sample with breakaway threshold.
    """
    import matplotlib.pyplot as plt

    seg = np.asarray(segment, dtype=float)
    n = len(seg)
    samples = np.arange(n)

    # ── Step 1: last above-threshold sample ───────────────────────────
    crossings = np.where(np.abs(seg) >= mv_threshold)[0]
    if len(crossings) == 0:
        return seg.copy()              # no large artifact — return unchanged

    last_cross = int(crossings[-1])

    # ── Step 2: exponential fit to N_fit samples starting at last_cross ─
    # Model: f(x) = a · exp(b · x) + c
    # b < 0 → decaying tail, b > 0 → rising tail; both are tried and the
    # better fit (lower residual sum-of-squares) is kept.
    fit_end = min(n, last_cross + N_fit)
    fit_len = fit_end - last_cross
    artifact_end = n                   # fallback: blank to end of segment

    # Arrays for plotting — populated only if fit is possible
    fit_x      = np.array([], dtype=float)
    fit_curve  = np.array([], dtype=float)
    residuals  = np.full(n, np.nan)

    def _exp_model(x, a, b, c):
        return a * np.exp(b * x) + c

    if fit_len >= 3:
        x = np.arange(fit_len, dtype=float)
        y = seg[last_cross:fit_end]
        c0 = float(y[-1])
        a0 = float(y[0]) - c0

        best_params = None
        best_sse    = np.inf
        for b0 in (-1.0 / max(fit_len, 1), 1.0 / max(fit_len, 1)):
            try:
                popt, _ = curve_fit(
                    _exp_model, x, y,
                    p0=[a0, b0, c0],
                    maxfev=2000,
                )
                sse = float(np.sum((_exp_model(x, *popt) - y) ** 2))
                if sse < best_sse:
                    best_sse    = sse
                    best_params = popt
            except Exception:
                pass

        if best_params is not None:
            x_full    = np.arange(n - last_cross, dtype=float)
            fit_x     = np.arange(last_cross, n, dtype=float)
            fit_curve = _exp_model(x_full, *best_params)

            # ── Step 3: residuals over full segment from last_cross ───────
            residuals[last_cross:] = np.abs(
                seg[last_cross:] - _exp_model(x_full, *best_params)
            )

            crossings = np.where(residuals[last_cross:] > breakaway_thr)[0]
            artifact_end = (last_cross + int(crossings[0])
                            if len(crossings) > 0 else n)

    # ── Step 4: blank artifact duration ──────────────────────────────
    result = seg.copy()
    result[:artifact_end] = 0.0

    # ── Optional diagnostic plot ──────────────────────────────────────
    if plot:
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(10, 5))
        gs  = gridspec.GridSpec(
            2, 2,
            width_ratios=[5, 1],
            hspace=0.35, wspace=0.35,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[:, 1])   # full height, right column

        uv2mv = 1e-3   # convert µV → mV for all plots

        # Top-left: raw signal, tail region, fitted curve
        ax1.plot(samples, seg * uv2mv, color='black', linewidth=0.8, label='raw segment')
        if last_cross < n:
            tail_end = min(last_cross + N_fit, n)
            ax1.plot(
                samples[last_cross:tail_end], seg[last_cross:tail_end] * uv2mv,
                color='purple', linewidth=1.2, label='tail region',
            )
        if len(fit_curve):
            ax1.plot(fit_x, fit_curve * uv2mv, color='green', linewidth=1.4,
                     linestyle='--', label='fitted curve')
        ax1.axvline(artifact_end, color='red', linewidth=0.8,
                    linestyle=':', label=f'artifact end ({artifact_end})')
        ax1.set_ylabel('amplitude (mV)')
        ax1.set_ylim(-7, 7)
        ax1.legend(fontsize=7, frameon=False)
        ax1.set_title('Artifact suppression — signal')

        # Bottom-left: residual vs sample + breakaway threshold
        valid = ~np.isnan(residuals)
        ax2.plot(samples[valid], residuals[valid] * uv2mv, color='black',
                 linewidth=0.8, label='|residual|')
        ax2.axhline(breakaway_thr * uv2mv, color='red', linewidth=1.0,
                    linestyle='--',
                    label=f'breakaway thr ({breakaway_thr * uv2mv:.3f} mV)')
        ax2.set_ylabel('|signal − fit| (mV)')
        ax2.set_xlabel('sample')
        ax2.set_ylim(-7, 7)
        ax2.legend(fontsize=7, frameon=False)
        ax2.set_title('Residual vs breakaway threshold')

        # Right: cleaned signal (full figure height)
        ax3.plot(samples, result * uv2mv, color='black', linewidth=0.5)
        ax3.set_xlabel('sample')
        ax3.set_ylabel('amplitude (mV)')
        ax3.set_title('cleaned', fontsize=8)

        plt.show()

    return result


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
        return _run_rhs_raw(rec)
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


def _run_rhs_raw(
    rec,
    win_size_ms: float = 30,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    RHS detection on raw data: artifact suppression per stim pulse → threshold detection.

    For each channel and each stim pulse:
      1. Extract the raw pulse window (stim onset → stim onset + win_size_ms).
      2. Apply ``suppress_stim_artifact`` with its default parameters to blank
         the artifact duration.
      3. Run threshold-based peak detection on the cleaned window.

    Parameters
    ----------
    rec : RetinalRecording
    win_size_ms : float
    threshold : float or None
        Detection threshold (µV). If None, computed per-channel from raw data.
    """
    window_samples = int(win_size_ms / 1000 * rec.sample_rate)
    raw_data = _resolve_data(rec, 'raw')
    stim_ch_name = rec.stim_channel_name
    rows = []

    for ch_idx, ch_name in enumerate(rec.channel_names):
        if stim_ch_name and ch_name == stim_ch_name:
            continue
        ch_idx = 8
        raw_ch = raw_data[ch_idx, :]
        ch_threshold = (threshold if threshold is not None
                        else _compute_threshold(
                            raw_ch, rec.stim_indices, rec.sample_rate
                        ))

        for pulse_idx, stim_idx in enumerate(rec.stim_indices):
            start = int(stim_idx)
            end   = int(min(stim_idx + window_samples, len(raw_ch)))
            if start >= end:
                continue

            segment = raw_ch[start:end]
            window  = suppress_stim_artifact(segment)


            if len(window) == 0 or abs(window.min()) < ch_threshold:
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