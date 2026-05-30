"""
IntactRetinaToolkit.dataobj.analysis.direct
=============================================
Direct (stimulus-evoked) response detection.

- RHS raw     : artifact suppression → local-minimum peak detection
- RHS blanked : threshold-based detection on blanked data
- EDF         : threshold-based detection on blanked+filtered signal

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
    N_fit: int = 10,
    breakaway_thr: float = 300.0,
    plot: bool = False,
    sample_rate: int = 20000,
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
    sample_rate   : recording sample rate in Hz, used to label the x-axis in ms
    """
    import matplotlib.pyplot as plt

    seg = np.asarray(segment, dtype=float)
    n = len(seg)
    samples = np.arange(n)
    ms_axis = samples / sample_rate * 1000.0   # sample index → ms

    # ── Step 1: last above-threshold sample ───────────────────────────
    crossings = np.where(np.abs(seg) >= mv_threshold)[0]
    if len(crossings) == 0:
        return seg.copy()              # no large artifact — return unchanged

    last_cross = int(crossings[-1])

    # ── Step 2: exponential fit to N_fit samples starting at last_cross ──
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
                popt, _ = curve_fit(_exp_model, x, y,p0=[a0, b0, c0],maxfev=2000)
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

            # ── Step 3: residuals from last_cross ────────────────────────
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

        fig = plt.figure(figsize=(6, 6))
        gs  = gridspec.GridSpec(
            2, 2,
            width_ratios=[5, 1],
            hspace=0.35, wspace=0.35,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[:, 1])   # full height, right column

        uv2mv = 1e-3   # convert µV → mV for all plots
        artifact_end_ms = artifact_end / sample_rate * 1000.0

        # Top-left: raw signal, tail region, fitted curve
        ax1.plot(ms_axis, seg * uv2mv, color='grey', linewidth=0.8, label='raw segment')
        if last_cross < n:
            tail_end = min(last_cross + N_fit, n)
            ax1.plot(
                ms_axis[last_cross:tail_end], seg[last_cross:tail_end] * uv2mv,
                color='purple', linewidth=2.0, label='tail region',
            )
        if len(fit_curve):
            ax1.plot(fit_x / sample_rate * 1000.0, fit_curve * uv2mv, color='green',
                     linewidth=5.0, alpha=0.35, label='fitted curve')
        ax1.axvline(artifact_end_ms, color='red', linewidth=0.8,
                    linestyle=':', label=f'artifact end ({artifact_end_ms:.2f} ms)')
        ax1.set_xlim(0, 1.2)
        ax1.set_ylabel('amplitude (mV)')
        ax1.set_ylim(-7, 7)
        ax1.legend(fontsize=7, frameon=False)
        ax1.set_title('Artifact suppression — signal', fontsize=7)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Bottom-left: residual vs sample + breakaway threshold
        valid = ~np.isnan(residuals)
        ax2.plot(ms_axis[valid], residuals[valid] * uv2mv, color='black',
                 linewidth=0.8, label='|residual|')
        ax2.axhline(breakaway_thr * uv2mv, color='red', linewidth=1.0,
                    linestyle='--',
                    label=f'breakaway thr ({breakaway_thr * uv2mv:.3f} mV)')
        ax2.set_ylabel('|signal − fit| (mV)')
        ax2.set_xlabel('time (ms)')
        ax2.set_xlim(0, 5)
        ax2.set_ylim(-7, 7)
        ax2.legend(fontsize=7, frameon=False)
        ax2.set_title('Residual vs breakaway threshold', fontsize=7)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Right: cleaned signal, clipped to 10 ms
        ax3.plot(ms_axis, result * uv2mv, color='black', linewidth=0.5)
        ax3.set_xlim(0, 20)
        ax3.set_xlabel('time (ms)')
        ax3.set_ylabel('amplitude (mV)')
        ax3.set_title('cleaned', fontsize=7)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        plt.show()

    return result


# ─────────────────────────────────────────────────────────────
# Artifact shape validation
# ─────────────────────────────────────────────────────────────

def evaluate_artifact_shape(
    segment: np.ndarray,
    sample_rate: int,
    duration_us: float,
    interphase_us: float,
    negative_first: bool = True,
    match_threshold: float = 0.5,
    charge_balance_thr: float = 0.3,
    plot: bool = False,
) -> tuple[bool, float, float]:
    """
    Check whether the beginning of a raw pulse segment contains a biphasic
    stimulation artifact matching the expected waveform shape.

    A rectangular biphasic template is built from the stimulus parameters and
    slid over the front of the segment.  At each position a Pearson-like
    normalised correlation is computed (amplitude-invariant).  If the best
    match exceeds ``match_threshold`` the artifact is considered present and
    processing should continue; otherwise the window contains no recognisable
    artifact and there is no point running artifact suppression or threshold
    detection.

    Template shape
    --------------
    negative_first=True  :  [─── phase1 (−1) ───][── gap (0) ──][─── phase2 (+1) ───]
    negative_first=False :  [─── phase1 (+1) ───][── gap (0) ──][─── phase2 (−1) ───]

    Parameters
    ----------
    segment         : 1-D raw voltage array for one pulse window (µV)
    sample_rate     : recording sample rate (samples per second)
    duration_us     : duration of each phase of the biphasic pulse (µs)
    interphase_us   : duration of the interphase gap between the two phases (µs)
    negative_first  : if True the first phase is negative; if False it is positive
    match_threshold : minimum normalised correlation to accept the artifact
                      as a valid biphasic pulse (0–1, default 0.5)
    plot            : if True, display a diagnostic figure (template | signal)

    Returns
    -------
    (is_valid, best_corr) : tuple[bool, float]
        is_valid  – True when the best match meets or exceeds match_threshold
        best_corr – peak normalised cross-correlation value (useful for diagnostics)
    """
    # ── Build biphasic rectangular template ─────────────────────────────
    # Shape: phase1 → interphase → phase2 → 500 µs silence
    ph_samples  = max(1, int(round(duration_us   / 1_000_000.0 * sample_rate)))
    gap_samples = max(0, int(round(interphase_us / 1_000_000.0 * sample_rate)))
    pad_samples = max(1, int(round(500           / 1_000_000.0 * sample_rate)))
    # core: 1 leading zero + phase1 + gap + phase2 + 1 trailing zero + 500 µs tail
    template_len = 2 * ph_samples + gap_samples + 2 + pad_samples

    sign = -1.0 if negative_first else 1.0
    template = np.zeros(template_len)
    template[1 : 1 + ph_samples]                                          =  sign   # phase 1
    # interphase gap stays 0
    template[1 + ph_samples + gap_samples : 1 + 2 * ph_samples + gap_samples] = -sign   # phase 2
    # trailing zero + 500 µs pad stay 0

    # Normalise template to zero mean, unit norm (amplitude-invariant)
    t = template - template.mean()
    t_norm_denom = np.linalg.norm(t)
    if t_norm_denom < 1e-12:
        return False, 0.0, 0.0
    t_norm = t / t_norm_denom

    # ── Search region: positions 0–3 only ───────────────────────────────
    max_start = 3
    sig = segment.astype(float)

    if len(sig) < template_len or np.ptp(sig) == 0:
        return False, 0.0, 0.0

    # ── Try positions 0, 1, 2, 3; pick best normalised correlation ───────
    n_positions = min(max_start + 1, len(sig) - template_len + 1)
    best_corr = 0.0
    best_pos  = 0
    for i in range(n_positions):
        s = sig[i : i + template_len]
        s_centered = s - s.mean()
        denom = np.linalg.norm(s_centered)
        if denom < 1e-12:
            continue
        corr = float(np.dot(t_norm, s_centered / denom))
        if corr > best_corr:
            best_corr = corr
            best_pos  = i

    if best_corr < match_threshold:
        return False, best_corr, 0.0

    # ── Charge-balance check on the best-aligned window ──────────────────
    best_win  = sig[best_pos : best_pos + template_len]
    imbalance = abs(np.sum(best_win)) / (np.sum(np.abs(best_win)) + 1e-12)

    is_valid = imbalance <= charge_balance_thr

    # ── Optional diagnostic plot ─────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt

        best_window = sig[best_pos : best_pos + template_len]
        t_axis = np.arange(template_len) / sample_rate * 1000  # ms

        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(5, 3))

        ax_t.plot(t_axis, template,             color='black', linewidth=1.0)
        ax_s.plot(t_axis, best_window * 1e-3,   color='black', linewidth=0.8)  # µV → mV

        ax_s.set_title(f'corr={best_corr:.2f}  imb={imbalance:.2f}', fontsize=7)

        for ax in (ax_t, ax_s):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('time (ms)')

        ax_t.set_ylabel('amplitude (a.u.)')
        ax_s.set_ylabel('amplitude (mV)')
        ax_s.set_ylim(-7, 7)

        plt.tight_layout()
        plt.show()

    return is_valid, best_corr, imbalance


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def run_direct_response(
    rec,
    data_type: str = 'raw',
    win_size_ms: float = 15.0,
    threshold: float | None = None,
    plot: bool = True,
    output_folder: str | None = None,
) -> pd.DataFrame:
    """
    Detect direct responses and return a DataFrame.

    Dispatches to the RHS or EDF pipeline based on rec.source.

    Parameters
    ----------
    rec : RetinalRecording
    data_type : str
        Which data array to use for detection.
        RHS accepts 'raw' (default) or 'blanked'.
        EDF always uses 'blanked' regardless of this argument.
    win_size_ms : float
        Window extracted around each stim pulse, in ms.
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
        df = _run_rhs(rec, data_type, win_size_ms, plot=plot, output_folder=output_folder)
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
        # ch_threshold = threshold

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

def _run_rhs(rec, data_type: str, win_size_ms: float, plot: bool = True, output_folder: str | None = None) -> pd.DataFrame:
    """
    Dispatch to the appropriate RHS detection method based on data_type.

    Parameters
    ----------
    data_type : {'raw', 'blanked'}
        - 'raw'     : artifact suppression + local-minimum peak detection.
        - 'blanked' : threshold-based detection on pre-blanked data.
    """
    if data_type == 'raw':
        return _run_rhs_raw(rec, plot=plot, output_folder=output_folder)
    elif data_type == 'blanked':
        return _run_threshold_detection(rec, win_size_ms)
    else:
        raise ValueError(
            f"data_type must be 'raw' or 'blanked'; got {data_type!r}"
        )


def _run_rhs_raw(
    rec,
    win_size_ms: float = 30,
    threshold: float | None = None,
    interphase_us: float = 50,
    negative_first: bool = True,
    artifact_match_threshold: float = 0.9,
    charge_balance_thr: float = 0.3,
    plot: bool = True,
    output_folder: str | None = None,
) -> pd.DataFrame:
    """
    RHS detection on raw data: per-pulse artifact suppression → peak detection.

    For each channel and each stim pulse:
      1. Extract the raw pulse window (stim onset → stim onset + win_size_ms).
      2. Validate biphasic artifact shape via ``evaluate_artifact_shape``.
      3. Suppress the artifact via ``suppress_stim_artifact`` and smooth with
         a 0.2 ms moving-average.
      4. Detect the negative peak via ``_evaluate_peak``: deepest sample from
         2 ms onward that is a strict local minimum in a ±0.2 ms neighbourhood,
         with amplitude < −0.02 mV and half-width < 5 ms.

    Parameters
    ----------
    rec : RetinalRecording
    win_size_ms : float
    threshold : float or None
        Detection threshold (µV). If None, computed per-channel from raw data.
    duration_us : float
        Expected duration of each phase of the biphasic stimulus pulse (µs).
        Used to build the shape-matching template.
    interphase_us : float
        Expected interphase gap between the two stimulus phases (µs).
    negative_first : bool
        True if the cathodic (negative) phase comes first.
    artifact_match_threshold : float
        Minimum normalised correlation (0–1) to consider the artifact present.
        Pulses whose artifact correlates below this value are skipped.
    plot : bool
        If True, display one figure per channel showing the cleaned window for
        every pulse (10 subplots per row, x-axis in ms, clipped to 20 ms).
    """
    import matplotlib.pyplot as plt

    duration_us = rec.stim_phase_duration_us

    window_samples = int(win_size_ms / 1000 * rec.sample_rate)
    ph_samples   = max(1, int(round(duration_us   / 1_000_000.0 * rec.sample_rate)))
    gap_samples  = max(0, int(round(interphase_us / 1_000_000.0 * rec.sample_rate)))
    pad_samples  = max(1, int(round(500           / 1_000_000.0 * rec.sample_rate)))
    template_len = 2 * ph_samples + gap_samples + 2 + pad_samples
    art_axis     = np.arange(template_len) / rec.sample_rate * 1000.0
    raw_data = _resolve_data(rec, 'raw')
    stim_ch_name = rec.stim_channel_name
    rows = []
    valid_spike_counts: dict[str, int] = {}    # ch_name → n valid spikes
    all_channel_windows: dict[str, list] = {}  # ch_name → [window, ...]

    min_lat_samp  = int(2.5  * rec.sample_rate / 1000)   # 2.5 ms in samples
    max_lat_samp  = int(10.0 * rec.sample_rate / 1000)   # 10 ms in samples
    local_min_rad = int(0.2  * rec.sample_rate / 1000)   # ±0.2 ms neighbourhood

    for ch_idx, ch_name in enumerate(rec.channel_names):
        if stim_ch_name and ch_name == stim_ch_name:
            continue
        # ch_idx = 1
        raw_ch = raw_data[ch_idx, :]
        ch_threshold = (threshold if threshold is not None
                        else _compute_threshold(
                            raw_ch, rec.stim_indices, rec.sample_rate
                        ))

        if plot:
            all_channel_windows[ch_name] = []

        for pulse_idx, stim_idx in enumerate(rec.stim_indices):
            start = int(stim_idx)
            end   = int(min(stim_idx + window_samples, len(raw_ch)))
            if start >= end:
                continue

            segment = raw_ch[start:end]

            # ── Step 2: biphasic artifact shape validation ────────────────
            artifact_valid, corr, imbalance = evaluate_artifact_shape(
                segment,
                sample_rate=rec.sample_rate,
                duration_us=duration_us,
                interphase_us=interphase_us,
                negative_first=negative_first,
                match_threshold=artifact_match_threshold,
                charge_balance_thr=charge_balance_thr,
            )

            # ── Step 3: artifact suppression + smoothing ──────────────────
            rms_before = float(np.sqrt(np.mean(segment ** 2))) if artifact_valid else float('nan')
            window = suppress_stim_artifact(segment, sample_rate=rec.sample_rate)
            _k     = max(1, int(0.4 * rec.sample_rate / 1000))
            window = np.convolve(window, np.ones(_k) / _k, mode='same')
            rms_after  = float(np.sqrt(np.mean(window ** 2)))  if artifact_valid else float('nan')

            # ── Step 4: peak detection ────────────────────────────────────
            spike_valid, peak_local, amp, lat_ms, wid_ms = (
                _evaluate_peak(window, min_lat_samp, max_lat_samp, local_min_rad, rec.sample_rate)
                if artifact_valid else
                (False, -1, float('nan'), float('nan'), float('nan'))
            )

            if spike_valid:
                valid_spike_counts[ch_name] = valid_spike_counts.get(ch_name, 0) + 1

            if plot:
                stored_peak = peak_local if spike_valid else -1
                all_channel_windows[ch_name].append((window, corr, segment[:template_len], imbalance, artifact_valid, spike_valid, stored_peak))

            rows.append({
                'channel':      ch_name,
                'pulse_index':  pulse_idx,
                'amplitude_mV': amp    if spike_valid else float('nan'),
                'latency_ms':   lat_ms if spike_valid else float('nan'),
                'width_ms':     wid_ms if spike_valid else float('nan'),
                'corr':         corr,
                'imbalance':    imbalance,
                'rms_before':   rms_before,
                'rms_after':    rms_after if spike_valid else float('nan'),
            })

    # ── One figure for the entire file: one subplot per channel ──────────
    if plot and all_channel_windows:
        ch_names_plot = list(all_channel_windows.keys())
        n_channels    = len(ch_names_plot)
        n_cols        = 8
        n_rows        = int(np.ceil(n_channels / n_cols))
        ms_axis       = np.arange(window_samples) / rec.sample_rate * 1000.0

        _CLR_ARTIFACT  = '#8B0000'   # dark red  — artifact invalid
        _CLR_NO_SPIKE  = 'grey'      # grey      — artifact ok, no spike
        _CLR_SPIKE     = 'black'     # black     — valid spike
        _CLR_AVG       = 'steelblue' # blue      — average trace
        _CLR_ANNOT     = 'purple'    # purple    — peak annotation

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 1.5, n_rows * 2.5),
            squeeze=False,
        )
        fig.suptitle(rec.file_name, fontsize=9)

        # Legend patches drawn on the first visible axis after the loop
        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=_CLR_ARTIFACT, label='artifact invalid'),
            mpatches.Patch(color=_CLR_NO_SPIKE, label='no spike'),
            mpatches.Patch(color=_CLR_SPIKE,    label='valid spike'),
            mpatches.Patch(color=_CLR_AVG,      label='average'),
        ]

        for ax_i, ch in enumerate(ch_names_plot):
            ax = axes[ax_i // n_cols][ax_i % n_cols]
            corrs = []
            windows_list = []
            peak_locals  = []
            amps_mv      = []
            wids_ms      = []
            imbs         = []
            for win, corr, _, imb, artifact_ok, spike_ok, pk in all_channel_windows[ch]:
                if not artifact_ok:
                    color = _CLR_ARTIFACT
                elif not spike_ok:
                    color = _CLR_NO_SPIKE
                else:
                    color = _CLR_SPIKE
                ax.plot(ms_axis[:len(win)], win * 1e-3,
                        color=color, linewidth=0.4, alpha=0.3)
                corrs.append(corr)
                imbs.append(imb)
                if spike_ok:
                    windows_list.append(win)
                    peak_locals.append(pk)
                    amps_mv.append(float(win[pk]) * 1e-3)
                    wids_ms.append(_estimate_width(win, pk, rec.sample_rate))

            if len(windows_list) > 5:
                min_len  = min(len(w) for w in windows_list)
                avg_win  = np.mean([w[:min_len] for w in windows_list], axis=0)
                avg_ms   = ms_axis[:min_len]

                med_lat_ms  = float(np.median(peak_locals)) / rec.sample_rate * 1000
                med_amp_mv  = float(np.median([a for a in amps_mv if not np.isnan(a)]))
                wids_valid  = [w for w in wids_ms if not np.isnan(w)]
                med_wid_ms  = float(np.median(wids_valid)) if wids_valid else float('nan')

                ax.plot(avg_ms, avg_win * 1e-3, color=_CLR_AVG, linewidth=1.0, alpha=0.9)
                ax.plot(med_lat_ms, med_amp_mv, 'v', color=_CLR_ANNOT, markersize=3, zorder=5)
                ax.annotate(
                    f'{med_amp_mv:.2f} mV\nlat={med_lat_ms:.2f} ms\nwid={med_wid_ms:.2f} ms',
                    xy=(med_lat_ms, med_amp_mv),
                    xytext=(3, -6), textcoords='offset points',
                    fontsize=6, color=_CLR_ANNOT,
                )

            ax.set_xlim(0, 20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            mean_corr = float(np.mean(corrs)) if corrs else float('nan')
            mean_imb  = float(np.mean(imbs))  if imbs  else float('nan')
            ax.set_title(f'ch-{ch}\ncorr={mean_corr:.2f} | sum={mean_imb:.2f}', fontsize=6)
            ax.tick_params(labelsize=6)

        # Hide unused subplots except the last slot, which holds the legend
        for ax_i in range(n_channels, n_rows * n_cols - 1):
            axes[ax_i // n_cols][ax_i % n_cols].set_visible(False)

        ax_legend = axes[-1][-1]
        ax_legend.set_visible(True)
        ax_legend.axis('off')
        ax_legend.legend(handles=legend_handles, fontsize=6, frameon=False, loc='center')

        plt.tight_layout()
        if output_folder:
            import os
            fig.savefig(os.path.join(output_folder, f'{rec.file_name}_direct_response.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        # ── Artifact figure ───────────────────────────────────────────────
        fig2, axes2 = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 1.8, n_rows * 1.8),
            squeeze=False,
        )
        fig2.suptitle(f'{rec.file_name} — artifact', fontsize=9)

        for ax_i, ch in enumerate(ch_names_plot):
            ax = axes2[ax_i // n_cols][ax_i % n_cols]
            corrs = []
            imbs  = []
            for _, corr, art, imb, valid, _spike_ok, _pk in all_channel_windows[ch]:
                color = 'steelblue' if valid else 'red'
                ax.plot(art_axis[:len(art)], art * 1e-3,
                        color=color, linewidth=0.4, alpha=0.3)
                corrs.append(corr)
                imbs.append(imb)
            ax.set_xlim(0, art_axis[-1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            mean_corr = float(np.mean(corrs)) if corrs else float('nan')
            mean_imb  = float(np.mean(imbs))  if imbs  else float('nan')
            ax.set_title(f'{ch}\ncorr={mean_corr:.2f}  imb={mean_imb:.2f}', fontsize=5)
            ax.tick_params(labelsize=5)

        for ax_i in range(n_channels, n_rows * n_cols):
            axes2[ax_i // n_cols][ax_i % n_cols].set_visible(False)

        plt.tight_layout()
        if output_folder:
            import os
            fig2.savefig(os.path.join(output_folder, f'{rec.file_name}_artifact.png'),
                         dpi=150, bbox_inches='tight')
            plt.close(fig2)
        else:
            plt.show()

    if not rows:
        return _empty_direct_df()

    df = pd.DataFrame(rows)
    # Mirror the figure's annotation threshold: only keep channels with >5 valid spikes
    responsive = {ch for ch, n in valid_spike_counts.items() if n > 5}
    df = df[df['channel'].isin(responsive)]
    return df if not df.empty else _empty_direct_df()


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

_DATA_ATTRS = {
    'blanked': 'blanked_data',
    'raw':     'recording_data',
}

def _resolve_data(rec, data_type: str) -> np.ndarray:
    """Return the requested data array from rec.

    Parameters
    ----------
    data_type : {'blanked', 'raw'}
    """
    attr = _DATA_ATTRS.get(data_type)
    if attr is None:
        raise ValueError(
            f"data_type must be 'blanked' or 'raw'; got {data_type!r}"
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


def _evaluate_peak(
    window: np.ndarray,
    min_lat_samp: int,
    max_lat_samp: int,
    local_min_rad: int,
    sample_rate: int,
) -> tuple:
    """
    Find and validate the negative peak in a cleaned pulse window.

    Tries up to twice: if the first candidate fails, masks it and searches
    for the next deepest sample.

    Returns (spike_valid, peak_local, amp_uv, lat_ms, wid_ms).
    spike_valid is False and the other values are -1 / nan if no valid peak found.
    """
    search_end = min(max_lat_samp, len(window))
    if search_end <= min_lat_samp:
        return False, -1, float('nan'), float('nan'), float('nan')

    min_wid_samp = int(0.5 * sample_rate / 1000)
    max_wid_samp = int(5.0 * sample_rate / 1000)

    # Find valleys in [min_lat_samp, max_lat_samp): negate for find_peaks
    candidates, _ = scipy.signal.find_peaks(
        -window[min_lat_samp:search_end],
        height=25.0,                          # amp < -25 µV
        width=(min_wid_samp, max_wid_samp),   # half-width < 5 ms
        distance=local_min_rad,               # enforce ±0.2 ms exclusion radius
    )

    if len(candidates) == 0:
        return False, -1, float('nan'), float('nan'), float('nan')

    # Sort deepest first
    candidates = candidates[np.argsort(window[min_lat_samp + candidates])]

    for offset in candidates[:2]:
        peak_local = min_lat_samp + int(offset)
        amp        = float(window[peak_local])
        lat_ms     = peak_local / sample_rate * 1000
        wid_ms     = _estimate_width(window, peak_local, sample_rate)

        if not np.isnan(wid_ms) and wid_ms < 5.0:
            return True, peak_local, amp, lat_ms, wid_ms

    return False, -1, float('nan'), float('nan'), float('nan')


def _estimate_width(
    window: np.ndarray,
    peak_local: int,
    sample_rate: int,
) -> float:
    """Estimate spike width at half-maximum in ms."""
    try:
        widths = scipy.signal.peak_widths(-window, [peak_local], rel_height=0.5)
        w = float(widths[3][0] - widths[2][0])
        return w / sample_rate * 1000 if w > 0 else float('nan')
    except Exception:
        return float('nan')


def _empty_direct_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'channel', 'pulse_index', 'amplitude_mV', 'latency_ms', 'width_ms',
        'corr', 'imbalance', 'rms_before', 'rms_after',
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