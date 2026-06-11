"""
0309_ex_vivo_compare_all.py
============================
Compares direct-response properties when stimulating with the MEA and
recording simultaneously with both the MEA and the probe16.

Files
-----
RHS (probe recording) : ChE11_20uA_300us_50us_1Hz_250309_172137.rhs
EDF (MEA  recording)  : id1 ChE11_20uA_300us_50us_1Hz_100pulsesB-00071.edf

Pipeline
--------
1. Load & detect direct responses in both files (with diagnostic plots).
2. Interactive channel selection — MEA: click on grid; Probe: type names.
3. Histogram comparison figure: amplitude | latency | width.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataobj import load_rhs, load_edf
from dataviz.viz import plot_spikes_layout_probe16

CACHE_DIR = os.path.join(os.path.dirname(__file__), '0309_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def _load_or_compute_dr(cache_name, compute_fn):
    """Return cached DataFrame if CSV exists, otherwise run compute_fn() and save."""
    path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(path):
        print(f'  [cache] loading {cache_name}')
        return pd.read_csv(path)
    df = compute_fn()
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        print(f'  [cache] saved {cache_name}')
    return df

# ── files ──────────────────────────────────────────────────────────────────────
RHS_MEA_STIM_FILE = (r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14'
                     r'\For figures\Stim with MEA\New folder'
                     r'\ChE11_20uA_300us_50us_1Hz_250309_172137.rhs')

RHS_PROBE_STIM_FILE = (r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14'
                       r'\Ch27_7uA_300us_50us_1Hz_256pulses_250309_190326.rhs')

RHS_PROBE_STIM_10HZ_FILE = (r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14'
                             r'\Ch27_7uA_300us_50us_10Hz_256pulses_250309_185941.rhs')

RHS_PROBE_STIM_100HZ_FILE = (r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14'
                              r'\Ch27_7uA_300us_50us_100Hz_256pulses_250309_191143.rhs')

EDF_MEA_STIM_FILE = (r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14'
                     r'\For figures\Stim with MEA\New folder'
                     r'\id1 ChE11_20uA_300us_50us_1Hz_100pulsesB-00071.edf')

EDF_STIM_ELECTRODE = 'E11'

EDF_MEA_STIM_FILE_2 = (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal'
                       r'\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf')
EDF_STIM_ELECTRODE_2 = 'J6'
EDF_SPEED_RECORD_CH      = 'F11'   # recording channel to extract for speed figure
PROBE_SPEED_RECORD_CH    = '0'     # recording channel to extract for speed figure

MEA_DISTANCE_MM   = 1.28   # J6 → E11 distance on MEA
PROBE_DISTANCE_MM = 0.95   # ch27 → ch0 distance on probe

# ── analysis params ─────────────────────────────────────────────────────────────
DIRECT_WIN_MS      = 10.0
BLANK_MS           = 2.5
EDF_THRESHOLD_MV   = 0.3
RHS_STIM_THRESHOLD = 470
RHS_THRESHOLD_MV   = 15


# ── interactive MEA channel selection (click-on-grid) ─────────────────────────

def _select_channels_mea(rec, win_size_ms, blank_ms, threshold):
    """Show 12×12 MEA grid; click subplots to toggle selection. Close to confirm."""
    data         = rec.blanked_data
    sr           = rec.sample_rate
    stim_indices = rec.stim_indices
    locations    = rec.channel_locations
    win_samples  = int(win_size_ms / 1000 * sr)

    pos_to_ch = {}
    for idx, name in enumerate(rec.channel_names):
        loc = locations[idx] if idx < len(locations) else None
        if loc is None:
            continue
        r, c = int(loc[0]), int(loc[1])
        if 0 <= r < 12 and 0 <= c < 12:
            pos_to_ch[(r, c)] = (idx, name)

    stim_pos = None
    if rec.stim_channel_name:
        for idx, name in enumerate(rec.channel_names):
            if rec.stim_channel_name in name:
                loc = locations[idx] if idx < len(locations) else None
                if loc is not None:
                    stim_pos = (int(loc[0]), int(loc[1]))
                break

    selected  = set()
    fig, axes = plt.subplots(12, 12, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    ax_to_ch  = {}
    used      = set()

    for (r, c), (idx, name) in pos_to_ch.items():
        used.add((r, c))
        ax       = axes[r, c]
        ax_to_ch[ax] = name
        ch_data  = data[idx]
        for si in stim_indices:
            start  = int(si)
            end    = min(start + win_samples, len(ch_data))
            window = ch_data[start:end]
            if len(window) == 0:
                continue
            t = np.arange(len(window)) / sr * 1000
            ax.plot(t, window, color='black', linewidth=0.2)
        if threshold is not None:
            ax.axhline(-threshold, color='grey', linewidth=0.5, linestyle='--')
        ax.axvline(blank_ms, color='grey', linewidth=0.5, linestyle='--')
        short = name.split()[-1].upper() if ' ' in name else name
        ax.text(0.5, 0.95, short, transform=ax.transAxes,
                fontsize=6, va='top', ha='center', color='dimgrey')
        if stim_pos and (r, c) == stim_pos:
            ax.plot(win_size_ms / 2, 0, 'o', color='red', markersize=4, zorder=5)
        ax.set_xlim(0, win_size_ms)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for side, spine in ax.spines.items():
            spine.set_visible(side in ('bottom', 'left'))

    for r in range(12):
        for c in range(12):
            if (r, c) not in used:
                axes[r, c].set_visible(False)

    def _on_click(event):
        if event.inaxes is None:
            return
        name = ax_to_ch.get(event.inaxes)
        if name is None:
            return
        if name in selected:
            selected.discard(name)
            event.inaxes.set_facecolor('white')
        else:
            selected.add(name)
            event.inaxes.set_facecolor('lightpink')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.suptitle('MEA — click channels to select (pink = selected). Close to confirm.',
                 fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show(block=False)
    print('MEA channel selection open — close window when done.')
    while plt.fignum_exists(fig.number):
        plt.pause(0.05)

    if not selected:
        print('No channels selected — using all.')
        return list(rec.channel_names)
    print(f'Selected MEA channels: {sorted(selected)}')
    return list(selected)


# ── interactive probe channel selection (type names) ───────────────────────────

def _select_channels_probe(rec, win_size_ms, threshold):
    """Show probe layout (catching the ymin bug), then ask for channel names."""
    plot_spikes_layout_probe16(rec=rec, win_size_ms=win_size_ms,
                               data_type='blanked', threshold=threshold,
                               output_folder=None)
    plt.tight_layout()
    plt.show(block=True)

    available = sorted(rec.channel_names)
    print(f'Available probe channels: {" ".join(available)}')
    raw = input('Channel names to include (space-separated), or Enter for all: ').strip()
    if not raw:
        print('Using all channels.')
        return list(rec.channel_names)
    typed    = {n for n in raw.split()}
    selected = [ch for ch in rec.channel_names if ch in typed]
    unknown  = typed - set(rec.channel_names)
    if unknown:
        print(f'Unknown channels ignored: {", ".join(sorted(unknown))}')
    print(f'Selected probe channels: {selected}')
    return selected


# ── helpers ────────────────────────────────────────────────────────────────────

def _normalized_amp_vs_pulse(dr):
    """Return (pulse_indices, mean_norm, std_norm) of (amp-amp0)/amp0 across channels.
    Pulses with no detection are filled with 0 before normalising."""
    full_range = np.arange(int(dr['pulse_index'].max()) + 1)
    records = []
    for _, grp in dr.groupby('channel'):
        grp_indexed = grp.set_index('pulse_index')['amplitude_mV']
        amps = np.abs(grp_indexed.reindex(full_range, fill_value=0).values)
        if len(amps) == 0 or amps[0] == 0:
            continue
        norm = (amps - amps[0]) / amps[0]
        for pidx, n in zip(full_range, norm):
            records.append({'pulse_index': int(pidx), 'norm_amp': n})
    if not records:
        return None, None, None
    stats = pd.DataFrame(records).groupby('pulse_index')['norm_amp'].agg(['mean', 'std']).reset_index()
    return stats['pulse_index'].values, stats['mean'].values, stats['std'].values


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── MEA stim, MEA record ──────────────────────────────────────────────────
    print('=' * 60)
    mea_channels = [
        'F2_B-00071 A9',  'F2_B-00071 B10', 'F2_B-00071 B9',
        'F2_B-00071 C10', 'F2_B-00071 C9',  'F2_B-00071 D11',
        'F2_B-00071 G12', 'F2_B-00071 H12', 'F2_B-00071 J12',
    ]
    def _compute_mea_dr():
        print('Loading MEA stim, MEA record (EDF) …')
        rec = load_edf(EDF_MEA_STIM_FILE, stim_electrode=EDF_STIM_ELECTRODE)
        rec.filter()
        rec.blank(duration_ms=BLANK_MS, source='filtered_data')
        rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                   threshold=EDF_THRESHOLD_MV,
                                   plot=True)
        dr = rec.direct_response
        if dr is not None and not dr.empty:
            dr = dr[dr['channel'].isin(mea_channels)].reset_index(drop=True)
        return dr
    mea_dr = _load_or_compute_dr('mea_stim_mea_rec_dr.csv', _compute_mea_dr)

    # ── MEA stim, Probe record ────────────────────────────────────────────────
    print('=' * 60)
    def _compute_probe_mea_stim_dr():
        print('Loading MEA stim, Probe record (RHS) …')
        rec = load_rhs(RHS_MEA_STIM_FILE, stim_threshold=RHS_STIM_THRESHOLD)
        rec.blank(duration_ms=BLANK_MS, source='raw_data')
        rec.detect_direct_response(win_size_ms=30,
                                   threshold=RHS_THRESHOLD_MV,
                                   data_type='blanked')
        plot_spikes_layout_probe16(rec=rec, win_size_ms=20,
                                   data_type='blanked', threshold=15)
        return rec.direct_response
    probe_mea_stim_dr = _load_or_compute_dr('mea_stim_probe_rec_dr.csv', _compute_probe_mea_stim_dr)

    # ── Probe stim, Probe record ──────────────────────────────────────────────
    print('=' * 60)
    probe_stim_channels = ['0', '1', '2', '30', '31']
    def _compute_probe_probe_stim_dr():
        print('Loading Probe stim, Probe record (RHS) …')
        rec = load_rhs(RHS_PROBE_STIM_FILE, stim_threshold=RHS_STIM_THRESHOLD)
        rec.blank(duration_ms=BLANK_MS, source='raw_data')
        rec.detect_direct_response(win_size_ms=30,
                                   threshold=RHS_THRESHOLD_MV,
                                   data_type='blanked')
        plot_spikes_layout_probe16(rec=rec, win_size_ms=20,
                                   data_type='blanked', threshold=15)
        dr = rec.direct_response
        if dr is not None and not dr.empty:
            dr = dr[dr['channel'].isin(probe_stim_channels)].reset_index(drop=True)
        return dr
    probe_probe_stim_dr = _load_or_compute_dr('probe_stim_probe_rec_dr.csv', _compute_probe_probe_stim_dr)

    # ── MEA stim, MEA record (file 2, for speed figure) ──────────────────────
    print('=' * 60)
    def _compute_mea_dr2():
        print('Loading MEA stim, MEA record file 2 (EDF) …')
        rec = load_edf(EDF_MEA_STIM_FILE_2, stim_electrode=EDF_STIM_ELECTRODE_2)
        rec.filter()
        rec.blank(duration_ms=BLANK_MS, source='filtered_data')
        rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                   threshold=0.1,
                                   plot=True)
        dr = rec.direct_response
        if dr is not None and not dr.empty:
            dr = dr[dr['channel'].str.endswith(EDF_SPEED_RECORD_CH)].reset_index(drop=True)
        return dr
    mea_dr2 = _load_or_compute_dr('mea_stim_mea_rec_dr2.csv', _compute_mea_dr2)

    # single-channel slices for speed figure
    mea_speed_dr   = mea_dr2[mea_dr2['channel'].str.endswith(EDF_SPEED_RECORD_CH)].reset_index(drop=True) \
                     if mea_dr2 is not None and not mea_dr2.empty else None
    probe_speed_dr = probe_probe_stim_dr[
                         probe_probe_stim_dr['channel'].astype(str) == PROBE_SPEED_RECORD_CH
                     ].reset_index(drop=True) \
                     if probe_probe_stim_dr is not None and not probe_probe_stim_dr.empty else None

    # ── Figure 1: spatial layout — Probe (left) and MEA (right) ──────────────
    from dataobj.channel_utils import mea_name_to_location
    from matplotlib.patches import Circle
    import matplotlib.colors as mcolors

    _CHANNEL_ANGLES: dict[int, float] = {
        26:   0,  5:  19, 25:  38,  6:  57, 24:  76,  7:  95,
        28: 114,  2: 133, 29: 152,  1: 171, 30: 189,  0: 208,
        31: 227,  3: 303, 27: 322,  4: 341,
    }
    _ELEC_RADIUS = 0.18
    _PROBE_RING  = 1.22
    _AXIS_LIM    = 1.55

    def _ch_mean_amp(df):
        """Return dict channel → mean |amplitude_mV| across all pulses."""
        if df is None or df.empty or 'amplitude_mV' not in df.columns:
            return {}
        result = {}
        for ch, grp in df.groupby('channel'):
            result[ch] = float(np.abs(grp['amplitude_mV']).mean())
        return result

    def _minmax(d):
        """Normalise dict values to [0, 1]."""
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        span = hi - lo or 1.0
        return {k: (v - lo) / span for k, v in d.items()}

    _grey = plt.get_cmap('Greys')
    _grey_bad = '#e8e8e8'

    # probe amplitude map (probe_mea_stim_dr uses int channels)
    _probe_amp_raw = _ch_mean_amp(probe_mea_stim_dr)
    _probe_amp_raw = {int(k): v for k, v in _probe_amp_raw.items()}
    _probe_amp = _minmax(_probe_amp_raw)

    # MEA amplitude map
    _mea_amp_raw = _ch_mean_amp(mea_dr)
    _mea_amp = _minmax(_mea_amp_raw)

    fig1, (ax_probe, ax_mea) = plt.subplots(1, 2, figsize=(8, 4))

    # ── left: probe ring layout ───────────────────────────────────────────────
    _ROT = 147  # degrees to rotate so ch3 sits at the top (90°)
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_probe.plot(np.cos(theta) * _PROBE_RING, np.sin(theta) * _PROBE_RING,
                  color='#bbbbbb', linewidth=0.8, zorder=1)
    for ch, angle_deg in _CHANNEL_ANGLES.items():
        angle_rad = np.deg2rad(angle_deg + _ROT)
        x, y = np.cos(angle_rad), np.sin(angle_rad)
        val   = _probe_amp.get(ch, np.nan)
        color = _grey(val) if np.isfinite(val) else _grey_bad
        ax_probe.add_patch(Circle((x, y), radius=_ELEC_RADIUS,
                                  color=color, ec='white', lw=0.5, zorder=3))
    ax_probe.set_xlim(-_AXIS_LIM, _AXIS_LIM)
    ax_probe.set_ylim(-_AXIS_LIM, _AXIS_LIM)
    ax_probe.set_aspect('equal')
    ax_probe.axis('off')

    # ── right: MEA 12×12 grid ────────────────────────────────────────────────
    _mea_grid = np.full((12, 12), np.nan)
    for ch_name, val in _mea_amp.items():
        short = str(ch_name).split()[-1] if ' ' in str(ch_name) else str(ch_name)
        loc = mea_name_to_location(short)
        if loc is not None:
            _mea_grid[loc[0], loc[1]] = val
    _cmap_mea = plt.get_cmap('Greys').copy()
    _cmap_mea.set_bad(_grey_bad)
    im = ax_mea.imshow(_mea_grid, cmap=_cmap_mea, aspect='equal',
                       origin='upper', vmin=0, vmax=1)
    # mark stim electrode
    _stim_loc = mea_name_to_location(EDF_STIM_ELECTRODE)
    if _stim_loc:
        ax_mea.plot(_stim_loc[1], _stim_loc[0], 'o', color='red',
                    markersize=6, zorder=5, markeredgewidth=0.8, markeredgecolor='white')
    ax_mea.set_xticks([])
    ax_mea.set_yticks([])
    for spine in ax_mea.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax_mea, shrink=0.7, pad=0.03).ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()
    plt.close(fig1)

    _mea_grey   = '#888888'
    _probe_grey = '#444444'

    # ── Probe stim 10 Hz ──────────────────────────────────────────────────────
    print('=' * 60)
    def _compute_probe_10hz_dr():
        print('Loading Probe stim 10 Hz (RHS) …')
        rec = load_rhs(RHS_PROBE_STIM_10HZ_FILE, stim_threshold=RHS_STIM_THRESHOLD)
        rec.blank(duration_ms=BLANK_MS, source='raw_data')
        rec.detect_direct_response(win_size_ms=30,
                                   threshold=RHS_THRESHOLD_MV,
                                   data_type='blanked')
        dr = rec.direct_response
        if dr is not None and not dr.empty:
            dr = dr[dr['channel'].isin(probe_stim_channels)].reset_index(drop=True)
        return dr
    probe_10hz_dr = _load_or_compute_dr('probe_stim_10hz_dr.csv', _compute_probe_10hz_dr)

    # ── Probe stim 100 Hz ─────────────────────────────────────────────────────
    print('=' * 60)
    def _compute_probe_100hz_dr():
        print('Loading Probe stim 100 Hz (RHS) …')
        rec = load_rhs(RHS_PROBE_STIM_100HZ_FILE, stim_threshold=RHS_STIM_THRESHOLD)
        rec.blank(duration_ms=BLANK_MS, source='raw_data')
        rec.detect_direct_response(win_size_ms=12,
                                   threshold=RHS_THRESHOLD_MV,
                                   data_type='blanked')

        dr = rec.direct_response
        if dr is not None and not dr.empty:
            dr = dr[dr['channel'].isin(probe_stim_channels)].reset_index(drop=True)
        return dr
    probe_100hz_dr = _load_or_compute_dr('probe_stim_100hz_dr.csv', _compute_probe_100hz_dr)

    # ── Figure 2+3 combined: row 0 = single-channel comparison,
    #                         row 1 = frequency amplitude distribution ──────────
    import matplotlib.gridspec as gridspec

    fig_combined = plt.figure(figsize=(10, 7))
    outer_gs = gridspec.GridSpec(2, 1, figure=fig_combined, hspace=0.45)

    # ── Row 0: amplitude, width, speed histograms ─────────────────────────────
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.35)
    axes2  = [fig_combined.add_subplot(gs_top[i]) for i in range(3)]

    speed_metrics = [
        ('amplitude_mV', 'amplitude (mV)', True,  None),
        ('width_ms',     'width (ms)',     False, None),
        ('latency_ms',   'velocity (m/s)', False, 'speed'),
    ]
    speed_datasets = [
        (mea_speed_dr,   f'MEA: {EDF_STIM_ELECTRODE_2}→{EDF_SPEED_RECORD_CH}',
         _mea_grey,   MEA_DISTANCE_MM),
        (probe_speed_dr, f'Probe: Ch27→Ch{PROBE_SPEED_RECORD_CH}',
         _probe_grey, PROBE_DISTANCE_MM),
    ]
    for ax, (col, xlabel, use_abs, derived) in zip(axes2, speed_metrics):
        for spd_df, label, color, dist_mm in speed_datasets:
            if spd_df is None or spd_df.empty or col not in spd_df.columns:
                continue
            vals = spd_df[col].dropna().values
            if use_abs:
                vals = np.abs(vals)
            if derived == 'speed':
                vals = dist_mm / vals
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue
            x_max   = np.percentile(vals, 99) * 1.1
            bins    = np.linspace(0, x_max, 30)
            weights = [100.0 / len(vals)] * len(vals)
            ax.hist(vals, bins=bins, weights=weights, color=color,
                    edgecolor='white', linewidth=0.3, alpha=0.7, label=label)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes2[0].set_ylabel('%', fontsize=8)
    axes2[0].legend(fontsize=7, frameon=False)

    # ── Row 1: frequency amplitude distribution + norm vs pulse ───────────────
    def _first100_ch0(freq_df):
        if freq_df is None or freq_df.empty:
            return freq_df
        return freq_df[(freq_df['pulse_index'] < 100) &
                       (freq_df['channel'].astype(str) == '0')].reset_index(drop=True)

    _freq_datasets = [
        (_first100_ch0(probe_probe_stim_dr), '1 Hz',   '#aaaaaa'),
        (_first100_ch0(probe_10hz_dr),       '10 Hz',  '#666666'),
        (_first100_ch0(probe_100hz_dr),      '100 Hz', '#222222'),
    ]

    _full_range = np.arange(100)
    _norm_arrays = {}
    for freq_df, label, color in _freq_datasets:
        if freq_df is None or freq_df.empty or 'amplitude_mV' not in freq_df.columns:
            _norm_arrays[label] = np.zeros(100)
            continue
        amps = np.abs(
            freq_df.set_index('pulse_index')['amplitude_mV']
            .reindex(_full_range, fill_value=0).values
        )
        norm = (amps - amps[0]) / amps[0] if amps[0] != 0 else np.zeros(100)
        _norm_arrays[label] = norm

    _all_norm    = np.concatenate(list(_norm_arrays.values()))
    _xmin        = np.percentile(_all_norm, 1)
    _xmax        = np.percentile(_all_norm, 99)
    _shared_bins = np.linspace(_xmin, _xmax, 30)

    gs_bot    = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer_gs[1],
                                                 wspace=0.35, hspace=0.15)
    ax_norm   = fig_combined.add_subplot(gs_bot[:, 1])
    hist_axes = [fig_combined.add_subplot(gs_bot[i, 0]) for i in range(3)]

    for (freq_df, label, color), ax_h in zip(_freq_datasets, hist_axes):
        norm_vals = _norm_arrays[label]
        weights   = [100.0 / len(norm_vals)] * len(norm_vals)
        ax_h.hist(norm_vals, bins=_shared_bins, weights=weights, color=color,
                  edgecolor='white', linewidth=0.3, alpha=0.8)
        ax_h.set_xlim(_xmin, _xmax)
        ax_h.set_ylabel('%', fontsize=7)
        ax_h.tick_params(labelsize=6)
        ax_h.spines['top'].set_visible(False)
        ax_h.spines['right'].set_visible(False)
        ax_h.text(0.89, 0.88, label, transform=ax_h.transAxes,
                  fontsize=7, ha='right', va='top', color=color)
        if ax_h is not hist_axes[-1]:
            ax_h.set_xticklabels([])
        ax_norm.plot(_full_range, norm_vals, color=color, linewidth=1.2, label=label)

    hist_axes[-1].set_xlabel('(y − y₀) / y₀', fontsize=8)

    ax_norm.set_xlabel('pulse #', fontsize=8)
    ax_norm.set_ylabel('(y − y₀) / y₀', fontsize=8)
    ax_norm.tick_params(labelsize=7)
    ax_norm.spines['top'].set_visible(False)
    ax_norm.spines['right'].set_visible(False)
    ax_norm.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    plt.show()
    plt.close(fig_combined)