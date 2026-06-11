"""
intact_instability_localization.py
====================================
Loads ALL intact direct-response CSV files, groups them by retina,
and produces one 1×5 circular probe figure per retina.

Each probe plot shows normalised amplitude (y − y₀) / y₀ on the 16-channel
ring probe at a fixed pulse index (10, 20, 30, 40, 50).  y₀ is the amplitude
at START_PULSE.  When the same channel appears in multiple files for a retina,
values are averaged.  The stimulated electrode is marked with a red dot.
"""

import os
import re
import glob
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable

# ── parameters ─────────────────────────────────────────────────────────────────
START_PULSE = 2
PULSE_LIMIT = 50
HIST_PULSES = [10, 20, 30, 40, 50]

# ── paths ──────────────────────────────────────────────────────────────────────
INTACT_DIRS = {
     1: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_data\1hz',
    10: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_data\10hz',
    20: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_data\20hz',
}

# ── probe layout: real channel → angle in degrees CCW from 3 o'clock ──────────
CHANNEL_ANGLES: dict[int, float] = {
    26:   0,  5:  19, 25:  38,  6:  57, 24:  76,  7:  95,
    28: 114,  2: 133, 29: 152,  1: 171, 30: 189,  0: 208,
    31: 227,
    # 246°, 265°, 284° are empty slots
     3: 303, 27: 322,  4: 341,
}

# ── retina overrides: filename → (retina_prefix, date_str) ────────────────────
INTACT_RETINA_OVERRIDES: dict[str, tuple[str, str]] = {
    'Ch01_300us_50us_7uA_20Hz_250528_113309.rhs_direct_response.csv': ('Retina3', '250528'),
}

# ── probe display geometry ─────────────────────────────────────────────────────
_ELEC_RADIUS = 0.18   # electrode circle radius in data coordinates
_PROBE_RING  = 1.22   # outer boundary ring radius
_AXIS_LIM    = 1.55   # half-extent of each subplot axis

_cmap_hm   = plt.get_cmap('RdBu_r').copy()
_cmap_amp  = plt.get_cmap('viridis').copy()
_cmap_norm = plt.get_cmap('RdBu_r').copy()

_SCATTER_FREQ_HZ = {10, 20}


# ── helpers ────────────────────────────────────────────────────────────────────

def _stim_ch_from_fname(filename: str) -> Optional[int]:
    m = re.search(r'Ch(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _retina_key_from_fname(filename: str) -> Optional[tuple[str, str]]:
    if filename in INTACT_RETINA_OVERRIDES:
        return INTACT_RETINA_OVERRIDES[filename]
    m_r = re.search(r'(Retina\d+)', filename, re.IGNORECASE)
    m_d = re.search(r'_(\d{6})_', filename)
    if not m_r or not m_d:
        return None
    return m_r.group(1), m_d.group(1)


def _file_freq_hz(filename: str) -> Optional[float]:
    m = re.search(r'(\d+(?:\.\d+)?)Hz', filename, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _load_all_channels(csv_path: str) -> list[dict]:
    """Return one record per probe channel restricted to the analysis window."""
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    df['channel'] = pd.to_numeric(df['channel'], errors='coerce')
    df = df.dropna(subset=['channel'])

    records = []
    for ch, ch_df in df.groupby('channel'):
        ch_int = int(ch)
        if ch_int not in CHANNEL_ANGLES:
            continue
        total_pulses = int(ch_df['pulse_index'].max()) + 1
        pulses = (
            ch_df[
                (ch_df['pulse_index'] >= START_PULSE) &
                (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
            ][['pulse_index', 'amplitude_mV']]
            .copy().reset_index(drop=True)
        )
        if not pulses.empty:
            records.append({'channel': ch_int, 'pulses': pulses,
                            'total_pulses': total_pulses})
    return records


def _normalise(pulses: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return DataFrame[pulse_index, norm_amp] = (amp − y0) / y0."""
    per_pulse = (
        pulses.groupby('pulse_index')['amplitude_mV']
        .max()
        .reindex(range(START_PULSE, START_PULSE + PULSE_LIMIT), fill_value=0)
        .reset_index()
    )
    per_pulse.columns = ['pulse_index', 'amplitude_mV']
    y0 = float(per_pulse.loc[per_pulse['pulse_index'] == START_PULSE, 'amplitude_mV'].iloc[0])
    if y0 == 0:
        return None
    per_pulse['norm_amp'] = (per_pulse['amplitude_mV'] - y0) / y0
    return per_pulse[['pulse_index', 'norm_amp']]


def _value_at_pulse(pulses: pd.DataFrame, pulse_idx: int,
                    total_pulses: int = 9999) -> Optional[float]:
    if pulse_idx >= total_pulses:
        return None
    norm_df = _normalise(pulses)
    if norm_df is None:
        return None
    row = norm_df[norm_df['pulse_index'] == pulse_idx]
    if row.empty:
        return None
    return float(row['norm_amp'].iloc[0])


def _draw_probe(ax, ch_values: dict[int, float], stim_ch: Optional[int],
                cmap, norm_obj: mcolors.Normalize) -> None:
    """
    Draw the 16-channel ring probe on ax.
    ch_values maps real channel number → value (NaN = not recorded).
    """
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta) * _PROBE_RING, np.sin(theta) * _PROBE_RING,
            color='#aaaaaa', linewidth=0.8, zorder=1)

    for ch, angle_deg in CHANNEL_ANGLES.items():
        angle_rad = np.deg2rad(angle_deg)
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        val = ch_values.get(ch, np.nan)
        color = cmap(norm_obj(val)) if np.isfinite(val) else '#e8e8e8'
        ax.add_patch(Circle((x, y), radius=_ELEC_RADIUS, color=color,
                             ec='white', lw=0.5, zorder=3))

    if stim_ch is not None and stim_ch in CHANNEL_ANGLES:
        angle_rad = np.deg2rad(CHANNEL_ANGLES[stim_ch])
        ax.add_patch(Circle(
            (np.cos(angle_rad), np.sin(angle_rad)),
            radius=0.07, color='red', ec='white', lw=0.6, zorder=5,
        ))

    ax.set_xlim(-_AXIS_LIM, _AXIS_LIM)
    ax.set_ylim(-_AXIS_LIM, _AXIS_LIM)
    ax.set_aspect('equal')
    ax.axis('off')


# ── load and group all CSVs by retina ─────────────────────────────────────────

retina_records: dict[str, list[dict]] = defaultdict(list)
retina_stim:   dict[str, Optional[int]] = {}

for freq_hz, src_dir in INTACT_DIRS.items():
    for path in sorted(glob.glob(os.path.join(src_dir, '*_direct_response.csv'))):
        fname = os.path.basename(path)
        key   = _retina_key_from_fname(fname)
        if key is None:
            continue
        retina = f'{key[0]}_{key[1]}'
        if retina not in retina_stim:
            retina_stim[retina] = _stim_ch_from_fname(fname)
        file_recs = _load_all_channels(path)
        if file_recs:
            pulse_counts = {r['channel']: r['total_pulses'] for r in file_recs}
            counts_str = '  '.join(f'ch{ch}:{n}' for ch, n in sorted(pulse_counts.items()))
            print(f'  {fname}  →  {counts_str}')
        for rec in file_recs:
            retina_records[retina].append({
                'channel':      rec['channel'],
                'pulses':       rec['pulses'],
                'total_pulses': rec['total_pulses'],
                'filename':     fname,
            })

for retina, recs in sorted(retina_records.items()):
    n_ch = len({r['channel'] for r in recs})
    print(f'{retina}: {len(recs)} channel×file records, {n_ch} unique channels')


# ── figure 1: combined probe figure — n_retinas rows × n_pulses cols ──────────

_hm_retinas = sorted(retina_records.keys())
_n_hm       = len(_hm_retinas)
_n_pulses   = len(HIST_PULSES)

fig_hm, axes_hm = plt.subplots(_n_hm, _n_pulses,
                                figsize=(3.2 * _n_pulses, 3.2 * _n_hm))

for row_i, retina in enumerate(_hm_retinas):
    records  = retina_records[retina]
    stim_ch  = retina_stim.get(retina)

    channels = sorted({r['channel'] for r in records})
    ch_pulse_vals: dict[int, dict[int, list[float]]] = {
        ch: defaultdict(list) for ch in channels
    }
    for rec in records:
        ch = rec['channel']
        for p in HIST_PULSES:
            v = _value_at_pulse(rec['pulses'], p, rec['total_pulses'])
            if v is not None and np.isfinite(v):
                ch_pulse_vals[ch][p].append(v)

    ch_pulse_mean: dict[int, dict[int, float]] = {
        ch: {p: float(np.mean(vals)) for p, vals in pdict.items() if vals}
        for ch, pdict in ch_pulse_vals.items()
    }

    all_finite = [
        v for pdict in ch_pulse_mean.values()
        for v in pdict.values() if np.isfinite(v)
    ]
    vmax = min(max(abs(min(all_finite)), abs(max(all_finite))), 1.0) if all_finite else 1.0
    norm_obj = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    for col_i, pulse_idx in enumerate(HIST_PULSES):
        ax = axes_hm[row_i, col_i]
        ch_vals = {ch: pdict.get(pulse_idx, np.nan)
                   for ch, pdict in ch_pulse_mean.items()}
        _draw_probe(ax, ch_vals, stim_ch, _cmap_hm, norm_obj)

        sm = ScalarMappable(cmap=_cmap_hm, norm=norm_obj)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=5)

        if row_i == 0:
            ax.set_title(f'Pulse {pulse_idx}', fontsize=9, pad=4)

    stim_label = f'stim: Ch{stim_ch:02d}' if stim_ch is not None else 'stim: unknown'
    axes_hm[row_i, 0].text(-0.12, 0.5, f'{retina}\n({stim_label})',
                            fontsize=7, va='center', ha='right',
                            transform=axes_hm[row_i, 0].transAxes)

fig_hm.suptitle(
    f'Instability localisation  (y − y₀) / y₀  ·  y₀ = pulse {START_PULSE}  ·  red dot = stim',
    fontsize=10,
)
plt.tight_layout()
plt.show()
plt.close(fig_hm)


# ── figure 2: summary — row 0 = first amp, row 1 = pulse-50 change ────────────

_LAST_PULSE = HIST_PULSES[-1]   # = 50
_retinas    = sorted(retina_records.keys())
_n_retinas  = len(_retinas)

fig_sum, axes_sum = plt.subplots(3, _n_retinas,
                                 figsize=(3.2 * _n_retinas, 9.6))

_is_last = lambda c: c == _n_retinas - 1

for col_i, retina in enumerate(_retinas):
    records = retina_records[retina]
    stim_ch = retina_stim.get(retina)

    # ── row 0: avg amplitude, min-max normalised per file, averaged ──────────────
    _ch_avg: dict[int, list[float]] = defaultdict(list)
    _by_file_avg: dict[str, list] = defaultdict(list)
    for r in records:
        _by_file_avg[r['filename']].append(r)
    for _recs_in_file in _by_file_avg.values():
        _file_avgs: dict[int, float] = {}
        for r in _recs_in_file:
            v = float(r['pulses']['amplitude_mV'].mean())
            if np.isfinite(v) and v > 0:
                _file_avgs[r['channel']] = v
        if not _file_avgs:
            continue
        _fmin   = min(_file_avgs.values())
        _frange = max(_file_avgs.values()) - _fmin or 1.0
        for ch, v in _file_avgs.items():
            _ch_avg[ch].append((v - _fmin) / _frange)

    ch_avg_mean: dict[int, float] = {
        ch: float(np.mean(vals)) for ch, vals in _ch_avg.items()
    }

    ax0 = axes_sum[0, col_i]
    norm_avg = mcolors.Normalize(vmin=0, vmax=1)
    _draw_probe(ax0, ch_avg_mean, stim_ch, _cmap_amp, norm_avg)
    ax0.set_title(retina, fontsize=8, pad=3)
    if _is_last(col_i):
        sm0 = ScalarMappable(cmap=_cmap_amp, norm=norm_avg)
        sm0.set_array([])
        cbar0 = plt.colorbar(sm0, ax=ax0, shrink=0.75, pad=0.02)
        cbar0.set_label('norm. avg amp', fontsize=5)
        cbar0.ax.tick_params(labelsize=5)

    # ── row 1: first-pulse amplitude, min-max normalised per file, averaged ─────
    _ch_first: dict[int, list[float]] = defaultdict(list)
    _by_file: dict[str, list] = defaultdict(list)
    for r in records:
        _by_file[r['filename']].append(r)
    for _recs_in_file in _by_file.values():
        _file_amps: dict[int, float] = {}
        for r in _recs_in_file:
            row = r['pulses'][r['pulses']['pulse_index'] == START_PULSE]
            if row.empty:
                continue
            v = float(row['amplitude_mV'].max())
            if np.isfinite(v) and v > 0:
                _file_amps[r['channel']] = v
        if not _file_amps:
            continue
        _fmin   = min(_file_amps.values())
        _frange = max(_file_amps.values()) - _fmin or 1.0
        for ch, v in _file_amps.items():
            _ch_first[ch].append((v - _fmin) / _frange)

    ch_first_mean: dict[int, float] = {
        ch: float(np.mean(vals)) for ch, vals in _ch_first.items()
    }

    ax1 = axes_sum[1, col_i]
    norm_amp = mcolors.Normalize(vmin=0, vmax=1)
    _draw_probe(ax1, ch_first_mean, stim_ch, _cmap_amp, norm_amp)
    if _is_last(col_i):
        sm1 = ScalarMappable(cmap=_cmap_amp, norm=norm_amp)
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, ax=ax1, shrink=0.75, pad=0.02)
        cbar1.set_label('norm. amplitude', fontsize=5)
        cbar1.ax.tick_params(labelsize=5)

    # ── row 2: pulse-50  (y − y₀) / y₀, averaged per channel ─────────────────
    _ch_p50: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        v = _value_at_pulse(rec['pulses'], _LAST_PULSE, rec['total_pulses'])
        if v is not None and np.isfinite(v):
            _ch_p50[rec['channel']].append(v)

    ch_p50_mean: dict[int, float] = {
        ch: float(np.mean(vals)) for ch, vals in _ch_p50.items()
    }

    _finite_p50 = list(ch_p50_mean.values())
    _vmax_p50 = (min(max(abs(min(_finite_p50)), abs(max(_finite_p50))), 1.0)
                 if _finite_p50 else 1.0)
    norm_p50 = mcolors.Normalize(vmin=-_vmax_p50, vmax=_vmax_p50)

    ax2 = axes_sum[2, col_i]
    _draw_probe(ax2, ch_p50_mean, stim_ch, _cmap_norm, norm_p50)
    if _is_last(col_i):
        sm2 = ScalarMappable(cmap=_cmap_norm, norm=norm_p50)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2, shrink=0.75, pad=0.02)
        cbar2.set_label('(y − y₀) / y₀', fontsize=5)
        cbar2.ax.tick_params(labelsize=5)

axes_sum[0, 0].text(-0.12, 0.5, 'avg amp\n(min-max)',
                    fontsize=7, va='center', ha='right',
                    transform=axes_sum[0, 0].transAxes)
axes_sum[1, 0].text(-0.12, 0.5, 'first amp\n(min-max)',
                    fontsize=7, va='center', ha='right',
                    transform=axes_sum[1, 0].transAxes)
axes_sum[2, 0].text(-0.12, 0.5, f'pulse {_LAST_PULSE}\n(y − y₀) / y₀',
                    fontsize=7, va='center', ha='right',
                    transform=axes_sum[2, 0].transAxes)

fig_sum.suptitle(
    f'Instability localisation — avg amp / first amp (min-max) / pulse-{_LAST_PULSE} change\n'
    f'y₀ = pulse {START_PULSE}  ·  red dot = stim electrode',
    fontsize=10,
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.show()
plt.close(fig_sum)


# ── figure 3: scatter — first amp (norm) vs pulse-50 norm amp, per retina ──────

_lat_cmap   = plt.get_cmap('tab10')
_n_cols_sc  = _n_retinas + 2   # retina subplots + all-fits + mean±std
_x_shared   = np.linspace(0, 1, 200)

fig_sc, axes_sc = plt.subplots(1, _n_cols_sc,
                                figsize=(3.5 * _n_cols_sc, 4),
                                sharey=True)

_all_fits: list[np.ndarray] = []

for col_i, retina in enumerate(_retinas):
    records = [r for r in retina_records[retina]
               if _file_freq_hz(r['filename']) in _SCATTER_FREQ_HZ]
    ax      = axes_sc[col_i]
    color   = _lat_cmap(col_i % 10)

    # x: first-amp normalised within each file, averaged per channel
    _ch_first_sc: dict[int, list[float]] = defaultdict(list)
    _by_file_sc: dict[str, list] = defaultdict(list)
    for r in records:
        _by_file_sc[r['filename']].append(r)
    for _recs_in_file in _by_file_sc.values():
        _file_amps: dict[int, float] = {}
        for r in _recs_in_file:
            _row = r['pulses'][r['pulses']['pulse_index'] == START_PULSE]
            if _row.empty:
                continue
            v = float(_row['amplitude_mV'].max())
            if np.isfinite(v) and v > 0:
                _file_amps[r['channel']] = v
        if not _file_amps:
            continue
        _fmin   = min(_file_amps.values())
        _frange = max(_file_amps.values()) - _fmin or 1.0
        for ch, v in _file_amps.items():
            _ch_first_sc[ch].append((v - _fmin) / _frange)

    _ch_first_mean: dict[int, float] = {
        ch: float(np.mean(vals)) for ch, vals in _ch_first_sc.items()
    }

    # y: pulse-50 norm amp averaged per channel
    _ch_p50_sc: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        v = _value_at_pulse(rec['pulses'], _LAST_PULSE, rec['total_pulses'])
        if v is not None and np.isfinite(v):
            _ch_p50_sc[rec['channel']].append(v)

    _ch_p50_mean: dict[int, float] = {
        ch: float(np.mean(vals)) for ch, vals in _ch_p50_sc.items()
    }

    _channels_sc = set(_ch_first_mean) & set(_ch_p50_mean)
    xs = np.array([_ch_first_mean[ch] for ch in _channels_sc])
    ys = np.array([_ch_p50_mean[ch]   for ch in _channels_sc])
    ax.scatter(xs, ys, color=color, s=22, alpha=0.75, linewidths=0, zorder=3)

    if len(xs) >= 2:
        _m, _b = np.polyfit(xs, ys, 1)
        _y_fit = _m * _x_shared + _b
        _all_fits.append(_y_fit)
        ax.plot(_x_shared, _y_fit, color='black', linewidth=1.5, zorder=4)
        ax.text(0.97, 0.97, f'y={_m:+.2f}x{_b:+.2f}',
                transform=ax.transAxes, fontsize=6,
                va='top', ha='right', family='monospace')

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
    ax.set_title(retina, fontsize=8, loc='left', pad=3)
    ax.set_xlabel('first amp (norm.)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes_sc[0].set_ylabel(f'(y − y₀) / y₀  at pulse {_LAST_PULSE}', fontsize=7)
# axes_sc[0].set_ylim(0, 1)

# ── all fits overlaid ──────────────────────────────────────────────────────────
ax_all = axes_sc[_n_retinas]
for col_i, (retina, _y_fit) in enumerate(zip(_retinas, _all_fits)):
    ax_all.plot(_x_shared, _y_fit, color=_lat_cmap(col_i % 10),
                linewidth=1.5, label=retina, zorder=3)
ax_all.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax_all.set_title('all fits', fontsize=8, loc='left', pad=3)
ax_all.set_xlabel('first amp (norm.)', fontsize=7)
ax_all.tick_params(labelsize=6)
ax_all.legend(fontsize=5, loc='upper right', framealpha=0.6,
              handlelength=1.2, labelspacing=0.3)
ax_all.spines['top'].set_visible(False)
ax_all.spines['right'].set_visible(False)

# ── mean ± std across fits ─────────────────────────────────────────────────────
ax_avg = axes_sc[_n_retinas + 1]
if _all_fits:
    _fits_arr = np.array(_all_fits)
    _fit_mean = _fits_arr.mean(axis=0)
    _fit_std  = _fits_arr.std(axis=0)
    ax_avg.plot(_x_shared, _fit_mean, color='black', linewidth=1.5, zorder=3)
    ax_avg.fill_between(_x_shared,
                        _fit_mean - _fit_std,
                        _fit_mean + _fit_std,
                        color='black', alpha=0.15, zorder=2)
ax_avg.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax_avg.set_title('mean ± std', fontsize=8, loc='left', pad=3)
ax_avg.set_xlabel('first amp (norm.)', fontsize=7)
ax_avg.tick_params(labelsize=6)
ax_avg.spines['top'].set_visible(False)
ax_avg.spines['right'].set_visible(False)

fig_sc.suptitle(
    f'First amplitude (min-max norm., avg across files) vs pulse-{_LAST_PULSE} change'
    f'  ·  {" & ".join(str(f) for f in sorted(_SCATTER_FREQ_HZ))} Hz only',
    fontsize=9,
)
plt.tight_layout()
plt.show()
plt.close(fig_sc)