"""
ex_vivo_instability_localization.py
============================
Loads ALL ex vivo direct-response CSV files, groups them by retina,
and produces one 1×5 heatmap figure per retina.

Each heatmap shows normalised amplitude (y − y₀) / y₀ on the 12×12 MEA grid
at a fixed pulse index (10, 20, 30, 40, 50).  y₀ is the amplitude at
START_PULSE.  When the same channel appears in multiple files for a retina,
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

from dataobj.channel_utils import mea_name_to_location

# ── parameters (same as combined_instability_analysis) ────────────────────────
START_PULSE = 2
PULSE_LIMIT = 50
HIST_PULSES = [10, 20, 30, 40, 50]

# ── paths ──────────────────────────────────────────────────────────────────────
EX_VIVO_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_all'

# ── stim electrode per retina (from main.py / ex_vivo_instability scripts) ────
STIM_ELECTRODE: dict[str, str] = {
    '2024.11.17':    'G6',
    '2025.11.02 J6': 'J6',
    '2025.11.02 J9': 'J9',
    '2025.11.12':    'G10',
}


# ── retina label (same logic as combined_instability_analysis) ─────────────────

def _retina_label(filename: str) -> str:
    if '2024-11-17' in filename:
        return '2024.11.17'
    if '2025-11-02' in filename:
        if 'J6' in filename:
            return '2025.11.02 J6'
        if 'J9' in filename:
            return '2025.11.02 J9'
        return '2025.11.02'
    if '2025-11-12' in filename:
        return '2025.11.12'
    return 'unknown'


# ── data loading ───────────────────────────────────────────────────────────────

def _load_all_channels(csv_path: str) -> list[dict]:
    """Return one record per channel restricted to the analysis window."""
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()

    keep_cols = ['pulse_index', 'amplitude_mV']
    if 'width_ms' in df.columns:
        keep_cols.append('width_ms')

    records = []
    for ch, ch_df in df.groupby('channel'):
        short_ch = str(ch).split()[-1].upper()
        pulses = (
            ch_df[
                (ch_df['pulse_index'] >= START_PULSE) &
                (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
            ][keep_cols]
            .copy()
            .reset_index(drop=True)
        )
        if not pulses.empty:
            records.append({'channel': short_ch, 'pulses': pulses})
    return records


# ── normalisation ──────────────────────────────────────────────────────────────

def _normalise(pulses: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return DataFrame[pulse_index, norm_amp] = (amp − y0) / y0.
    Missing pulses are filled with 0 before normalisation, so they become -1."""
    per_pulse = (
        pulses
        .groupby('pulse_index')['amplitude_mV']
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


def _value_at_pulse(pulses: pd.DataFrame, pulse_idx: int) -> Optional[float]:
    norm_df = _normalise(pulses)
    if norm_df is None:
        return None
    row = norm_df[norm_df['pulse_index'] == pulse_idx]
    if row.empty:
        return None
    return float(row['norm_amp'].iloc[0])


# ── load and group all CSVs by retina ─────────────────────────────────────────

retina_records: dict[str, list[dict]] = defaultdict(list)

for path in sorted(glob.glob(os.path.join(EX_VIVO_DIR, '*_direct_response.csv'))):
    fname  = os.path.basename(path)
    retina = _retina_label(fname)
    if retina == 'unknown':
        continue
    for rec in _load_all_channels(path):
        retina_records[retina].append({
            'channel': rec['channel'],
            'pulses':  rec['pulses'],
            'filename': fname,
        })

for retina, recs in sorted(retina_records.items()):
    n_ch = len({r['channel'] for r in recs})
    print(f'{retina}: {len(recs)} channel×file records, {n_ch} unique channels')


# ── combined heatmap figure: one row per retina, one col per pulse ─────────────

_hm_retinas = sorted(retina_records.keys())
_n_hm       = len(_hm_retinas)
_n_pulses   = len(HIST_PULSES)

_cmap_hm = plt.get_cmap('RdBu_r').copy()
_cmap_hm.set_bad(color='#e8e8e8')

fig_hm, axes_hm = plt.subplots(_n_hm, _n_pulses,
                                figsize=(3.0 * _n_pulses, 3.0 * _n_hm))

for row_i, retina in enumerate(_hm_retinas):
    records  = retina_records[retina]
    stim_ch  = STIM_ELECTRODE.get(retina)
    stim_pos = mea_name_to_location(stim_ch) if stim_ch else None

    channels = sorted({r['channel'] for r in records})

    ch_pulse_vals: dict[str, dict[int, list[float]]] = {
        ch: defaultdict(list) for ch in channels
    }
    for rec in records:
        ch = rec['channel']
        for p in HIST_PULSES:
            v = _value_at_pulse(rec['pulses'], p)
            if v is not None and np.isfinite(v):
                ch_pulse_vals[ch][p].append(v)

    ch_pulse_mean: dict[str, dict[int, float]] = {
        ch: {p: float(np.mean(vals)) for p, vals in pdict.items() if vals}
        for ch, pdict in ch_pulse_vals.items()
    }

    def _grid(pulse_idx: int) -> np.ndarray:
        g = np.full((12, 12), np.nan)
        for ch, pdict in ch_pulse_mean.items():
            loc = mea_name_to_location(ch)
            if loc is None:
                continue
            val = pdict.get(pulse_idx)
            if val is not None:
                g[loc[0], loc[1]] = val
        return g

    grids = {p: _grid(p) for p in HIST_PULSES}

    all_finite = [v for g in grids.values() for v in g.flatten() if np.isfinite(v)]
    vmax = min(max(abs(min(all_finite)), abs(max(all_finite))), 1.0) if all_finite else 1.0

    for col_i, pulse_idx in enumerate(HIST_PULSES):
        ax = axes_hm[row_i, col_i]
        im = ax.imshow(grids[pulse_idx], cmap=_cmap_hm, aspect='equal',
                       origin='upper', vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.03)
        cbar.ax.tick_params(labelsize=5)

        if stim_pos is not None:
            ax.plot(stim_pos[1], stim_pos[0], 'o', color='red', markersize=6,
                    zorder=5, markeredgewidth=0.8, markeredgecolor='white')

        if row_i == 0:
            ax.set_title(f'Pulse {pulse_idx}', fontsize=9, pad=4)

    stim_label = f'stim: {stim_ch}' if stim_ch else 'stim: unknown'
    axes_hm[row_i, 0].set_ylabel(f'{retina}\n({stim_label})', fontsize=7)

fig_hm.suptitle(
    f'Instability localisation  (y − y₀) / y₀  ·  y₀ = pulse {START_PULSE}  ·  red dot = stim',
    fontsize=10,
)
plt.tight_layout()
plt.show()
plt.close(fig_hm)


# ── summary figure: row0 = first-amp, row1 = pulse-50, one column per retina ──

_LAST_PULSE  = HIST_PULSES[-1]   # = 50
_retinas     = sorted(retina_records.keys())
_n_retinas   = len(_retinas)

_cmap_amp  = plt.get_cmap('viridis').copy()
_cmap_amp.set_bad(color='#e8e8e8')
_cmap_norm = plt.get_cmap('RdBu_r').copy()
_cmap_norm.set_bad(color='#e8e8e8')

fig_sum, axes_sum = plt.subplots(3, _n_retinas,
                                 figsize=(3.2 * _n_retinas, 9.6))

for col_i, retina in enumerate(_retinas):
    records  = retina_records[retina]
    stim_ch  = STIM_ELECTRODE.get(retina)
    stim_pos = mea_name_to_location(stim_ch) if stim_ch else None

    def _stim_dot(ax):
        if stim_pos is not None:
            ax.plot(stim_pos[1], stim_pos[0], 'o', color='red', markersize=6,
                    zorder=5, markeredgewidth=0.8, markeredgecolor='white')

    def _clean_ax(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── row 0: average amplitude, min-max normalised per file, averaged ──────────
    _ch_avg_amp: dict[str, list[float]] = defaultdict(list)
    _by_file_avg: dict[str, list] = defaultdict(list)
    for r in records:
        _by_file_avg[r['filename']].append(r)
    for _recs_in_file in _by_file_avg.values():
        _file_avgs: dict[str, float] = {}
        for r in _recs_in_file:
            v = float(r['pulses']['amplitude_mV'].mean())
            if np.isfinite(v) and v > 0:
                _file_avgs[r['channel']] = v
        if not _file_avgs:
            continue
        _fmin   = min(_file_avgs.values())
        _frange = max(_file_avgs.values()) - _fmin or 1.0
        for ch, v in _file_avgs.items():
            _ch_avg_amp[ch].append((v - _fmin) / _frange)

    grid_avg = np.full((12, 12), np.nan)
    for ch, vals in _ch_avg_amp.items():
        loc = mea_name_to_location(ch)
        if loc is not None:
            grid_avg[loc[0], loc[1]] = float(np.mean(vals))

    ax0 = axes_sum[0, col_i]
    im0 = ax0.imshow(grid_avg, cmap=_cmap_amp, aspect='equal',
                     origin='upper', vmin=0, vmax=1)
    ax0.set_title(retina, fontsize=8, pad=3)
    _clean_ax(ax0)
    if col_i == _n_retinas - 1:
        cbar0 = plt.colorbar(im0, ax=ax0, shrink=0.80, pad=0.03)
        cbar0.set_label('norm. avg amplitude', fontsize=5)
        cbar0.ax.tick_params(labelsize=5)
    _stim_dot(ax0)

    # ── row 1: first-pulse amplitude, min-max normalised per file, averaged ──────
    _ch_first: dict[str, list[float]] = defaultdict(list)
    _by_file_sum: dict[str, list] = defaultdict(list)
    for r in records:
        _by_file_sum[r['filename']].append(r)
    for _recs_in_file in _by_file_sum.values():
        _file_amps: dict[str, float] = {}
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

    grid_first = np.full((12, 12), np.nan)
    for ch, vals in _ch_first.items():
        loc = mea_name_to_location(ch)
        if loc is not None:
            grid_first[loc[0], loc[1]] = float(np.mean(vals))

    ax1 = axes_sum[1, col_i]
    im1 = ax1.imshow(grid_first, cmap=_cmap_amp, aspect='equal',
                     origin='upper', vmin=0, vmax=1)
    _clean_ax(ax1)
    if col_i == _n_retinas - 1:
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.80, pad=0.03)
        cbar1.set_label('norm. amplitude', fontsize=5)
        cbar1.ax.tick_params(labelsize=5)
    _stim_dot(ax1)

    # ── row 2: pulse-50  (y − y₀) / y₀, averaged per channel ──────────────────
    _ch_p50: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        v = _value_at_pulse(rec['pulses'], _LAST_PULSE)
        if v is not None and np.isfinite(v):
            _ch_p50[rec['channel']].append(v)

    grid_p50 = np.full((12, 12), np.nan)
    for ch, vals in _ch_p50.items():
        loc = mea_name_to_location(ch)
        if loc is not None:
            grid_p50[loc[0], loc[1]] = float(np.mean(vals))

    _finite_p50 = [v for v in grid_p50.flatten() if np.isfinite(v)]
    _vmax_p50   = min(max(abs(min(_finite_p50)), abs(max(_finite_p50))), 1.0) if _finite_p50 else 1.0

    ax2 = axes_sum[2, col_i]
    im2 = ax2.imshow(grid_p50, cmap=_cmap_norm, aspect='equal',
                     origin='upper', vmin=-_vmax_p50, vmax=_vmax_p50)
    _clean_ax(ax2)
    if col_i == _n_retinas - 1:
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.80, pad=0.03)
        cbar2.set_label('(y − y₀) / y₀', fontsize=5)
        cbar2.ax.tick_params(labelsize=5)
    _stim_dot(ax2)

axes_sum[0, 0].set_ylabel('avg amplitude', fontsize=7)
axes_sum[1, 0].set_ylabel('first amp (min-max)', fontsize=7)
axes_sum[2, 0].set_ylabel(f'pulse {_LAST_PULSE}  (y − y₀) / y₀', fontsize=7)

fig_sum.suptitle(
    f'Instability localisation — avg amplitude / first amplitude (min-max) / pulse-{_LAST_PULSE} change\n'
    f'y₀ = pulse {START_PULSE}  ·  red dot = stim electrode',
    fontsize=10,
)
plt.tight_layout()
plt.show()
plt.close(fig_sum)


# ── scatter: first amp (norm, avg across files) vs pulse-50 norm amp, per retina ──

_SCATTER_FREQ_HZ = {1, 10, 20}   # frequencies to include

def _file_freq_hz(filename: str) -> Optional[float]:
    m = re.search(r'(\d+(?:\.\d+)?)Hz', filename, re.IGNORECASE)
    return float(m.group(1)) if m else None


_freq_list   = sorted(_SCATTER_FREQ_HZ)
_freq_cmap   = plt.get_cmap('tab10')
_freq_colors = {f: _freq_cmap(i) for i, f in enumerate(_freq_list)}

fig_sc2, axes_sc2 = plt.subplots(1, _n_retinas,
                                  figsize=(3.5 * _n_retinas, 4),
                                  sharey=True)

if _n_retinas == 1:
    axes_sc2 = [axes_sc2]

for col_i, retina in enumerate(_retinas):
    ax = axes_sc2[col_i]

    for freq in _freq_list:
        records = [r for r in retina_records[retina]
                   if _file_freq_hz(r['filename']) == freq]

        # x: first amp min-max normalised within each file, averaged per channel
        _ch_first: dict[str, list[float]] = defaultdict(list)
        _by_file: dict[str, list] = defaultdict(list)
        for r in records:
            _by_file[r['filename']].append(r)
        for _recs_in_file in _by_file.values():
            _file_amps: dict[str, float] = {}
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
                _ch_first[ch].append((v - _fmin) / _frange)

        _ch_first_mean: dict[str, float] = {
            ch: float(np.mean(vals)) for ch, vals in _ch_first.items()
        }

        # y: pulse-50 norm amp averaged per channel
        _ch_p50: dict[str, list[float]] = defaultdict(list)
        for rec in records:
            v = _value_at_pulse(rec['pulses'], _LAST_PULSE)
            if v is not None and np.isfinite(v):
                _ch_p50[rec['channel']].append(v)

        _ch_p50_mean: dict[str, float] = {
            ch: float(np.mean(vals)) for ch, vals in _ch_p50.items()
        }

        _channels_sc = set(_ch_first_mean) & set(_ch_p50_mean)
        if not _channels_sc:
            continue
        xs = np.array([_ch_first_mean[ch] for ch in _channels_sc])
        ys = np.array([_ch_p50_mean[ch]   for ch in _channels_sc])
        ax.scatter(xs, ys, color=_freq_colors[freq], s=22, alpha=0.75,
                   linewidths=0, zorder=3, label=f'{freq:g} Hz')

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
    ax.set_title(retina, fontsize=8, loc='left', pad=3)
    ax.set_xlabel('first amp (norm.)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper right', framealpha=0.6,
              handlelength=1.0, labelspacing=0.3)

axes_sc2[0].set_ylabel(f'(y − y₀) / y₀  at pulse {_LAST_PULSE}', fontsize=7)

fig_sc2.suptitle(
    f'First amplitude (min-max norm.) vs pulse-{_LAST_PULSE} change  ·  color = frequency',
    fontsize=9,
)
plt.tight_layout()
plt.show()
plt.close(fig_sc2)