"""
instability_localization.py
============================
Loads ALL ex vivo direct-response CSV files, groups them by retina,
and produces one 1×5 heatmap figure per retina.

Each heatmap shows normalised amplitude (y − y₀) / y₀ on the 12×12 MEA grid
at a fixed pulse index (10, 20, 30, 40, 50).  y₀ is the amplitude at
START_PULSE.  When the same channel appears in multiple files for a retina,
values are averaged.  The stimulated electrode is marked with a red dot.
"""

import os
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

    records = []
    for ch, ch_df in df.groupby('channel'):
        short_ch = str(ch).split()[-1].upper()
        pulses = (
            ch_df[
                (ch_df['pulse_index'] >= START_PULSE) &
                (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
            ][['pulse_index', 'amplitude_mV']]
            .copy()
            .reset_index(drop=True)
        )
        if not pulses.empty:
            records.append({'channel': short_ch, 'pulses': pulses})
    return records


# ── normalisation ──────────────────────────────────────────────────────────────

def _has_full_window(pulses: pd.DataFrame) -> bool:
    return int(pulses['pulse_index'].max()) >= START_PULSE + PULSE_LIMIT - 1


def _normalise(pulses: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return DataFrame[pulse_index, norm_amp] = (amp − y0) / y0."""
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
    detected = per_pulse[per_pulse['amplitude_mV'] > 0].copy()
    if detected.empty:
        return None
    detected['norm_amp'] = (detected['amplitude_mV'] - y0) / y0
    return detected[['pulse_index', 'norm_amp']]


def _value_at_pulse(pulses: pd.DataFrame, pulse_idx: int) -> Optional[float]:
    if not _has_full_window(pulses):
        return None
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


# ── figure per retina ──────────────────────────────────────────────────────────

def _make_figure(retina: str, records: list[dict]) -> None:
    stim_ch   = STIM_ELECTRODE.get(retina)
    stim_pos  = mea_name_to_location(stim_ch) if stim_ch else None

    channels = sorted({r['channel'] for r in records})

    # Collect normalised values per (channel, pulse)
    ch_pulse_vals: dict[str, dict[int, list[float]]] = {
        ch: defaultdict(list) for ch in channels
    }
    for rec in records:
        ch = rec['channel']
        for p in HIST_PULSES:
            v = _value_at_pulse(rec['pulses'], p)
            if v is not None and np.isfinite(v):
                ch_pulse_vals[ch][p].append(v)

    # Average across files for each channel
    ch_pulse_mean: dict[str, dict[int, float]] = {}
    for ch in channels:
        ch_pulse_mean[ch] = {
            p: float(np.mean(vals))
            for p, vals in ch_pulse_vals[ch].items()
            if vals
        }

    # Build 12×12 grids
    def _grid(pulse_idx: int) -> np.ndarray:
        g = np.full((12, 12), np.nan)
        for ch, pd_dict in ch_pulse_mean.items():
            loc = mea_name_to_location(ch)
            if loc is None:
                continue
            r, c = loc
            val = pd_dict.get(pulse_idx)
            if val is not None:
                g[r, c] = val
        return g

    grids = {p: _grid(p) for p in HIST_PULSES}

    all_finite = [v for g in grids.values() for v in g.flatten() if np.isfinite(v)]
    vmax = min(max(abs(min(all_finite)), abs(max(all_finite))), 1.0) if all_finite else 1.0
    vmin = -vmax

    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='#e8e8e8')

    n = len(HIST_PULSES)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2))

    for ax, pulse_idx in zip(axes, HIST_PULSES):
        im = ax.imshow(grids[pulse_idx], cmap=cmap, aspect='equal',
                       origin='upper', vmin=vmin, vmax=vmax)
        ax.set_title(f'Pulse {pulse_idx}', fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.03)
        cbar.set_label('(y − y₀) / y₀', fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        if stim_pos is not None:
            sr, sc = stim_pos
            ax.plot(sc, sr, 'o', color='red', markersize=6, zorder=5,
                    markeredgewidth=0.8, markeredgecolor='white')

    stim_label = f'stim: {stim_ch}' if stim_ch else 'stim: unknown'
    fig.suptitle(
        f'{retina} — instability localisation   (y − y₀) / y₀\n'
        f'y₀ = amplitude at pulse {START_PULSE}  ·  {stim_label} (red dot)',
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)


for retina, records in sorted(retina_records.items()):
    _make_figure(retina, records)