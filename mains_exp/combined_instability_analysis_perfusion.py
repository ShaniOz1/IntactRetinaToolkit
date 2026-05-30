"""
combined_instability_analysis_perfusion.py
===========================================
Loads ex vivo perfusion-experiment direct-response CSV data and produces
the same instability figures as combined_instability_analysis_glucose.py,
with one subplot per experimental phase.

Phases come directly from the phase* subfolder names under SOURCE_DIR:
    phase1-normal          (baseline)
    phase2-high perfusion  (high perfusion)

Each CSV in PERFUSION_DIR is matched back to its source phase folder via
its embedded EDF filename (same convention as perfusion_instability_exploration.py).

Source CSVs  : Results/ex_vivo_perfusion/
Channels used: set ALLOWED_CHANNELS below (leave empty to include all channels)

Each record dict contains:
    phase       : str   (folder name, e.g. 'phase2-high perfusion')
    current_uA  : float
    freq_Hz     : float
    channel     : str   (short name, upper-cased, e.g. 'K4')
    pulses      : DataFrame[pulse_index, amplitude_mV]
    filename    : str   (source CSV basename)
"""

import os
import re
import glob
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── analysis parameters ───────────────────────────────────────────────────────

START_PULSE   = 2      # first pulse index to include (pulses 0 and 1 are skipped)
PULSE_LIMIT   = 50     # number of pulses starting from START_PULSE
                       #   -> kept window: [START_PULSE, START_PULSE + PULSE_LIMIT)
HIST_PULSES   = [10, 20, 30, 40, 50]   # pulse indices shown as histogram rows
SMOOTH_WINDOW = 7      # rolling-average window for overview lines (1 = disabled)

N_PHASES      = 2      # how many phase* folders to use (sorted alphabetically)

# Stimulation frequencies to include (Hz).
# Leave empty  →  all frequencies are kept.
FREQ_HZ: set[float] = {1, 10, 20}


# ── paths ─────────────────────────────────────────────────────────────────────

SOURCE_DIR    = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3'
PERFUSION_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_perfusion'

# Reference channels for the 2025.11.02 Retina3 (J9) perfusion experiment.
ALLOWED_CHANNELS: set[str] = {'L9', 'E9'}


# ── build EDF -> phase-folder lookup from SOURCE_DIR ─────────────────────────

phase_dirs = sorted(glob.glob(os.path.join(SOURCE_DIR, 'phase*')))[:N_PHASES]

if not phase_dirs:
    print(f'No phase* folders found under {SOURCE_DIR}')
    raise SystemExit

PHASE_ORDER: list[str] = [os.path.basename(d) for d in phase_dirs]

# {edf_basename (lower) -> phase_folder_name}
_edf_to_phase: dict[str, str] = {}
for _phase_dir in phase_dirs:
    _phase_name = os.path.basename(_phase_dir)
    for _edf_path in glob.glob(os.path.join(_phase_dir, '*.edf')):
        _edf_to_phase[os.path.basename(_edf_path).lower()] = _phase_name

print(f'Phase folders used: {PHASE_ORDER}')
print(f'EDF files indexed : {len(_edf_to_phase)}\n')


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_current_freq(filename: str):
    """Return (current_uA, freq_Hz) parsed from filename, or (None, None)."""
    cur  = re.search(r'(\d+(?:\.\d+)?)\s*uA', filename, re.IGNORECASE)
    freq = re.search(r'(\d+(?:\.\d+)?)\s*Hz', filename, re.IGNORECASE)
    if not cur or not freq:
        return None, None
    current = float(cur.group(1))
    # Fix timestamp bleed into current value (e.g. "197uA" -> 7 uA)
    if current == int(current) and int(current) % 10 == 7 and current != 7:
        current = 7.0
    return current, float(freq.group(1))


def csv_to_phase(csv_fname: str) -> Optional[str]:
    """
    Recover the source EDF filename from a CSV basename, then look up its
    phase folder.
    CSV format: '{normal|high_perfusion}_{edf_fname}_direct_response.csv'
    Returns None if the EDF cannot be matched to any indexed phase folder.
    """
    stem = re.sub(r'^(?:normal|high_perfusion)_', '', csv_fname, flags=re.IGNORECASE)
    stem = re.sub(r'_direct_response\.csv$', '', stem, flags=re.IGNORECASE)
    return _edf_to_phase.get(stem.lower())


def load_perfusion_pulses(csv_path: str) -> list[dict]:
    """
    Load pulse data from one perfusion CSV.
    If ALLOWED_CHANNELS is non-empty, only those channel short-names are kept.
    Returns list of {'channel': short_name, 'pulses': DataFrame[pulse_index, amplitude_mV]}.
    """
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()

    records = []
    for ch, ch_df in df.groupby('channel'):
        short_ch = str(ch).split()[-1].upper()
        if ALLOWED_CHANNELS and short_ch not in ALLOWED_CHANNELS:
            continue
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][['pulse_index', 'amplitude_mV']]
                  .copy()
                  .reset_index(drop=True))
        records.append({'channel': short_ch, 'pulses': pulses})
    return records


# ── load all perfusion CSVs ───────────────────────────────────────────────────

perfusion_records: list[dict] = []

for path in sorted(glob.glob(os.path.join(PERFUSION_DIR, '*_direct_response.csv'))):
    fname = os.path.basename(path)
    phase = csv_to_phase(fname)
    if phase is None:
        print(f'  [skip] could not match to a phase folder: {fname}')
        continue

    current, freq = parse_current_freq(fname)
    if current is None:
        print(f'  [skip] could not parse current/freq: {fname}')
        continue

    if FREQ_HZ and freq not in FREQ_HZ:
        continue

    for ch_rec in load_perfusion_pulses(path):
        perfusion_records.append({
            'phase':      phase,
            'current_uA': current,
            'freq_Hz':    freq,
            'channel':    ch_rec['channel'],
            'pulses':     ch_rec['pulses'],
            'filename':   fname,
        })


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(records: list[dict]) -> None:
    if not records:
        print('\nPERFUSION: no records loaded.')
        return

    df = pd.DataFrame([
        {'phase': r['phase'], 'current_uA': r['current_uA'],
         'freq_Hz': r['freq_Hz'], 'channel': r['channel']}
        for r in records
    ])

    counts = (df.groupby(['phase', 'current_uA', 'freq_Hz'])['channel']
                .nunique()
                .reset_index(name='n_channels'))

    col_keys   = sorted(counts[['current_uA', 'freq_Hz']]
                        .drop_duplicates()
                        .itertuples(index=False, name=None))
    col_labels = {(c, f): f'{c:g}uA / {f:g}Hz' for c, f in col_keys}

    phases = sorted(counts['phase'].unique(),
                    key=lambda p: PHASE_ORDER.index(p) if p in PHASE_ORDER else 999)

    pivot: dict[str, dict[str, str]] = {}
    for _, row in counts.iterrows():
        key = (row['current_uA'], row['freq_Hz'])
        pivot.setdefault(row['phase'], {})[col_labels[key]] = str(int(row['n_channels']))

    phase_w = max(len('Phase'), max(len(p) for p in phases))
    col_ws  = {lbl: max(len(lbl), 1) for lbl in col_labels.values()}

    header = f'  {"Phase":<{phase_w}}' + ''.join(f'  {lbl:>{col_ws[lbl]}}' for lbl in col_labels.values())
    sep    = '  ' + '-' * (len(header) - 2)

    print(f'\nPERFUSION  (total records: {len(records)})')
    print(sep)
    print(header)
    print(sep)
    for phase in phases:
        row_str = f'  {phase:<{phase_w}}'
        for lbl in col_labels.values():
            val = pivot.get(phase, {}).get(lbl, '-')
            row_str += f'  {val:>{col_ws[lbl]}}'
        print(row_str)
    print(sep)


_print_summary(perfusion_records)
print()


# ── group records by phase ────────────────────────────────────────────────────

records_by_phase: dict[str, list[dict]] = {}
for rec in perfusion_records:
    records_by_phase.setdefault(rec['phase'], []).append(rec)

present_phases = [p for p in PHASE_ORDER if p in records_by_phase]
n_phases = len(present_phases)

if n_phases == 0:
    print('No data loaded -- check PERFUSION_DIR, SOURCE_DIR, and ALLOWED_CHANNELS.')
    raise SystemExit


# ── helpers: full-window guard and normalisation ──────────────────────────────

def _has_full_window(rec: dict) -> bool:
    """Return True only if the record contains pulses reaching the end of the window."""
    return int(rec['pulses']['pulse_index'].max()) >= START_PULSE + PULSE_LIMIT - 1


def _normalise(rec: dict) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns [pulse_index, norm_amp],
    normalised as (amp - y0) / y0  where y0 = amplitude at START_PULSE.
    Returns None if START_PULSE has no detected response.
    """
    per_pulse = (
        rec['pulses']
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


# ── overview figure: row 0 = normalised, row 1 = raw amplitude ───────────────

_grid_ys = [0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75]

fig_ov, axes_ov = plt.subplots(2, n_phases,
                               figsize=(4 * n_phases, 6),
                               sharex=True)

if n_phases == 1:
    axes_ov = [[axes_ov[0]], [axes_ov[1]]]

# ── row 0: normalised traces ──────────────────────────────────────────────────
for col_i, phase in enumerate(present_phases):
    ax = axes_ov[0][col_i]
    for rec in records_by_phase[phase]:
        if not _has_full_window(rec):
            continue
        norm_df = _normalise(rec)
        if norm_df is None:
            continue
        y_smooth = (norm_df['norm_amp']
                    .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                    .mean())
        ax.plot(norm_df['pulse_index'], y_smooth,
                color='black', linewidth=0.4, alpha=0.4)

    for _y in _grid_ys:
        ax.axhline(_y, color='grey', linewidth=0.35, alpha=0.3)

    ax.set_ylim(-1, 1)
    ax.set_title(phase, fontsize=10, loc='left', pad=3)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes_ov[0][0].set_ylabel('(y - y0) / y0', fontsize=7)

# ── row 1: raw amplitude traces ───────────────────────────────────────────────
for col_i, phase in enumerate(present_phases):
    ax = axes_ov[1][col_i]
    for rec in records_by_phase[phase]:
        if not _has_full_window(rec):
            continue
        per_pulse = (rec['pulses']
                     .groupby('pulse_index')['amplitude_mV']
                     .max()
                     .reset_index())
        y_smooth = (per_pulse['amplitude_mV']
                    .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                    .mean())
        ax.plot(per_pulse['pulse_index'], y_smooth,
                color='black', linewidth=0.4, alpha=0.4)

    ax.set_xlim(START_PULSE, START_PULSE + PULSE_LIMIT)
    ax.set_ylim(0, 9)
    ax.set_xlabel('# pulse', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes_ov[1][0].set_ylabel('amplitude (mV)', fontsize=7)

plt.tight_layout()
plt.show()
plt.close(fig_ov)


# ── histograms: normalised amplitude at each HIST_PULSE, one column per phase ─

def _value_at_pulse(rec: dict, pulse_idx: int) -> Optional[float]:
    """Return the normalised amplitude at a specific pulse index, or None."""
    if not _has_full_window(rec):
        return None
    norm_df = _normalise(rec)
    if norm_df is None:
        return None
    row = norm_df[norm_df['pulse_index'] == pulse_idx]
    if row.empty:
        return None
    return float(row['norm_amp'].iloc[0])


n_hist_rows = len(HIST_PULSES)
_hist_bins  = np.linspace(-1, 1, 11)   # 10 bins of width 0.2, shared across all subplots

fig_hist, axes_hist = plt.subplots(n_hist_rows, n_phases,
                                   figsize=(3 * n_phases, 1.6 * n_hist_rows),
                                   sharex=True, sharey=True)

# Ensure axes_hist is always 2-D: [row][col]
if n_hist_rows == 1 and n_phases == 1:
    axes_hist = [[axes_hist]]
elif n_hist_rows == 1:
    axes_hist = [list(axes_hist)]
elif n_phases == 1:
    axes_hist = [[axes_hist[r]] for r in range(n_hist_rows)]

for row_i, pulse_idx in enumerate(HIST_PULSES):
    for col_i, phase in enumerate(present_phases):
        ax = axes_hist[row_i][col_i]
        vals = [v for rec in records_by_phase[phase]
                if (v := _value_at_pulse(rec, pulse_idx)) is not None
                and np.isfinite(v)]

        if vals:
            weights = [100.0 / len(vals)] * len(vals)
            bar_heights, _ = np.histogram(vals, bins=_hist_bins, weights=weights)
            ax.hist(vals, bins=_hist_bins, weights=weights,
                    color='#c8c8c8', edgecolor='black', linewidth=0.4, alpha=0.9)
            ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
            if len(vals) > 1:
                x_kde    = np.linspace(-1, 1, 300)
                kde_vals = gaussian_kde(vals)(x_kde)
                max_bar  = bar_heights.max()
                if kde_vals.max() > 0 and max_bar > 0:
                    kde_vals = kde_vals / kde_vals.max() * max_bar
                ax.plot(x_kde, kde_vals, color='black', linewidth=1.5)

        ax.set_xlim(-1, 1)
        ax.set_title(f'{phase}  pulse {pulse_idx}', fontsize=7, loc='left', pad=2)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes_hist[row_i][0].set_ylabel('%', fontsize=6)

for col_i in range(n_phases):
    axes_hist[-1][col_i].set_xlabel('(y - y0) / y0', fontsize=6)

plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
plt.close(fig_hist)
