"""
combined_instability_analysis_glucose.py
=========================================
Loads ex vivo glucose-experiment direct-response CSV data and produces the
same instability figures as combined_instability_analysis.py, but with one
subplot per experimental phase instead of ex vivo / intact.

Phases come directly from the Phase* subfolder names under SOURCE_DIR:
    Phase1 - Normal   (pre-glucose baseline)
    Phase2 - Low      (low glucose)
    Phase3 -Normal    (post-glucose recovery)

Only the first N_PHASES folders (sorted) are used.  Each CSV in GLUCOSE_DIR
is matched back to its source Phase folder via its embedded EDF filename.

Source CSVs  : Results/ex_vivo_glucose/
Channels used: K9 and D11  (2025.11.12 retina)

Each record dict contains:
    phase       : str   (folder name, e.g. 'Phase2 - Low')
    current_uA  : float
    freq_Hz     : float
    channel     : str   (short name, upper-cased, e.g. 'K9')
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

from dataobj.channel_utils import mea_name_to_location


# ── analysis parameters ───────────────────────────────────────────────────────

START_PULSE   = 2      # first pulse index to include (pulses 0 and 1 are skipped)
PULSE_LIMIT   = 50     # number of pulses starting from START_PULSE
                       #   → kept window: [START_PULSE, START_PULSE + PULSE_LIMIT)
HIST_PULSES   = [10, 20, 30, 40, 50]   # pulse indices shown as histogram rows
SMOOTH_WINDOW = 7      # rolling-average window for overview lines (1 = disabled)

N_PHASES      = 3      # how many Phase* folders to use (sorted alphabetically)


# ── paths ─────────────────────────────────────────────────────────────────────

SOURCE_DIR  = r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1'
GLUCOSE_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_glucose'

# Reference channels for the 2025.11.12 retina (glucose experiment)
# ALLOWED_CHANNELS: set[str] = {'K9', 'D11'}
ALLOWED_CHANNELS: set[str] = {'K9', 'J9','L9',  'D11', 'K10', 'C11'}


# ── build EDF → phase-folder lookup from SOURCE_DIR ───────────────────────────
# Scan the first N_PHASES Phase* subfolders and map every EDF basename to its
# folder name.  This is used to assign a phase to each CSV in GLUCOSE_DIR.

phase_dirs = sorted(glob.glob(os.path.join(SOURCE_DIR, 'Phase*')))[:N_PHASES]

if not phase_dirs:
    print(f'No Phase* folders found under {SOURCE_DIR}')
    raise SystemExit

# Canonical phase order = alphabetical sort of the first N_PHASES folders
PHASE_ORDER: list[str] = [os.path.basename(d) for d in phase_dirs]

# {edf_basename (lower) → phase_folder_name}
_edf_to_phase: dict[str, str] = {}
for phase_dir in phase_dirs:
    phase_name = os.path.basename(phase_dir)
    for edf_path in glob.glob(os.path.join(phase_dir, '*.edf')):
        _edf_to_phase[os.path.basename(edf_path).lower()] = phase_name

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
    Phase folder.  CSV format: '{low|normal}_{edf_fname}_direct_response.csv'
    Returns None if the EDF cannot be matched to any indexed Phase folder.
    """
    # Strip low_/normal_ prefix and _direct_response.csv suffix
    stem = re.sub(r'^(?:low|normal)_', '', csv_fname, flags=re.IGNORECASE)
    stem = re.sub(r'_direct_response\.csv$', '', stem, flags=re.IGNORECASE)
    return _edf_to_phase.get(stem.lower())


def load_glucose_pulses(csv_path: str) -> list[dict]:
    """
    Load pulse data for ALLOWED_CHANNELS from one glucose CSV.
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
        if short_ch not in ALLOWED_CHANNELS:
            continue
        cols = ['pulse_index', 'amplitude_mV']
        for _extra in ('latency_ms', 'width_ms'):
            if _extra in ch_df.columns:
                cols.append(_extra)
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][cols]
                  .copy()
                  .reset_index(drop=True))
        records.append({'channel': short_ch, 'pulses': pulses})
    return records


# ── load all glucose CSVs ─────────────────────────────────────────────────────

glucose_records: list[dict] = []

for path in sorted(glob.glob(os.path.join(GLUCOSE_DIR, '*_direct_response.csv'))):
    fname = os.path.basename(path)
    phase = csv_to_phase(fname)
    if phase is None:
        print(f'  [skip] could not match to a Phase folder: {fname}')
        continue

    current, freq = parse_current_freq(fname)
    if current is None:
        print(f'  [skip] could not parse current/freq: {fname}')
        continue

    for ch_rec in load_glucose_pulses(path):
        glucose_records.append({
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
        print('\nGLUCOSE: no records loaded.')
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

    print(f'\nGLUCOSE  (total records: {len(records)})')
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


_print_summary(glucose_records)
print()


# ── group records by phase (only phases with data, in canonical order) ─────────

records_by_phase: dict[str, list[dict]] = {}
for rec in glucose_records:
    records_by_phase.setdefault(rec['phase'], []).append(rec)

present_phases = [p for p in PHASE_ORDER if p in records_by_phase]
n_phases = len(present_phases)

if n_phases == 0:
    print('No data loaded -- check GLUCOSE_DIR, SOURCE_DIR, and ALLOWED_CHANNELS.')
    raise SystemExit


# ── normalisation ─────────────────────────────────────────────────────────────

def _normalise(rec: dict) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns [pulse_index, norm_amp] for detected pulses,
    normalised as (amp - y0) / y0  where y0 = amplitude at pulse START_PULSE.
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


# ── overview figure: row 1 = normalised, row 2 = raw amplitude ───────────────

_grid_ys = [0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75]

fig_ov, axes_ov = plt.subplots(2, n_phases,
                               figsize=(4 * n_phases, 6),
                               sharex=True)

# axes_ov is always 2-D [row][col]
if n_phases == 1:
    axes_ov = [[axes_ov[0]], [axes_ov[1]]]

# ── row 0: normalised traces ──────────────────────────────────────────────────
for col_i, phase in enumerate(present_phases):
    ax = axes_ov[0][col_i]
    for rec in records_by_phase[phase]:
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

axes_ov[0][0].set_ylabel('(y − y₀) / y₀', fontsize=7)

# ── row 1: raw amplitude traces ───────────────────────────────────────────────
for col_i, phase in enumerate(present_phases):
    ax = axes_ov[1][col_i]
    for rec in records_by_phase[phase]:
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
    ax.set_ylim(0, 7)
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
                # Scale KDE peak to match tallest bar — keeps shape without exceeding 100%
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


# ════════════════════════════════════════════════════════════════════════════════
# INTACT GLUCOSE
# ════════════════════════════════════════════════════════════════════════════════

INTACT_GLUCOSE_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_glucose'
INTACT_PHASE_BASE  = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2'

_INTACT_PHASE_RE = re.compile(r'^phase\d+$', re.IGNORECASE)

# Build RHS-basename (lower) → phase-folder-name lookup
_rhs_to_phase: dict[str, str] = {}
for _root, _dirs, _files in os.walk(INTACT_PHASE_BASE):
    _top = os.path.relpath(_root, INTACT_PHASE_BASE).split(os.sep)[0]
    if _INTACT_PHASE_RE.match(_top):
        for _f in _files:
            if _f.lower().endswith('.rhs'):
                _rhs_to_phase[_f.lower()] = _top

INTACT_PHASE_ORDER: list[str] = sorted(set(_rhs_to_phase.values()))[:3]
print(f'Intact phase folders: {INTACT_PHASE_ORDER}')
print(f'RHS files indexed   : {len(_rhs_to_phase)}\n')


def intact_csv_to_phase(csv_fname: str) -> Optional[str]:
    """Recover the source RHS basename from a CSV name and look up its phase."""
    # 'Retina2_Ch4_..._251105_160302.rhs_direct_response.csv'
    #   → strip 'Retina2_'  and '_direct_response.csv'
    stem = re.sub(r'^Retina\d+_', '', csv_fname, flags=re.IGNORECASE)
    stem = re.sub(r'_direct_response\.csv$', '', stem, flags=re.IGNORECASE)
    return _rhs_to_phase.get(stem.lower())


def load_intact_glucose_pulses(csv_path: str) -> list[dict]:
    """Load pulse data for all probe channels from one intact glucose CSV."""
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    df['channel'] = pd.to_numeric(df['channel'], errors='coerce')
    df = df.dropna(subset=['channel'])

    records = []
    for ch, ch_df in df.groupby('channel'):
        cols = ['pulse_index', 'amplitude_mV']
        for _extra in ('latency_ms', 'width_ms'):
            if _extra in ch_df.columns:
                cols.append(_extra)
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][cols]
                  .copy().reset_index(drop=True))
        if not pulses.empty:
            records.append({'channel': int(ch), 'pulses': pulses})
    return records


# ── load ───────────────────────────────────────────────────────────────────────

intact_glucose_records: list[dict] = []

for path in sorted(glob.glob(os.path.join(INTACT_GLUCOSE_DIR, '*_direct_response.csv'))):
    fname = os.path.basename(path)
    phase = intact_csv_to_phase(fname)
    if phase is None:
        print(f'  [intact skip] no phase match: {fname}')
        continue

    current, freq = parse_current_freq(fname)
    if current is None:
        print(f'  [intact skip] no current/freq: {fname}')
        continue

    for ch_rec in load_intact_glucose_pulses(path):
        intact_glucose_records.append({
            'phase':      phase,
            'current_uA': current,
            'freq_Hz':    freq,
            'channel':    ch_rec['channel'],
            'pulses':     ch_rec['pulses'],
            'filename':   fname,
        })

_intact_unique_channels = sorted({r['channel'] for r in intact_glucose_records})
print(f'Intact glucose: {len(intact_glucose_records)} records loaded.')
print(f'  unique channels ({len(_intact_unique_channels)}): {_intact_unique_channels}\n')

# ── group by phase ─────────────────────────────────────────────────────────────

intact_by_phase: dict[str, list[dict]] = {}
for rec in intact_glucose_records:
    intact_by_phase.setdefault(rec['phase'], []).append(rec)

intact_present = [p for p in INTACT_PHASE_ORDER if p in intact_by_phase]
n_intact       = len(intact_present)

if n_intact == 0:
    print('No intact glucose data — skipping intact figures.')
else:

    # ── overview figure ────────────────────────────────────────────────────────

    fig_iv, axes_iv = plt.subplots(2, n_intact,
                                   figsize=(4 * n_intact, 6),
                                   sharex=True)
    if n_intact == 1:
        axes_iv = [[axes_iv[0]], [axes_iv[1]]]

    for col_i, phase in enumerate(intact_present):
        ax = axes_iv[0][col_i]
        for rec in intact_by_phase[phase]:
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

    axes_iv[0][0].set_ylabel('(y − y₀) / y₀', fontsize=7)

    for col_i, phase in enumerate(intact_present):
        ax = axes_iv[1][col_i]
        for rec in intact_by_phase[phase]:
            per_pulse = (rec['pulses']
                         .groupby('pulse_index')['amplitude_mV']
                         .max().reset_index())
            y_smooth = (per_pulse['amplitude_mV']
                        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                        .mean())
            ax.plot(per_pulse['pulse_index'], y_smooth,
                    color='black', linewidth=0.4, alpha=0.4)
        ax.set_xlim(START_PULSE, START_PULSE + PULSE_LIMIT)
        ax.set_ylim(0, 7)
        ax.set_xlabel('# pulse', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes_iv[1][0].set_ylabel('amplitude (mV)', fontsize=7)

    fig_iv.suptitle('Intact glucose — instability overview', fontsize=10)
    plt.tight_layout()
    plt.show()
    plt.close(fig_iv)

    # ── histogram figure ───────────────────────────────────────────────────────

    fig_ih, axes_ih = plt.subplots(n_hist_rows, n_intact,
                                   figsize=(3 * n_intact, 1.6 * n_hist_rows),
                                   sharex=True, sharey=True)
    if n_hist_rows == 1 and n_intact == 1:
        axes_ih = [[axes_ih]]
    elif n_hist_rows == 1:
        axes_ih = [list(axes_ih)]
    elif n_intact == 1:
        axes_ih = [[axes_ih[r]] for r in range(n_hist_rows)]

    for row_i, pulse_idx in enumerate(HIST_PULSES):
        for col_i, phase in enumerate(intact_present):
            ax = axes_ih[row_i][col_i]
            vals = [v for rec in intact_by_phase[phase]
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

        axes_ih[row_i][0].set_ylabel('%', fontsize=6)

    for col_i in range(n_intact):
        axes_ih[-1][col_i].set_xlabel('(y − y₀) / y₀', fontsize=6)

    fig_ih.suptitle('Intact glucose — instability histograms', fontsize=10)
    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()
    plt.close(fig_ih)


# ── latency distribution: ex vivo (row 0) then intact (row 1) ─────────────────

def _latencies(records_by_ph, phases):
    """Return {phase: [latency_ms, ...]} with only finite detected values."""
    out = {}
    for p in phases:
        vals = []
        for rec in records_by_ph.get(p, []):
            if 'latency_ms' in rec['pulses'].columns:
                vals.extend(
                    rec['pulses']['latency_ms']
                    .dropna()
                    .values.tolist()
                )
        out[p] = [v for v in vals if np.isfinite(v)]
    return out


ev_lat  = _latencies(records_by_phase,  present_phases)
_n_ev   = len(present_phases)

_has_intact_lat = n_intact > 0
in_lat  = _latencies(intact_by_phase, intact_present) if _has_intact_lat else {}
_n_in   = n_intact if _has_intact_lat else 0

_n_lat_cols = max(_n_ev, _n_in)
_n_lat_rows = 1 + (1 if _has_intact_lat else 0)

fig_lat, axes_lat = plt.subplots(_n_lat_rows, _n_lat_cols,
                                 figsize=(2.8 * _n_lat_cols, 2.8 * _n_lat_rows),
                                 sharey=False)

if _n_lat_rows == 1:
    axes_lat = [list(np.atleast_1d(axes_lat))]
else:
    axes_lat = [list(row) for row in axes_lat]

def _phase_short(phase: str) -> str:
    """Extract 'normal' / 'low' from phase name, or the phase number for intact."""
    m = re.search(r'(normal|low)', phase, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r'\d+', phase)
    return m.group(0) if m else phase


# ── helper: draw one distribution histogram ───────────────────────────────────
def _lat_hist(ax, vals, corner_label, color='#c8c8c8', x_max=7, y_max=80):
    if vals:
        bins  = np.linspace(0, x_max, 20)
        weights = [100.0 / len(vals)] * len(vals)
        bar_heights, _ = np.histogram(vals, bins=bins, weights=weights)
        ax.hist(vals, bins=bins, weights=weights,
                color=color, edgecolor='black', linewidth=0.4, alpha=0.9)
        if len(vals) > 1:
            x_kde    = np.linspace(0, x_max, 300)
            kde_vals = gaussian_kde(vals)(x_kde)
            max_bar  = bar_heights.max()
            if kde_vals.max() > 0 and max_bar > 0:
                kde_vals = kde_vals / kde_vals.max() * max_bar
            ax.plot(x_kde, kde_vals, color='black', linewidth=1.5)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.text(0.03, 0.97, corner_label, transform=ax.transAxes,
            fontsize=6, va='top', ha='left', style='italic', color='dimgrey')
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


for col_i, phase in enumerate(present_phases):
    _lat_hist(axes_lat[0][col_i], ev_lat[phase],
              f'ex vivo · {_phase_short(phase)}')
for col_i in range(_n_ev, _n_lat_cols):
    axes_lat[0][col_i].set_visible(False)
axes_lat[0][0].set_ylabel('%', fontsize=6)

_INTACT_PHASE_LABELS = ['normal', 'low', 'normal']

if _has_intact_lat:
    for col_i, phase in enumerate(intact_present):
        phase_label = _INTACT_PHASE_LABELS[col_i] if col_i < len(_INTACT_PHASE_LABELS) else str(col_i + 1)
        _lat_hist(axes_lat[1][col_i], in_lat[phase],
                  f'intact · {phase_label}',
                  color='#a8c8e8')
    for col_i in range(_n_in, _n_lat_cols):
        axes_lat[1][col_i].set_visible(False)
    axes_lat[1][0].set_ylabel('%', fontsize=6)
    for ax in axes_lat[1]:
        ax.set_xlabel('latency (ms)', fontsize=6)
else:
    for ax in axes_lat[0]:
        ax.set_xlabel('latency (ms)', fontsize=6)

fig_lat.suptitle('Latency distribution per phase', fontsize=10)
plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
plt.close(fig_lat)


# ── shared helper: build a figure for any pulse-level metric ──────────────────

def _metric_figure(col, xlabel, x_max, y_max, title):
    """Draw a 2-row (ex vivo / intact) figure for a given pulses column."""
    def _collect(records_by_ph, phases):
        out = {}
        for p in phases:
            vals = []
            for r in records_by_ph.get(p, []):
                if col in r['pulses'].columns:
                    vals.extend(r['pulses'][col].dropna().values.tolist())
            out[p] = [v for v in vals if np.isfinite(v) and v > 0]
        return out

    ev_vals = _collect(records_by_phase, present_phases)
    in_vals = _collect(intact_by_phase, intact_present) if _has_intact_lat else {}

    n_rows = 1 + (1 if _has_intact_lat else 0)
    n_cols = max(_n_ev, _n_in if _has_intact_lat else 0)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_rows),
                             sharey=False)
    if n_rows == 1:
        axes = [list(np.atleast_1d(axes))]
    else:
        axes = [list(row) for row in axes]

    for col_i, phase in enumerate(present_phases):
        _lat_hist(axes[0][col_i], ev_vals[phase],
                  f'ex vivo · {_phase_short(phase)}',
                  x_max=x_max, y_max=y_max)
    for col_i in range(_n_ev, n_cols):
        axes[0][col_i].set_visible(False)
    axes[0][0].set_ylabel('%', fontsize=6)

    if _has_intact_lat:
        for col_i, phase in enumerate(intact_present):
            phase_label = _INTACT_PHASE_LABELS[col_i] if col_i < len(_INTACT_PHASE_LABELS) else str(col_i + 1)
            _lat_hist(axes[1][col_i], in_vals[phase],
                      f'intact · {phase_label}',
                      color='#a8c8e8', x_max=x_max, y_max=y_max)
        for col_i in range(_n_in, n_cols):
            axes[1][col_i].set_visible(False)
        axes[1][0].set_ylabel('%', fontsize=6)
        for ax in axes[1]:
            ax.set_xlabel(xlabel, fontsize=6)
    else:
        for ax in axes[0]:
            ax.set_xlabel(xlabel, fontsize=6)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()
    plt.close(fig)


_metric_figure('width_ms',     'width (ms)', x_max=5,  y_max=80, title='Width distribution per phase')
_metric_figure('amplitude_mV', 'amplitude (mV)', x_max=2, y_max=80, title='Amplitude distribution per phase')


# ── speed figure: distance / latency ──────────────────────────────────────────

# Ex vivo: stim electrode G10, pitch 200 µm
_EV_STIM   = 'G10'
_MEA_PITCH = 0.2   # mm per electrode step

def _mea_dist_mm(ch: str) -> float:
    s = mea_name_to_location(_EV_STIM)
    t = mea_name_to_location(ch)
    if s is None or t is None:
        return float('nan')
    return np.sqrt((s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2) * _MEA_PITCH

EV_DIST_MM: dict[str, float] = {ch: _mea_dist_mm(ch) for ch in ALLOWED_CHANNELS}
print('Ex vivo distances from stim electrode:')
for ch, d in EV_DIST_MM.items():
    print(f'  {_EV_STIM} → {ch}: {d:.3f} mm')

# Intact: stim electrode Ch4, distances in mm
IN_DIST_MM: dict[int, float] = {
    2:  0.9,
    6:  0.5,
    7:  0.8,
    24: 0.7,
}


def _collect_speeds(records_by_ph, phases, dist_map):
    """Return {phase: [speed_m_s, ...]} using distance_mm / latency_ms = m/s."""
    out = {}
    for p in phases:
        speeds = []
        for r in records_by_ph.get(p, []):
            ch   = r['channel']
            dist = dist_map.get(ch)
            if dist is None or not np.isfinite(dist):
                continue
            if 'latency_ms' not in r['pulses'].columns:
                continue
            for lat in r['pulses']['latency_ms'].dropna().values:
                if lat > 0:
                    speeds.append(dist / lat)   # mm/ms = m/s
        out[p] = [v for v in speeds if np.isfinite(v)]
    return out


ev_speeds = _collect_speeds(records_by_phase, present_phases, EV_DIST_MM)
in_speeds = _collect_speeds(intact_by_phase,  intact_present, IN_DIST_MM) if _has_intact_lat else {}

n_rows_sp = 1 + (1 if _has_intact_lat else 0)
n_cols_sp = max(_n_ev, _n_in if _has_intact_lat else 0)

fig_sp, axes_sp = plt.subplots(n_rows_sp, n_cols_sp,
                               figsize=(2.8 * n_cols_sp, 2.8 * n_rows_sp),
                               sharey=False)
if n_rows_sp == 1:
    axes_sp = [list(np.atleast_1d(axes_sp))]
else:
    axes_sp = [list(row) for row in axes_sp]

for col_i, phase in enumerate(present_phases):
    _lat_hist(axes_sp[0][col_i], ev_speeds[phase],
              f'ex vivo · {_phase_short(phase)}',
              x_max=1.0, y_max=80)
for col_i in range(_n_ev, n_cols_sp):
    axes_sp[0][col_i].set_visible(False)
axes_sp[0][0].set_ylabel('%', fontsize=6)

if _has_intact_lat:
    for col_i, phase in enumerate(intact_present):
        phase_label = _INTACT_PHASE_LABELS[col_i] if col_i < len(_INTACT_PHASE_LABELS) else str(col_i + 1)
        _lat_hist(axes_sp[1][col_i], in_speeds[phase],
                  f'intact · {phase_label}',
                  color='#a8c8e8', x_max=1.0, y_max=80)
    for col_i in range(_n_in, n_cols_sp):
        axes_sp[1][col_i].set_visible(False)
    axes_sp[1][0].set_ylabel('%', fontsize=6)
    for ax in axes_sp[1]:
        ax.set_xlabel('speed (m/s)', fontsize=6)
else:
    for ax in axes_sp[0]:
        ax.set_xlabel('speed (m/s)', fontsize=6)

fig_sp.suptitle('Propagation speed per phase  (distance / latency)', fontsize=10)
plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
plt.close(fig_sp)


# ── scatter: distance vs latency / vs pulse-N norm ────────────────────────────

_SCATTER_PULSE_G = 20

_all_freqs_g  = sorted({r['freq_Hz'] for r in glucose_records + intact_glucose_records})
_freq_cmap_g  = plt.get_cmap('tab10')
_freq_color_g = {f: _freq_cmap_g(i % 10) for i, f in enumerate(_all_freqs_g)}
_n_fg         = len(_all_freqs_g)
_freq_jitter_g = {f: (i - (_n_fg - 1) / 2) * 0.02 for i, f in enumerate(_all_freqs_g)}

# First pass: shared axis limits
_adl_g, _adp_g = [], []
for _recs, _dmap in [(glucose_records, EV_DIST_MM), (intact_glucose_records, IN_DIST_MM)]:
    for _r in _recs:
        _d = _dmap.get(_r['channel'])
        if _d is None or not np.isfinite(_d):
            continue
        if 'latency_ms' in _r['pulses'].columns:
            for _lat in _r['pulses']['latency_ms'].dropna().values:
                if np.isfinite(_lat) and _lat > 0:
                    _adl_g.append((_d, _lat))
        _pv = _value_at_pulse(_r, _SCATTER_PULSE_G)
        if _pv is not None and np.isfinite(_pv):
            _adp_g.append((_d, _pv))

_xlim_lat_g = (0, max(d for d, _ in _adl_g) * 1.1) if _adl_g else (0, 1)
_ylim_lat_g = (0, max(l for _, l in _adl_g) * 1.1) if _adl_g else (0, 10)
_xlim_p_g   = (0, max(d for d, _ in _adp_g) * 1.1) if _adp_g else (0, 1)
_pv_g       = [v for _, v in _adp_g]
_ylim_p_g   = ((min(_pv_g) * 1.1, max(_pv_g) * 1.1) if _pv_g else (-1, 1))

fig_sc_g, axes_sc_g = plt.subplots(2, 2, figsize=(8, 6))

for row_i, (sc_recs, dist_map, src_label) in enumerate([
    (glucose_records,        EV_DIST_MM, 'Ex vivo'),
    (intact_glucose_records, IN_DIST_MM, 'Intact'),
]):
    ax_lat = axes_sc_g[row_i, 0]
    ax_p   = axes_sc_g[row_i, 1]

    for rec in sc_recs:
        dist = dist_map.get(rec['channel'])
        if dist is None or not np.isfinite(dist):
            continue
        color  = _freq_color_g[rec['freq_Hz']]
        jitter = _freq_jitter_g[rec['freq_Hz']]

        if 'latency_ms' in rec['pulses'].columns:
            for lat in rec['pulses']['latency_ms'].dropna().values:
                if np.isfinite(lat) and lat > 0:
                    ax_lat.scatter(dist + jitter, lat, color=color, s=12,
                                   alpha=0.6, linewidths=0, zorder=3)

        pv = _value_at_pulse(rec, _SCATTER_PULSE_G)
        if pv is not None and np.isfinite(pv):
            ax_p.scatter(dist + jitter, pv, color=color, s=12,
                         alpha=0.6, linewidths=0, zorder=3)

    ax_lat.set_xlim(_xlim_lat_g);  ax_lat.set_ylim(_ylim_lat_g)
    ax_p.set_xlim(_xlim_p_g);      ax_p.set_ylim(_ylim_p_g)

    for ax in (ax_lat, ax_p):
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_lat.set_xlabel('distance from stim (mm)', fontsize=7)
    ax_lat.set_ylabel('latency (ms)', fontsize=7)
    ax_lat.set_title(f'{src_label} — latency vs distance', fontsize=8, loc='left', pad=3)

    ax_p.axhline(0, color='grey', linewidth=0.7, linestyle='--', alpha=0.5)
    ax_p.set_xlabel('distance from stim (mm)', fontsize=7)
    ax_p.set_ylabel(f'(y − y₀) / y₀  at pulse {_SCATTER_PULSE_G}', fontsize=7)
    ax_p.set_title(f'{src_label} — pulse {_SCATTER_PULSE_G} vs distance',
                   fontsize=8, loc='left', pad=3)

_freq_handles_g = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=_freq_color_g[f], markersize=5,
                               label=f'{f:g} Hz')
                   for f in _all_freqs_g]
axes_sc_g[0, 0].legend(handles=_freq_handles_g, fontsize=5, loc='upper right',
                       framealpha=0.5, handlelength=0.8, labelspacing=0.3)

fig_sc_g.suptitle(
    f'Distance from stim vs latency / pulse-{_SCATTER_PULSE_G}  ·  color = frequency',
    fontsize=10,
)
plt.tight_layout()
plt.show()
plt.close(fig_sc_g)
