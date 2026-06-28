"""
combined_instability_analysis.py
=================================
Loads ex vivo and intact retina direct-response CSV data for combined
instability analysis.  No calculations are performed here — data is simply
loaded into two flat lists of records for downstream analysis.

Ex vivo  : Results/ex_vivo_all  — 2 reference channels per retina defined
           in EX_VIVO_RETINA_CHANNELS (same logic as ex_vivo_instability_exploration_analyse.py)

Intact   : Results/1hz + 10hz + 20hz — specific numeric channels per
           (retina, date) defined in INTACT_CHANNELS
           (same loading logic as intact_instability_exploration_analyse.py)

After running this script, two lists are available:
    ex_vivo_records  — one dict per (file × channel)
    intact_records   — one dict per (file × channel)

Each dict has:
    source      : 'ex_vivo' | 'intact'
    retina      : str  (e.g. '2024.11.17', 'Retina1_250525')
    current_uA  : float
    freq_Hz     : float
    channel     : str (ex vivo short name, e.g. 'G5') | int (intact numeric)
    pulses      : DataFrame with columns [pulse_index, amplitude_mV]
    filename    : source CSV filename (no path)
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

START_PULSE          = 2     # first pulse index to include (e.g. 2 → skip pulse 0 and 1)
PULSE_LIMIT          = 50    # number of pulses to include starting from START_PULSE
                             #   → kept window: [START_PULSE, START_PULSE + PULSE_LIMIT)
HIST_PULSES          = [25, 50]   # pulse indices shown as histogram rows


# ── paths ─────────────────────────────────────────────────────────────────────

EX_VIVO_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_all'

INTACT_DIRS = {
     1: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_data\1hz',
    10: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_data\10hz',
    20: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_data\20hz',
}


# ── ex vivo channel selection ─────────────────────────────────────────────────
# Keys match ex_vivo_retina_label(filename).
# Values are the 2 reference channel short names (last token of full channel
# name, upper-cased — e.g. "E_B-00071 G5" → "G5").

EX_VIVO_RETINA_CHANNELS: dict[str, set[str]] = {
    '2024.11.17':    {'G5', 'G7'},
    '2025.11.02 J6': {'K4', 'H8'},
    '2025.11.02 J9': {'K9', 'G9'},
    '2025.11.12':    {'K9', 'D11'},
}

# EX_VIVO_RETINA_CHANNELS: dict[str, set[str]] = {
#     '2024.11.17':    {'F1', 'F2', 'G1', 'G4', 'G5', 'G7', 'G8', 'H10', 'H11', 'H12'},
#     '2025.11.02 J6': {'F10', 'F11', 'G10', 'G9', 'H8', 'K3', 'K4', 'K5', 'L3'},
#     '2025.11.02 J9': {'A9', 'B9', 'C9', 'E9', 'F9', 'G9', 'L9'},
#     '2025.11.12':    {'C11', 'D11', 'H10', 'J10', 'J9', 'K9', 'L9'},
# }


# ── intact channel selection ──────────────────────────────────────────────────
# Keys: (retina_prefix, date_str)  e.g. ('Retina1', '250525')
# Values: set of integer channel numbers to keep from the CSV's 'channel' column.

INTACT_CHANNELS: dict[tuple[str, str], set[int]] = {
    ('Retina1', '250525'): {1, 6},
    ('Retina4', '250525'): {1, 6},
    ('Retina1', '250528'): {1, 6},
    ('Retina3', '250528'): {26, 5},
    ('Retina5', '250528'): {2},
}

# Some intact files are missing the "RetinaN" prefix in their filename.
# Map them explicitly: filename → (retina_prefix, date_str).
INTACT_RETINA_OVERRIDES: dict[str, tuple[str, str]] = {
    'Ch01_300us_50us_7uA_20Hz_250528_113309.rhs_direct_response.csv': ('Retina3', '250528'),
}

# ── stim electrodes & distance helpers ───────────────────────────────────────

EX_VIVO_STIM: dict[str, str] = {
    '2024.11.17':    'G6',
    '2025.11.02 J6': 'J6',
    '2025.11.02 J9': 'J9',
    '2025.11.12':    'G10',
}
_MEA_PITCH_MM = 0.2   # 200 µm per electrode step

# Intact probe: real channel → angle in degrees CCW from 3 o'clock
_CHANNEL_ANGLES: dict[int, float] = {
    26:   0,  5:  19, 25:  38,  6:  57, 24:  76,  7:  95,
    28: 114,  2: 133, 29: 152,  1: 171, 30: 189,  0: 208,
    31: 227,  3: 303, 27: 322,  4: 341,
}


def _ev_distance_mm(stim_ch: str, rec_ch: str) -> Optional[float]:
    s = mea_name_to_location(stim_ch)
    t = mea_name_to_location(rec_ch)
    if s is None or t is None:
        return None
    return float(np.sqrt((s[0]-t[0])**2 + (s[1]-t[1])**2)) * _MEA_PITCH_MM


# Hardcoded distances (mm) between intact stim and recorded channels.
# Keys are (stim_real_ch, recorded_real_ch); distance is symmetric.
_IN_DIST_MM: dict[tuple[int, int], float] = {
    (4, 1): 0.930,
    (4, 6): 0.500,
    (5, 1): 0.860,
    (5, 6): 0.256,
    (1, 26): 0.904,
    (4, 2): 0.860,
}


def _intact_dist_mm(stim_ch: int, rec_ch: int) -> Optional[float]:
    return _IN_DIST_MM.get((stim_ch, rec_ch)) or _IN_DIST_MM.get((rec_ch, stim_ch))


def _intact_stim_ch(filename: str) -> Optional[int]:
    m = re.search(r'Ch(\d+)', filename, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_current_freq(filename: str):
    """Return (current_uA, freq_Hz) parsed from filename, or (None, None)."""
    cur  = re.search(r'(\d+(?:\.\d+)?)\s*uA', filename, re.IGNORECASE)
    freq = re.search(r'(\d+(?:\.\d+)?)\s*Hz', filename, re.IGNORECASE)
    if not cur or not freq:
        return None, None
    current = float(cur.group(1))
    # Fix timestamp bleed into current value (e.g. "197uA" → 7 uA)
    if current == int(current) and int(current) % 10 == 7 and current != 7:
        current = 7.0
    return current, float(freq.group(1))


def ex_vivo_retina_label(filename: str) -> str:
    """Map an ex vivo CSV filename to a retina identifier string."""
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


def load_ex_vivo_pulses(csv_path: str, allowed_channels: set[str]) -> list[dict]:
    """
    Load pulse data for the 2 allowed channels from one ex vivo CSV.
    Copied from ex_vivo_instability_exploration_analyse.collect_raw_pulses(),
    but restricted to the specified channel short-names.
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
        if short_ch not in allowed_channels:
            continue
        cols = ['pulse_index', 'amplitude_mV']
        for _c in ('latency_ms', 'width_ms'):
            if _c in ch_df.columns:
                cols.append(_c)
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][cols]
                  .copy()
                  .reset_index(drop=True))
        records.append({'channel': short_ch, 'pulses': pulses})
    return records


def load_intact_pulses(csv_path: str, allowed_channels: set[int]) -> list[dict]:
    """
    Load pulse data for the specified numeric channels from one intact CSV.
    Copied from intact_instability_exploration_analyse.collect_raw_pulses(),
    but restricted to the specified integer channel numbers.
    Returns list of {'channel': int, 'pulses': DataFrame[pulse_index, amplitude_mV]}.
    """
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    df['channel'] = pd.to_numeric(df['channel'], errors='coerce')

    records = []
    for ch, ch_df in df.groupby('channel'):
        ch_int = int(ch)  # type: ignore[arg-type]
        if ch_int not in allowed_channels:
            continue
        cols = ['pulse_index', 'amplitude_mV']
        for _c in ('latency_ms', 'width_ms'):
            if _c in ch_df.columns:
                cols.append(_c)
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][cols]
                  .copy()
                  .reset_index(drop=True))
        records.append({'channel': ch_int, 'pulses': pulses})
    return records


# ── load ex vivo ──────────────────────────────────────────────────────────────

ex_vivo_records: list[dict] = []

for path in sorted(glob.glob(os.path.join(EX_VIVO_DIR, '*_direct_response.csv'))):
    fname   = os.path.basename(path)
    retina  = ex_vivo_retina_label(fname)
    allowed = EX_VIVO_RETINA_CHANNELS.get(retina)
    if allowed is None:
        continue

    current, freq = parse_current_freq(fname)
    if current is None:
        print(f'  [ex vivo skip] could not parse current/freq: {fname}')
        continue

    for ch_rec in load_ex_vivo_pulses(path, allowed):
        ex_vivo_records.append({
            'source':         'ex_vivo',
            'retina':         retina,
            'current_uA':     current,
            'freq_Hz':        freq,
            'channel':        ch_rec['channel'],
            'pulses':         ch_rec['pulses'],
            'filename':       fname,
            'stim_electrode': EX_VIVO_STIM.get(retina),
        })


# ── load intact ───────────────────────────────────────────────────────────────

intact_records: list[dict] = []

for freq_hz, src_dir in INTACT_DIRS.items():
    for path in sorted(glob.glob(os.path.join(src_dir, '*_direct_response.csv'))):
        fname = os.path.basename(path)

        # Resolve retina prefix + date from filename or from the override table
        if fname in INTACT_RETINA_OVERRIDES:
            retina_prefix, date_str = INTACT_RETINA_OVERRIDES[fname]
        else:
            m_retina = re.search(r'(Retina\d+)', fname, re.IGNORECASE)
            m_date   = re.search(r'_(\d{6})_', fname)
            if not m_retina or not m_date:
                print(f'  [intact skip] could not parse retina/date: {fname}')
                continue
            retina_prefix = m_retina.group(1)   # e.g. 'Retina1'
            date_str      = m_date.group(1)     # e.g. '250525'

        allowed = INTACT_CHANNELS.get((retina_prefix, date_str))
        if allowed is None:
            continue

        retina_label = f'{retina_prefix}_{date_str}'

        # Current: read from filename if present, else default to 7 µA
        m_cur   = re.search(r'(\d+(?:\.\d+)?)\s*uA', fname, re.IGNORECASE)
        current = float(m_cur.group(1)) if m_cur else 7.0

        _stim_ch = _intact_stim_ch(fname)
        for ch_rec in load_intact_pulses(path, allowed):
            intact_records.append({
                'source':     'intact',
                'retina':     retina_label,
                'current_uA': current,
                'freq_Hz':    float(freq_hz),
                'channel':    ch_rec['channel'],
                'pulses':     ch_rec['pulses'],
                'filename':   fname,
                'stim_ch':    _stim_ch,
            })


# ── summary ───────────────────────────────────────────────────────────────────

# ── pulse count for 1 Hz intact records ──────────────────────────────────────
print('Intact 1 Hz — pulse counts per file × channel:')
_1hz_recs = [r for r in intact_records if r['freq_Hz'] == 1.0]
if _1hz_recs:
    _seen = {}
    for r in _1hz_recs:
        key = (r['filename'], r['channel'])
        n   = int(r['pulses']['pulse_index'].max()) + 1
        _seen[key] = n
    for (fname, ch), n in sorted(_seen.items()):
        print(f'  {fname}  ch{ch}: {n} pulses')
else:
    print('  (none)')
print()


def _pivot_table(records: list[dict], title: str) -> None:
    """
    Print a pivot table: rows = (current_uA, freq_Hz), columns = retina,
    cell value = number of unique channels loaded.
    """
    if not records:
        print(f'\n{title}: no records loaded.')
        return

    df = pd.DataFrame([
        {'retina': r['retina'], 'current_uA': r['current_uA'],
         'freq_Hz': r['freq_Hz'], 'channel': r['channel']}
        for r in records
    ])

    # Count unique channels per (retina, current_uA, freq_Hz)
    counts = (df.groupby(['retina', 'current_uA', 'freq_Hz'])['channel']
                .nunique()
                .reset_index(name='n_channels'))

    # Build row labels: "7µA / 1Hz" style, sorted by (current, freq)
    row_keys = sorted(counts[['current_uA', 'freq_Hz']]
                      .drop_duplicates()
                      .itertuples(index=False, name=None))
    row_labels = {(c, f): f'{c:g}µA / {f:g}Hz' for c, f in row_keys}

    retinas = sorted(counts['retina'].unique())

    # Build the pivot dict
    pivot: dict[str, dict[str, str]] = {}
    for _, row in counts.iterrows():
        key = (row['current_uA'], row['freq_Hz'])
        pivot.setdefault(row_labels[key], {})[row['retina']] = str(int(row['n_channels']))

    # Column widths
    cond_w  = max(len('Condition'), max(len(lbl) for lbl in row_labels.values()))
    col_ws  = {ret: max(len(ret), 1) for ret in retinas}

    # Header
    header = f'  {"Condition":<{cond_w}}' + ''.join(f'  {ret:>{col_ws[ret]}}' for ret in retinas)
    sep    = '  ' + '─' * (len(header) - 2)

    print(f'\n{title}  (total records: {len(records)})')
    print(sep)
    print(header)
    print(sep)
    for key in row_keys:
        lbl = row_labels[key]
        row_str = f'  {lbl:<{cond_w}}'
        for ret in retinas:
            val = pivot.get(lbl, {}).get(ret, '–')
            row_str += f'  {val:>{col_ws[ret]}}'
        print(row_str)
    print(sep)


_pivot_table(ex_vivo_records, 'EX VIVO')
_pivot_table(intact_records,  'INTACT')
print()



def _has_full_window(rec: dict) -> bool:
    """Return True only if the record contains pulses reaching the end of the window."""
    pulses = rec['pulses']
    if pulses.empty:
        return False
    mx = pulses['pulse_index'].max()
    return pd.notna(mx) and int(mx) >= START_PULSE + PULSE_LIMIT - 1


def _normalise(rec: dict) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame with columns [pulse_index, norm_amp] for detected pulses,
    normalised as (amp − y0) / y0  where y0 = amplitude at pulse START_PULSE.
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

    # y0 is anchored to START_PULSE specifically
    y0 = float(per_pulse.loc[per_pulse['pulse_index'] == START_PULSE, 'amplitude_mV'].iloc[0])
    if y0 == 0:
        return None

    detected = per_pulse[per_pulse['amplitude_mV'] > 0].copy()
    if detected.empty:
        return None
    detected['norm_amp'] = (detected['amplitude_mV'] - y0) / y0
    return detected[['pulse_index', 'norm_amp']]


# ── overview figure: row 0 = all black, row 1 = coloured by (current, freq) ───

SMOOTH_WINDOW = 7   # rolling-average window for overview lines (set to 1 to disable)

_pulse_idx = list(range(START_PULSE, START_PULSE + PULSE_LIMIT))
_grid_ys   = [0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75]

fig_ov, axes_ov = plt.subplots(1, 2, figsize=(8, 3), sharey=True, sharex=True)

for col_i, (ov_records, title) in enumerate([
    (ex_vivo_records, 'Ex vivo'),
    (intact_records,  'Intact'),
]):
    ax0 = axes_ov[col_i]
    for rec in ov_records:
        if not _has_full_window(rec):
            continue
        norm_df = _normalise(rec)
        if norm_df is None:
            continue
        y_smooth = (norm_df['norm_amp']
                    .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
                    .mean())
        ax0.plot(norm_df['pulse_index'], y_smooth,
                 color='black', linewidth=0.4, alpha=0.4)

    for _y in _grid_ys:
        ax0.axhline(_y, color='grey', linewidth=0.35, alpha=0.3)

    ax0.set_xlim(START_PULSE, START_PULSE + PULSE_LIMIT)
    ax0.set_ylim(-1, 1)
    ax0.set_title(title, fontsize=10, loc='left', pad=3)
    ax0.set_xlabel('# pulse', fontsize=7)
    ax0.tick_params(labelsize=6)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

axes_ov[0].set_ylabel('(y − y₀) / y₀', fontsize=7)

plt.tight_layout()
plt.show()
plt.close(fig_ov)


# ── histograms: normalised amplitude at each pulse in HIST_PULSES ─────────────

def _value_at_pulse(rec: dict, pulse_idx: int) -> Optional[float]:
    """Return the normalised amplitude at a specific pulse index, or None.
    Returns None if the record does not cover the full PULSE_LIMIT window."""
    if not _has_full_window(rec):
        return None
    norm_df = _normalise(rec)
    if norm_df is None:
        return None
    row = norm_df[norm_df['pulse_index'] == pulse_idx]
    if row.empty:
        return None
    return float(row['norm_amp'].iloc[0])


def _value_at_pulse_relaxed(rec: dict, pulse_idx: int) -> Optional[float]:
    """Like _value_at_pulse but only requires the recording reached pulse_idx,
    not the full PULSE_LIMIT window. Used in scatter plots."""
    if int(rec['pulses']['pulse_index'].max()) < pulse_idx:
        return None
    norm_df = _normalise(rec)
    if norm_df is None:
        return None
    row = norm_df[norm_df['pulse_index'] == pulse_idx]
    if row.empty:
        return None
    return float(row['norm_amp'].iloc[0])


def _print_stats_table(rows: list[tuple[str, list[float]]], title: str) -> None:
    """Print a table of n / mean / median / std for each (label, vals) row."""
    print(f'\n{title}')
    header = f'  {"":<30}{"n":>6}{"mean":>10}{"median":>10}{"std":>10}'
    sep    = '  ' + '-' * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for label, vals in rows:
        if vals:
            arr = np.array(vals)
            print(f'  {label:<30}{len(arr):>6}{arr.mean():>10.3f}{np.median(arr):>10.3f}{arr.std():>10.3f}')
        else:
            print(f'  {label:<30}{0:>6}{"-":>10}{"-":>10}{"-":>10}')
    print(sep)


n_hist_rows = len(HIST_PULSES)
_hist_bins  = np.linspace(-1, 1, 11)   # 10 equal bins of width 0.2, shared across all subplots

fig_hist, axes_hist = plt.subplots(n_hist_rows, 2,
                                   figsize=(6, 1.6 * n_hist_rows),
                                   sharex=True, sharey=True)

_hist_table_rows: list[tuple[str, list[float]]] = []

for row_i, pulse_idx in enumerate(HIST_PULSES):
    for col_i, (h_records, title) in enumerate([
        (ex_vivo_records, 'Ex vivo'),
        (intact_records,  'Intact'),
    ]):
        ax = axes_hist[row_i][col_i]
        vals = [v for rec in h_records
                if (v := _value_at_pulse(rec, pulse_idx)) is not None
                and np.isfinite(v)]
        _hist_table_rows.append((f'{title} · pulse {pulse_idx}', vals))

        if vals:
            weights = [100.0 / len(vals)] * len(vals)
            bar_heights, _ = np.histogram(vals, bins=_hist_bins, weights=weights)
            ax.hist(vals, bins=_hist_bins, weights=weights,
                    color='#c8c8c8', edgecolor='black', linewidth=0.4, alpha=0.9)
            ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
            # KDE peak scaled to match tallest bar — keeps shape without exceeding 100%
            if len(vals) > 1:
                x_kde    = np.linspace(-1, 1, 300)
                kde_vals = gaussian_kde(vals)(x_kde)
                max_bar  = bar_heights.max()
                if kde_vals.max() > 0 and max_bar > 0:
                    kde_vals = kde_vals / kde_vals.max() * max_bar
                ax.plot(x_kde, kde_vals, color='black', linewidth=1.5)

        ax.set_xlim(-1, 1)
        if row_i == 0:
            ax.set_title(f'{title}\npulse {pulse_idx}', fontsize=7, loc='left', pad=2)
        else:
            ax.set_title(f'pulse {pulse_idx}', fontsize=7, loc='left', pad=2)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes_hist[row_i][0].set_ylabel('%', fontsize=6)

for col_i in range(2):
    axes_hist[-1][col_i].set_xlabel('(y − y₀) / y₀', fontsize=6)

plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
plt.close(fig_hist)

_print_stats_table(_hist_table_rows, 'NORMALISED AMPLITUDE AT HIST_PULSES (figure 2)')


# ── histograms: absolute amplitude (mV) at each pulse in HIST_PULSES ──────────

def _abs_value_at_pulse(rec: dict, pulse_idx: int) -> Optional[float]:
    """Return the raw amplitude (mV) at a specific pulse index, or None.
    Returns None if the record does not cover the full PULSE_LIMIT window."""
    if not _has_full_window(rec):
        return None
    per_pulse = (
        rec['pulses']
        .groupby('pulse_index')['amplitude_mV']
        .max()
        .reset_index()
    )
    row = per_pulse[per_pulse['pulse_index'] == pulse_idx]
    if row.empty:
        return None
    val = float(row['amplitude_mV'].iloc[0])
    return val if np.isfinite(val) else None


fig_hist_abs, axes_hist_abs = plt.subplots(1, 2, figsize=(6, 2.2), sharey=True)

for col_i, (h_records, title) in enumerate([
    (ex_vivo_records, 'Ex vivo'),
    (intact_records,  'Intact'),
]):
    ax   = axes_hist_abs[col_i]
    vals = [v for rec in h_records
            if (v := _abs_value_at_pulse(rec, START_PULSE)) is not None]

    if vals:
        weights        = [100.0 / len(vals)] * len(vals)
        x_max          = max(vals) * 1.05
        bins_abs       = np.linspace(0, x_max, 11)
        bar_heights, _ = np.histogram(vals, bins=bins_abs, weights=weights)
        ax.hist(vals, bins=bins_abs, weights=weights,
                color='#c8c8c8', edgecolor='black', linewidth=0.4, alpha=0.9)
        if len(vals) > 1:
            x_kde    = np.linspace(0, x_max, 300)
            kde_vals = gaussian_kde(vals)(x_kde)
            max_bar  = bar_heights.max()
            if kde_vals.max() > 0 and max_bar > 0:
                kde_vals = kde_vals / kde_vals.max() * max_bar
            ax.plot(x_kde, kde_vals, color='black', linewidth=1.5)

    if col_i == 1:   # intact: fixed x range
        ax.set_xlim(0, 1)
    ax.set_title(f'{title}  ·  first amplitude', fontsize=7, loc='left', pad=2)
    ax.set_xlabel('amplitude (mV)', fontsize=6)
    ax.tick_params(labelsize=5, length=2, pad=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes_hist_abs[0].set_ylabel('%', fontsize=6)

plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
plt.close(fig_hist_abs)


SCATTER_PULSE = 20


# ── distribution histograms: 4 metrics × 2 sources ───────────────────────────

def _speed_vals(records, is_ev):
    vals = []
    for r in records:
        ch   = r['channel']
        stim = r.get('stim_electrode') if is_ev else r.get('stim_ch')
        dist = _ev_distance_mm(stim, ch) if is_ev else _intact_dist_mm(stim, ch)
        if dist is None or not np.isfinite(dist):
            continue
        if 'latency_ms' not in r['pulses'].columns:
            continue
        for lat in r['pulses']['latency_ms'].dropna().values:
            if np.isfinite(lat) and lat > 0:
                vals.append(dist / lat)   # mm/ms = m/s
    return [v for v in vals if np.isfinite(v) and v > 0]


def _col_vals(records, col):
    vals = []
    for r in records:
        if col in r['pulses'].columns:
            vals.extend(r['pulses'][col].dropna().values.tolist())
    return [v for v in vals if np.isfinite(v) and v > 0]


def _draw_hist(ax, vals, xlabel, x_max):
    if vals:
        bins    = np.linspace(0, x_max, 40)
        weights = [100.0 / len(vals)] * len(vals)
        ax.hist(vals, bins=bins, weights=weights,
                color='#c8c8c8', edgecolor='black', linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


_metrics = [
    ('latency_ms',    'latency (ms)',   10),
    ('speed',         'speed (m/s)',     1),
    ('amplitude_mV',  'amplitude (mV)', 2),
    ('width_ms',      'width (ms)',      5),
]

_sources = [
    (ex_vivo_records, 'ex vivo', True),
    (intact_records,  'intact',  False),
]

fig_dist, axes_dist = plt.subplots(2, len(_metrics),
                                    figsize=(2.5 * len(_metrics), 5),
                                    sharey=False)

_dist_table_rows: list[tuple[str, list[float]]] = []

for row_i, (src_records, src_label, is_ev) in enumerate(_sources):
    for col_i, (col, xlabel, x_max) in enumerate(_metrics):
        ax = axes_dist[row_i, col_i]
        vals = _speed_vals(src_records, is_ev) if col == 'speed' else _col_vals(src_records, col)
        _dist_table_rows.append((f'{src_label} · {xlabel}', vals))
        _draw_hist(ax, vals, xlabel, x_max)
        if row_i == 0:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
    axes_dist[row_i, 0].set_ylabel(f'{src_label}\n[%]', fontsize=7)

plt.tight_layout()
plt.show()
plt.close(fig_dist)

_print_stats_table(_dist_table_rows, 'LATENCY / SPEED / AMPLITUDE / WIDTH DISTRIBUTIONS (figure 4)')


# ── figure 4 duplicate: 7uA sessions highlighted in light purple ────────────

def _speed_vals_split(records, is_ev):
    vals_other, vals_7uA = [], []
    for r in records:
        ch   = r['channel']
        stim = r.get('stim_electrode') if is_ev else r.get('stim_ch')
        dist = _ev_distance_mm(stim, ch) if is_ev else _intact_dist_mm(stim, ch)
        if dist is None or not np.isfinite(dist):
            continue
        if 'latency_ms' not in r['pulses'].columns:
            continue
        target = vals_7uA if r.get('current_uA') == 7.0 else vals_other
        for lat in r['pulses']['latency_ms'].dropna().values:
            if np.isfinite(lat) and lat > 0:
                target.append(dist / lat)   # mm/ms = m/s
    return ([v for v in vals_other if np.isfinite(v) and v > 0],
            [v for v in vals_7uA if np.isfinite(v) and v > 0])


def _col_vals_split(records, col):
    vals_other, vals_7uA = [], []
    for r in records:
        if col not in r['pulses'].columns:
            continue
        target = vals_7uA if r.get('current_uA') == 7.0 else vals_other
        target.extend(r['pulses'][col].dropna().values.tolist())
    return ([v for v in vals_other if np.isfinite(v) and v > 0],
            [v for v in vals_7uA if np.isfinite(v) and v > 0])


def _draw_hist_split(ax, vals_other, vals_7uA, xlabel, x_max):
    n_total = len(vals_other) + len(vals_7uA)
    if n_total:
        bins = np.linspace(0, x_max, 40)
        weights_other = [100.0 / n_total] * len(vals_other)
        weights_7uA   = [100.0 / n_total] * len(vals_7uA)
        ax.hist([vals_other, vals_7uA], bins=bins, weights=[weights_other, weights_7uA],
                stacked=True, color=['#c8c8c8', 'plum'],
                edgecolor='black', linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


fig_dist2, axes_dist2 = plt.subplots(2, len(_metrics),
                                      figsize=(2.5 * len(_metrics), 5),
                                      sharey=False)

for row_i, (src_records, src_label, is_ev) in enumerate(_sources):
    for col_i, (col, xlabel, x_max) in enumerate(_metrics):
        ax = axes_dist2[row_i, col_i]
        vals_other, vals_7uA = (_speed_vals_split(src_records, is_ev) if col == 'speed'
                                 else _col_vals_split(src_records, col))
        _draw_hist_split(ax, vals_other, vals_7uA, xlabel, x_max)
        if row_i == 0:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
    axes_dist2[row_i, 0].set_ylabel(f'{src_label}\n[%]', fontsize=7)

from matplotlib.patches import Patch
fig_dist2.legend(handles=[Patch(facecolor='#c8c8c8', edgecolor='black', label='other currents'),
                           Patch(facecolor='plum', edgecolor='black', label='7µA')],
                  loc='upper right', fontsize=7, frameon=False)

plt.tight_layout()
plt.show()
plt.close(fig_dist2)


# ── distance distribution ─────────────────────────────────────────────────────

def _dist_vals(records, is_ev):
    vals = []
    for r in records:
        ch   = r['channel']
        stim = r.get('stim_electrode') if is_ev else r.get('stim_ch')
        dist = _ev_distance_mm(stim, ch) if is_ev else _intact_dist_mm(stim, ch)
        if dist is not None and np.isfinite(dist):
            vals.append(dist)
    return vals


fig_dd, axes_dd = plt.subplots(1, 2, figsize=(7, 3), sharey=False)

for ax, records, src_label, is_ev in [
    (axes_dd[0], ex_vivo_records, 'ex vivo', True),
    (axes_dd[1], intact_records,  'intact',  False),
]:
    vals = _dist_vals(records, is_ev)
    if vals:
        x_max   = max(vals) * 1.1
        bins    = np.linspace(0, x_max, 40)
        weights = [100.0 / len(vals)] * len(vals)
        ax.hist(vals, bins=bins, weights=weights,
                color='#c8c8c8', edgecolor='black', linewidth=0.4, alpha=0.9)
        ax.set_xlim(0, x_max)
    ax.text(0.03, 0.97, src_label, transform=ax.transAxes,
            fontsize=7, va='top', ha='left', style='italic', color='dimgrey')
    ax.set_xlabel('distance from stim (mm)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes_dd[0].set_ylabel('%', fontsize=7)
plt.tight_layout()
plt.show()
plt.close(fig_dd)


# ── 4-panel scatter: distance vs latency / vs pulse-N norm ───────────────────

_tab10 = plt.get_cmap('tab10')

# Shared frequency → color mapping across both sources
_all_freqs  = sorted({r['freq_Hz'] for r in ex_vivo_records + intact_records})
_freq_color  = {f: _tab10(i % 10) for i, f in enumerate(_all_freqs)}
_n_f         = len(_all_freqs)
_freq_jitter = {f: (i - (_n_f - 1) / 2) * 0.02
                for i, f in enumerate(_all_freqs)}

# First pass: collect all data to compute shared axis limits
_all_dist_lat, _all_dist_p50 = [], []   # (dist, lat) and (dist, p50) across both sources

for _recs, _is_ev in [(ex_vivo_records, True), (intact_records, False)]:
    for _r in _recs:
        _ch   = _r['channel']
        _stim = _r.get('stim_electrode') if _is_ev else _r.get('stim_ch')
        _dist = _ev_distance_mm(_stim, _ch) if _is_ev else _intact_dist_mm(_stim, _ch)
        if _dist is None or not np.isfinite(_dist):
            continue
        if 'latency_ms' in _r['pulses'].columns:
            for _lat in _r['pulses']['latency_ms'].dropna().values:
                if np.isfinite(_lat) and _lat > 0:
                    _all_dist_lat.append((_dist, _lat))
        _p50 = _value_at_pulse_relaxed(_r, SCATTER_PULSE)
        if _p50 is not None and np.isfinite(_p50):
            _all_dist_p50.append((_dist, _p50))

_xlim_lat = (0, max(d for d, _ in _all_dist_lat) * 1.05) if _all_dist_lat else (0, 1)
_ylim_lat = (0, max(l for _, l in _all_dist_lat) * 1.05) if _all_dist_lat else (0, 10)
_xlim_p50 = (0, max(d for d, _ in _all_dist_p50) * 1.05) if _all_dist_p50 else (0, 1)
_p50_vals  = [v for _, v in _all_dist_p50]
_ylim_p50  = (min(_p50_vals) * 1.1, max(_p50_vals) * 1.1) if _p50_vals else (-1, 1)

fig_sc2, axes_sc2 = plt.subplots(2, 2, figsize=(8, 6))

_panel_cfg = [
    (ex_vivo_records, 'Ex vivo', 0, True),
    (intact_records,  'Intact',  1, False),
]

for sc_records, src_label, col_i, is_ev in _panel_cfg:
    ax_lat = axes_sc2[0, col_i]
    ax_p50 = axes_sc2[1, col_i]

    for rec in sc_records:
        ch   = rec['channel']
        stim = rec.get('stim_electrode') if is_ev else rec.get('stim_ch')
        dist = _ev_distance_mm(stim, ch) if is_ev else _intact_dist_mm(stim, ch)
        if dist is None or not np.isfinite(dist):
            continue
        color  = _freq_color[rec['freq_Hz']]
        jitter = _freq_jitter[rec['freq_Hz']]

        if 'latency_ms' in rec['pulses'].columns:
            for lat in rec['pulses']['latency_ms'].dropna().values:
                if np.isfinite(lat) and lat > 0:
                    ax_lat.scatter(dist + jitter, lat, color=color, s=12,
                                   alpha=0.6, linewidths=0, zorder=3)

        p50 = _value_at_pulse_relaxed(rec, SCATTER_PULSE)
        if p50 is not None and np.isfinite(p50):
            ax_p50.scatter(dist + jitter, p50, color=color, s=12,
                           alpha=0.6, linewidths=0, zorder=3)

    ax_lat.set_xlim(_xlim_lat)
    ax_lat.set_ylim(_ylim_lat)
    ax_p50.set_xlim(_xlim_p50)
    ax_p50.set_ylim(_ylim_p50)

    for ax in (ax_lat, ax_p50):
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_lat.set_xlabel('distance from stim (mm)', fontsize=7)
    ax_lat.set_ylabel('latency (ms)', fontsize=7)
    ax_lat.set_title(f'{src_label} — latency vs distance', fontsize=8, loc='left', pad=3)

    ax_p50.axhline(0, color='grey', linewidth=0.7, linestyle='--', alpha=0.5)
    ax_p50.set_xlabel('distance from stim (mm)', fontsize=7)
    ax_p50.set_ylabel(f'(y − y₀) / y₀  at pulse {SCATTER_PULSE}', fontsize=7)
    ax_p50.set_title(f'{src_label} — pulse {SCATTER_PULSE} vs distance', fontsize=8, loc='left', pad=3)

# Shared frequency legend on the first panel
_freq_handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=_freq_color[f], markersize=5,
                             label=f'{f:g} Hz')
                 for f in _all_freqs]
axes_sc2[0, 0].legend(handles=_freq_handles, fontsize=5, loc='upper right',
                      framealpha=0.5, handlelength=0.8, labelspacing=0.3)

fig_sc2.suptitle('Distance from stim electrode vs latency / instability  ·  color = frequency',
                 fontsize=10)
plt.tight_layout()
plt.show()
plt.close(fig_sc2)