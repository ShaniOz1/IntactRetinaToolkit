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


# ── analysis parameters ───────────────────────────────────────────────────────

START_PULSE          = 2     # first pulse index to include (e.g. 2 → skip pulse 0 and 1)
PULSE_LIMIT          = 50    # number of pulses to include starting from START_PULSE
                             #   → kept window: [START_PULSE, START_PULSE + PULSE_LIMIT)
HIST_PULSES          = [10, 20, 30, 40, 50]   # pulse indices shown as histogram rows


# ── paths ─────────────────────────────────────────────────────────────────────

EX_VIVO_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_all'

INTACT_DIRS = {
     1: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\1hz',
    10: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\10hz',
    20: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\20hz',
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
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][['pulse_index', 'amplitude_mV']]
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
        pulses = (ch_df[
                      (ch_df['pulse_index'] >= START_PULSE) &
                      (ch_df['pulse_index'] <  START_PULSE + PULSE_LIMIT)
                  ][['pulse_index', 'amplitude_mV']]
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
            'source':     'ex_vivo',
            'retina':     retina,
            'current_uA': current,
            'freq_Hz':    freq,
            'channel':    ch_rec['channel'],
            'pulses':     ch_rec['pulses'],
            'filename':   fname,
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

        for ch_rec in load_intact_pulses(path, allowed):
            intact_records.append({
                'source':     'intact',
                'retina':     retina_label,
                'current_uA': current,
                'freq_Hz':    float(freq_hz),
                'channel':    ch_rec['channel'],
                'pulses':     ch_rec['pulses'],
                'filename':   fname,
            })


# ── summary ───────────────────────────────────────────────────────────────────

def _pivot_table(records: list[dict], title: str) -> None:
    """
    Print a pivot table: rows = retina, columns = (current_uA, freq_Hz),
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

    # Build column labels: "7µA / 1Hz" style, sorted by (current, freq)
    col_keys = sorted(counts[['current_uA', 'freq_Hz']]
                      .drop_duplicates()
                      .itertuples(index=False, name=None))
    col_labels = {(c, f): f'{c:g}µA / {f:g}Hz' for c, f in col_keys}

    retinas = sorted(counts['retina'].unique())

    # Build the pivot dict
    pivot: dict[str, dict[str, str]] = {}
    for _, row in counts.iterrows():
        key = (row['current_uA'], row['freq_Hz'])
        pivot.setdefault(row['retina'], {})[col_labels[key]] = str(int(row['n_channels']))

    # Column widths
    ret_w   = max(len('Retina'), max(len(r) for r in retinas))
    col_ws  = {lbl: max(len(lbl), 1) for lbl in col_labels.values()}

    # Header
    header = f'  {"Retina":<{ret_w}}' + ''.join(f'  {lbl:>{col_ws[lbl]}}' for lbl in col_labels.values())
    sep    = '  ' + '─' * (len(header) - 2)

    print(f'\n{title}  (total records: {len(records)})')
    print(sep)
    print(header)
    print(sep)
    for retina in retinas:
        row_str = f'  {retina:<{ret_w}}'
        for lbl in col_labels.values():
            val = pivot.get(retina, {}).get(lbl, '–')
            row_str += f'  {val:>{col_ws[lbl]}}'
        print(row_str)
    print(sep)


_pivot_table(ex_vivo_records, 'EX VIVO')
_pivot_table(intact_records,  'INTACT')
print()



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


# ── overview figure: all normalised traces ────────────────────────────────────

SMOOTH_WINDOW = 7   # rolling-average window for overview lines (set to 1 to disable)

_pulse_idx = list(range(START_PULSE, START_PULSE + PULSE_LIMIT))
_grid_ys   = [0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75]

fig_ov, (ax_ev, ax_in) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

for ax, ov_records, title in [
    (ax_ev, ex_vivo_records, 'Ex vivo'),
    (ax_in, intact_records,  'Intact'),
]:
    for rec in ov_records:
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

    ax.set_xlim(START_PULSE, START_PULSE + PULSE_LIMIT)
    ax.set_ylim(-1, 1)
    ax.set_title(title, fontsize=10, loc='left', pad=3)
    ax.set_xlabel('# pulse', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_ev.set_ylabel('(y − y₀) / y₀', fontsize=7)

plt.tight_layout()
plt.show()
plt.close(fig_ov)


# ── histograms: normalised amplitude at each pulse in HIST_PULSES ─────────────

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
_hist_bins  = np.linspace(-1, 1, 11)   # 10 equal bins of width 0.2, shared across all subplots

fig_hist, axes_hist = plt.subplots(n_hist_rows, 2,
                                   figsize=(6, 1.6 * n_hist_rows),
                                   sharex=True, sharey=True)

for row_i, pulse_idx in enumerate(HIST_PULSES):
    for col_i, (h_records, title) in enumerate([
        (ex_vivo_records, 'Ex vivo'),
        (intact_records,  'Intact'),
    ]):
        ax = axes_hist[row_i][col_i]
        vals = [v for rec in h_records
                if (v := _value_at_pulse(rec, pulse_idx)) is not None
                and np.isfinite(v)]

        if vals:
            bin_w   = _hist_bins[1] - _hist_bins[0]   # 0.2
            weights = [100.0 / len(vals)] * len(vals)
            ax.hist(vals, bins=_hist_bins, weights=weights,
                    color='#c8c8c8', edgecolor='black', linewidth=0.4, alpha=0.9)
            ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
            # KDE scaled to match the % y-axis: density × bin_width × 100
            if len(vals) > 1:
                x_kde = np.linspace(-1, 1, 300)
                kde   = gaussian_kde(vals)
                ax.plot(x_kde, kde(x_kde) * bin_w * 100,
                        color='black', linewidth=1.5)

        ax.set_xlim(-1, 1)
        ax.set_title(f'{title}  ·  pulse {pulse_idx}', fontsize=7, loc='left', pad=2)
        ax.tick_params(labelsize=5, length=2, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes_hist[row_i][0].set_ylabel('%', fontsize=6)

for col_i in range(2):
    axes_hist[-1][col_i].set_xlabel('(y − y₀) / y₀', fontsize=6)

plt.tight_layout(h_pad=0.5, w_pad=0.5)
plt.show()
plt.close(fig_hist)