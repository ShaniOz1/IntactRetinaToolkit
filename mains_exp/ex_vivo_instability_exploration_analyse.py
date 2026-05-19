"""
ex_vivo_instability_exploration_analyse.py
==========================================
Reads all *_direct_response.csv files from RESULTS_DIR, computes per-channel
response efficiency, and plots a heatmap of efficiency vs stimulation current
and frequency.

Response efficiency (per channel, per file):
  - Normalize each detected amplitude by the channel's max amplitude.
  - Pulses with no detected response count as 0.
  - Efficiency = mean of the normalized values across all pulses.

Only channels whose max amplitude >= MIN_AMP_MV are included.
Heatmap colour = mean efficiency across all qualifying channels and files
sharing the same (current, frequency).
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_all'
MIN_AMP_MV  = 1.0   # µV threshold: channels with max amp < 1 mV are excluded


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_current_freq(filename):
    """Return (current_uA, freq_Hz) parsed from filename, or (None, None)."""
    cur  = re.search(r'(\d+(?:\.\d+)?)\s*uA', filename, re.IGNORECASE)
    freq = re.search(r'(\d+(?:\.\d+)?)\s*Hz', filename, re.IGNORECASE)
    if not cur or not freq:
        return None, None
    current = float(cur.group(1))
    # Timestamp seconds can merge with the current value when the filename has no
    # separator, e.g. seconds "19" + "7uA" → "197uA".  Any integer ending in 7
    # that isn't exactly 7 is treated as 7 uA.
    if current == int(current) and int(current) % 10 == 7 and current != 7:
        current = 7.0
    return current, float(freq.group(1))


def channel_max_amplitudes(csv_path):
    """Return list of per-channel max amplitude (mV) across all pulses, unfiltered."""
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    maxes = []
    for _, ch_df in df.groupby('channel'):
        max_amp = ch_df.groupby('pulse_index')['amplitude_mV'].max().max()
        if not pd.isna(max_amp):
            maxes.append(max_amp)
    return maxes


def channel_efficiencies(csv_path):
    """
    Returns list of (efficiency, max_amp_mV) per qualifying channel.

    efficiency = mean(normalized_amplitudes over all pulses)
    where missing pulses are treated as 0, and each channel is
    normalised by its own max amplitude.
    """
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []

    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    n_pulses = int(df['pulse_index'].max()) + 1

    results = []
    for ch, ch_df in df.groupby('channel'):
        per_pulse = ch_df.groupby('pulse_index')['amplitude_mV'].max()
        max_amp   = per_pulse.max()
        if pd.isna(max_amp):
            continue
        amps = per_pulse.reindex(range(n_pulses), fill_value=0).values
        efficiency = (amps / max_amp).mean()
        results.append((efficiency, max_amp))

    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_direct_response.csv')))
    if not csv_files:
        print(f'No CSV files found in {RESULTS_DIR}')
        exit()

    # {(current_uA, freq_Hz): [(efficiency, max_amp_mV), ...]}
    data: dict[tuple, list] = {}

    for path in csv_files:
        fname = os.path.basename(path)
        current, freq = parse_current_freq(fname)
        if current is None:
            print(f'  [skip] could not parse current/freq from: {fname}')
            continue

        results = channel_efficiencies(path)
        if not results:
            continue

        key = (current, freq)
        data.setdefault(key, []).extend(results)

    if not data:
        print('No qualifying channels found.')
        exit()

    # ── build 3 heatmaps by amplitude range ──────────────────────────────────
    currents = sorted({k[0] for k in data})
    freqs    = sorted({k[1] for k in data})

    cur_idx  = {c: i for i, c in enumerate(currents)}
    freq_idx = {f: i for i, f in enumerate(freqs)}

    amp_ranges = [
        (0.05, 0.25, '0.05 < max amplitude < 0.25 mV'),
        (0.25, 0.5,  '0.25 < max amplitude < 0.5 mV'),
        (1.0,  None, 'max amplitude > 1 mV'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(currents) * 2.2),
                                            max(3,  len(freqs)    * 0.7)))

    for i, (ax, (lo, hi, label)) in enumerate(zip(axes, amp_ranges)):
        matrix = np.full((len(freqs), len(currents)), np.nan)
        counts = np.zeros_like(matrix, dtype=int)

        for (cur, freq), results in data.items():
            effs = [e for e, amp in results
                    if (lo is None or amp >= lo) and (hi is None or amp < hi)]
            if not effs:
                continue
            r, c = freq_idx[freq], cur_idx[cur]
            matrix[r, c] = np.mean(effs)
            counts[r, c] = len(effs)

        im = ax.imshow(matrix, aspect='auto', cmap='Purples',
                       vmin=0, vmax=1, origin='lower')

        ax.set_title(label, fontsize=8, loc='left')

        ax.set_xticks(range(len(currents)))
        ax.set_xticklabels([f'{c:g} µA' for c in currents], fontsize=8)
        ax.set_yticks(range(len(freqs)))
        if i == 0:
            ax.set_yticklabels([f'{f:g} Hz' for f in freqs], fontsize=8)
            ax.set_ylabel('Stimulation frequency', fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.set_box_aspect(1)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        for r in range(len(freqs)):
            for c in range(len(currents)):
                if not np.isnan(matrix[r, c]):
                    ax.text(c, r, f'{matrix[r, c]:.2f}\n(n={counts[r, c]})',
                            ha='center', va='center', fontsize=7,
                            color='white' if matrix[r, c] > 0.6 else 'black')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)
    # Add colorbar in a manually positioned axes so it never steals space from subplots
    pos = axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
    fig.colorbar(im, cax=cax, label='average normalized amplitude')

    out_path = os.path.join(RESULTS_DIR, 'response_efficiency_heatmap.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'Saved → {out_path}')
