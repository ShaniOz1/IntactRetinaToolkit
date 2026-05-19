"""
glucose_instability_exploration_analyse.py
==========================================
Reads all *_direct_response.csv files from RESULTS_DIR (glucose experiment),
computes per-channel response efficiency, and plots a heatmap of efficiency
vs stimulation current and frequency — separated by experimental phase.

Phases are inferred from the timestamp embedded in each filename:
  Phase 1 – normal (pre)    : recordings before noon  (11:xx)
  Phase 2 – low glucose     : 12:00–12:30
  Phase 3 – normal (post)   : 12:30–13:00
  Phase 4 – normal (extended): 13:xx

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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR     = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_glucose'
MIN_AMP_MV      = 1.0   # channels with max amp < 1 mV are excluded
SEPARATE_PHASES = True  # True → one figure per phase; False → all combined


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


def phase_label(filename):
    """Map a CSV filename to its experimental phase based on the timestamp."""
    m = re.search(r'T(\d{2})-(\d{2})', filename)
    if not m:
        tag = 'low' if filename.startswith('low_') else 'normal'
        return f'{tag}_unknown'
    t = int(m.group(1)) * 60 + int(m.group(2))
    if t < 12 * 60:
        return 'Phase 1 – normal (pre)'
    elif t < 12 * 60 + 30:
        return 'Phase 2 – low glucose'
    elif t < 13 * 60:
        return 'Phase 3 – normal (post)'
    else:
        return 'Phase 4 – normal (extended)'


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
        last_resp = np.where(~np.isnan(amps))[0]
        if len(last_resp) > 0:
            amps[last_resp[-1] + 1:] = 0
        efficiency = np.nanmean(amps / max_amp)
        results.append((efficiency, max_amp))

    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_direct_response.csv')))
    if not csv_files:
        print(f'No CSV files found in {RESULTS_DIR}')
        exit()

    # {retina_label: {(current_uA, freq_Hz): [(efficiency, max_amp_mV), ...]}}
    data_by_phase: dict[str, dict[tuple, list]] = {}

    for path in csv_files:
        fname = os.path.basename(path)
        current, freq = parse_current_freq(fname)
        if current is None:
            print(f'  [skip] could not parse current/freq from: {fname}')
            continue

        results = channel_efficiencies(path)
        if not results:
            continue

        label = phase_label(fname) if SEPARATE_PHASES else 'all'
        data_by_phase.setdefault(label, {}).setdefault((current, freq), []).extend(results)

    if not data_by_phase:
        print('No qualifying channels found.')
        exit()

    # ── figure: max amplitude distributions across all channels & files ───────
    all_max_amps = []
    for path in csv_files:
        fname = os.path.basename(path)
        current, freq = parse_current_freq(fname)
        if current is None:
            continue
        all_max_amps.extend(channel_max_amplitudes(path))

    if all_max_amps:
        all_max_amps = np.array(all_max_amps)

        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 4))
        fig_dist.suptitle('Max amplitude distribution – all channels, all files', fontsize=11)

        # linear scale
        ax = axes_dist[0]
        ax.hist(all_max_amps, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
        for lo, hi, label in [(0.05, 0.25, '0.05–0.25'), (0.25, 0.5, '0.25–0.5'), (1.0, None, '>1.0')]:
            ax.axvline(lo, color='tomato', lw=1, ls='--', alpha=0.8)
            if hi is not None:
                ax.axvline(hi, color='tomato', lw=1, ls='--', alpha=0.8)
        ax.set_xlabel('Max amplitude (mV)', fontsize=10)
        ax.set_ylabel('Channel count', fontsize=10)
        ax.set_title('Linear scale', fontsize=9)

        # log x-scale to better see the spread
        ax2 = axes_dist[1]
        positive = all_max_amps[all_max_amps > 0]
        bins_log = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60)
        ax2.hist(positive, bins=bins_log, color='steelblue', edgecolor='white', linewidth=0.4)
        for lo, hi, label in [(0.05, 0.25, '0.05–0.25'), (0.25, 0.5, '0.25–0.5'), (1.0, None, '>1.0')]:
            ax2.axvline(lo, color='tomato', lw=1, ls='--', alpha=0.8,
                        label=f'{label} mV range' if hi is None else None)
            if hi is not None:
                ax2.axvline(hi, color='tomato', lw=1, ls='--', alpha=0.8)
        ax2.set_xscale('log')
        ax2.set_xlabel('Max amplitude (mV) – log scale', fontsize=10)
        ax2.set_ylabel('Channel count', fontsize=10)
        ax2.set_title('Log scale', fontsize=9)

        # shared stats annotation
        stats_txt = (f'n={len(all_max_amps)}  median={np.median(all_max_amps):.3f} mV  '
                     f'mean={np.mean(all_max_amps):.3f} mV  max={all_max_amps.max():.3f} mV')
        fig_dist.text(0.5, 0.01, stats_txt, ha='center', fontsize=8, color='dimgray')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_dist = os.path.join(RESULTS_DIR, 'max_amplitude_distribution.png')
        fig_dist.savefig(out_dist, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_dist)
        print(f'Saved → {out_dist}')

    amp_ranges = [
        (0.2, 0.5,  '0.2 – 0.5 mV'),
        (0.5, 1.0,  '0.5 – 1 mV'),
        (1.0, 1.5,  '1 – 1.5 mV'),
        (1.5, 2.0,  '1.5 – 2 mV'),
        (2.0, None, '> 2.0 mV'),
    ]

    def _plot_heatmap(data, agg_fn, agg_name, out_name, title_suffix=''):
        currents = sorted({k[0] for k in data})
        freqs    = sorted({k[1] for k in data})
        cur_idx  = {c: i for i, c in enumerate(currents)}
        freq_idx = {f: i for i, f in enumerate(freqs)}

        n_ranges = len(amp_ranges)
        n_rows = 2 if n_ranges > 3 else 1
        n_cols = math.ceil(n_ranges / n_rows)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(max(2.5 * n_cols, len(currents) * n_cols * 0.45),
                                          max(3, len(freqs) * 0.7) * n_rows))
        axes_flat = np.array(axes).flatten()
        for ax in axes_flat[n_ranges:]:
            ax.set_visible(False)

        for i, (ax, (lo, hi, label)) in enumerate(zip(axes_flat, amp_ranges)):
            matrix = np.full((len(freqs), len(currents)), np.nan)
            counts = np.zeros_like(matrix, dtype=int)

            for (cur, freq), results in data.items():
                effs = [e for e, amp in results
                        if (lo is None or amp >= lo) and (hi is None or amp < hi)]
                if not effs:
                    continue
                r, c = freq_idx[freq], cur_idx[cur]
                matrix[r, c] = agg_fn(effs)
                counts[r, c] = len(effs)

            im = ax.imshow(matrix, aspect='auto', cmap='Purples',
                           vmin=0, vmax=1, origin='lower')

            ax.set_title(label, fontsize=8, loc='left')
            ax.set_xticks(range(len(currents)))
            ax.set_xticklabels([f'{c:g} µA' for c in currents], fontsize=8)
            ax.set_yticks(range(len(freqs)))
            if i % n_cols == 0:
                ax.set_yticklabels([f'{f:g} Hz' for f in freqs], fontsize=8)
                ax.set_ylabel('Stimulation frequency', fontsize=9)
            else:
                ax.set_yticklabels([])

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
            last_im = im

        title = f'Response efficiency — {agg_name}'
        if title_suffix:
            title += f'  [{title_suffix}]'
        fig.suptitle(title, fontsize=10, y=1.01)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, right=0.88)
        pos = axes_flat[n_ranges - 1].get_position()
        cax = fig.add_axes([0.90, pos.y0, 0.02, pos.height])
        fig.colorbar(last_im, cax=cax, label=f'{agg_name} normalized amplitude')

        out_path = os.path.join(RESULTS_DIR, out_name)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'Saved → {out_path}')

    for phase, data in sorted(data_by_phase.items()):
        suffix = phase if SEPARATE_PHASES else ''
        tag    = f'_{phase.replace(" ", "_").replace("–", "-")}' if SEPARATE_PHASES else ''
        # _plot_heatmap(data, np.median, 'median', f'response_efficiency_heatmap_median{tag}.png', suffix)
        _plot_heatmap(data, np.mean, 'mean', f'response_efficiency_heatmap_mean{tag}.png', suffix)
