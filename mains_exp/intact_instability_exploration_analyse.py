"""
intact_instability_exploration_analyse.py
==========================================
Reads all *_direct_response.csv files from Results/10hz and Results/20hz,
computes per-channel response efficiency, and plots a single heatmap of
efficiency vs stimulation current and frequency (no amplitude-range split).

Response efficiency (per channel, per file):
  - Normalize each detected amplitude by the channel's max amplitude.
  - Pulses with no detected response count as 0.
  - Efficiency = mean of the normalized values across all pulses.

Only channels whose max amplitude >= MIN_AMP_MV are included.
"""

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── choose data source ────────────────────────────────────────────────────────
# True  → read from a single flat directory; freq is parsed from each filename
# False → read from per-frequency subdirectories; freq comes from the dict key
USE_INTACT_ALL = True

INTACT_ALL_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_all3_seperate_retinas'

SOURCE_DIRS = {
     1: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\1hz',
    10: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\10hz',
    20: r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\20hz',
}

RESULTS_DIR      = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\intact'
MIN_AMP_MV       = 0
SEPARATE_RETINAS = True  # True → one figure per retina; False → all combined


# ── helpers ──────────────────────────────────────────────────────────────────

def retina_label(filename):
    """Return a unique retina identifier from the CSV filename: '<date> <RetinaX>'."""
    m_retina = re.search(r'(Retina\d+)', filename, re.IGNORECASE)
    m_date   = re.search(r'(\d{6})', filename)
    retina   = m_retina.group(1) if m_retina else 'unknown'
    date     = m_date.group(1)   if m_date   else ''
    return f'{date} {retina}' if date else retina


def channel_max_amplitudes(csv_path):
    """Return list of per-channel max amplitude (mV) across all pulses."""
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
    Returns list of (efficiency, max_amp_mV) per channel with max_amp >= MIN_AMP_MV.
    efficiency = mean(normalized_amplitudes over all pulses), missing pulses = 0.
    """
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []

    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    n_pulses = int(df['pulse_index'].max()) + 1

    results = []
    for _, ch_df in df.groupby('channel'):
        per_pulse = ch_df.groupby('pulse_index')['amplitude_mV'].max()
        max_amp   = per_pulse.max()
        if pd.isna(max_amp) or max_amp < MIN_AMP_MV:
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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # [(path, current_uA, freq_Hz), ...]
    csv_entries = []
    if USE_INTACT_ALL:
        for path in sorted(glob.glob(os.path.join(INTACT_ALL_DIR, '*_direct_response.csv'))):
            fname = os.path.basename(path)
            m_freq = re.search(r'(\d+(?:\.\d+)?)\s*Hz', fname, re.IGNORECASE)
            m_cur  = re.search(r'(\d+(?:_\d+)?)\s*uA',  fname, re.IGNORECASE)
            if m_freq is None:
                print(f'  [skip] no freq in filename: {fname}')
                continue
            freq_hz = float(m_freq.group(1))
            current = float(m_cur.group(1).replace('_', '.')) if m_cur else 7.0
            if current != 7.0:
                continue
            csv_entries.append((path, current, freq_hz))
    else:
        for freq_hz, src in SOURCE_DIRS.items():
            for path in sorted(glob.glob(os.path.join(src, '*_direct_response.csv'))):
                csv_entries.append((path, 7.0, float(freq_hz)))

    if not csv_entries:
        src_desc = INTACT_ALL_DIR if USE_INTACT_ALL else list(SOURCE_DIRS.values())
        print(f'No CSV files found in: {src_desc}')
        exit()

    print(f'Using {len(csv_entries)} CSV files.')

    # {(current_uA, freq_Hz): [(efficiency, max_amp_mV), ...]}
    data_by_retina: dict[str, dict[tuple, list]] = {}
    files_used = 0

    for path, current, freq in csv_entries:
        results = channel_efficiencies(path)
        if not results:
            continue
        label = retina_label(os.path.basename(path)) if SEPARATE_RETINAS else 'all'
        data_by_retina.setdefault(label, {}).setdefault((current, freq), []).extend(results)
        files_used += 1

    if not data_by_retina:
        print('No qualifying channels found.')
        exit()

    print(f'Files contributing to figure: {files_used} / {len(csv_entries)}')

    # ── amplitude distribution ────────────────────────────────────────────────
    all_max_amps = []
    for path, _, _ in csv_entries:
        all_max_amps.extend(channel_max_amplitudes(path))

    if all_max_amps:
        all_max_amps = np.array(all_max_amps)

        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 4))
        fig_dist.suptitle('Max amplitude distribution – all channels, all files', fontsize=11)

        ax = axes_dist[0]
        ax.hist(all_max_amps, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
        ax.axvline(MIN_AMP_MV, color='tomato', lw=1, ls='--', alpha=0.8, label=f'threshold {MIN_AMP_MV} mV')
        ax.set_xlabel('Max amplitude (mV)', fontsize=10)
        ax.set_ylabel('Channel count', fontsize=10)
        ax.set_title('Linear scale', fontsize=9)
        ax.legend(fontsize=8)

        ax2 = axes_dist[1]
        positive = all_max_amps[all_max_amps > 0]
        bins_log = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60)
        ax2.hist(positive, bins=bins_log, color='steelblue', edgecolor='white', linewidth=0.4)
        ax2.axvline(MIN_AMP_MV, color='tomato', lw=1, ls='--', alpha=0.8)
        ax2.set_xscale('log')
        ax2.set_xlabel('Max amplitude (mV) – log scale', fontsize=10)
        ax2.set_ylabel('Channel count', fontsize=10)
        ax2.set_title('Log scale', fontsize=9)

        stats_txt = (f'n={len(all_max_amps)}  median={np.median(all_max_amps):.3f} mV  '
                     f'mean={np.mean(all_max_amps):.3f} mV  max={all_max_amps.max():.3f} mV')
        fig_dist.text(0.5, 0.01, stats_txt, ha='center', fontsize=8, color='dimgray')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_dist = os.path.join(RESULTS_DIR, 'max_amplitude_distribution.png')
        # fig_dist.savefig(out_dist, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_dist)
        print(f'Saved → {out_dist}')

    # ── heatmap ───────────────────────────────────────────────────────────────
    amp_ranges = [
        (None, 0.5,  '< 0.5 mV'),
        (0.5,  None, '≥ 0.5 mV'),
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
                                 figsize=(max(1.5 * n_cols, len(currents) * n_cols * 0.45),
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

        title = f'{agg_name}'
        if title_suffix:
            title += f'  [{title_suffix}]'
        fig.suptitle(title, fontsize=6)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)

        out_path = os.path.join(RESULTS_DIR, out_name)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'Saved → {out_path}')

    for retina, data in sorted(data_by_retina.items()):
        suffix = retina if SEPARATE_RETINAS else ''
        tag    = f'_{retina.replace(" ", "_")}' if SEPARATE_RETINAS else ''
        # _plot_heatmap(data, np.median, 'median', f'response_efficiency_heatmap_median{tag}.png', suffix)
        _plot_heatmap(data, np.mean, 'mean', f'response_efficiency_heatmap_mean{tag}.png', suffix)
