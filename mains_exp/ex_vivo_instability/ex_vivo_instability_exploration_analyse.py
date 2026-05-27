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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR      = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_all'
MIN_AMP_MV       = 1.0   # channels with max amp < 1 mV are excluded
SEPARATE_RETINAS = False  # True → one figure per retina; False → all combined
PULSE_LIMIT      = 98    # number of pulses used for the norm-sum-100 metric

# Two reference channels per retina (matched against the last token of the
# full channel name, e.g. "E_B-00071 G5" → "G5").
RETINA_CHANNELS: dict[str, set[str]] = {
    '2024.11.17':    {'G5', 'G7'},
    '2025.11.02 J6': {'K4', 'H8'},
    '2025.11.02 J9': {'K9', 'G9'},
    '2025.11.12':    {'K9', 'D11'},
}


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


def retina_label(filename):
    """Map a CSV filename to a retina identifier."""
    if '2024-11-17' in filename:
        return '2024.11.17'
    if '2024-11-20' in filename:
        return '2024.11.20'
    if '2025-11-02' in filename:
        if 'J6' in filename:
            return '2025.11.02 J6'
        if 'J9' in filename:
            return '2025.11.02 J9'
        return '2025.11.02'
    if '2025-11-12' in filename:
        return '2025.11.12'
    return 'unknown'


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


def collect_raw_pulses(csv_path, allowed_channels=None):
    """Return DataFrame (channel, pulse_index, amplitude_mV) for qualifying channels."""
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()
    if allowed_channels is not None:
        df = df[df['channel'].apply(lambda c: c.split()[-1].upper() in allowed_channels)]
    return df[['channel', 'pulse_index', 'amplitude_mV']]


def channel_efficiencies(csv_path, allowed_channels=None, pulse_limit=None):
    """
    Returns list of (efficiency, max_amp_mV) per qualifying channel.

    Two modes controlled by pulse_limit:
      pulse_limit=None : efficiency = mean(norm_amps) over all pulses,
          with zeros after the last detected response zeroed out.
      pulse_limit=N    : efficiency = sum(norm_amps[:N]) / N, using only
          the first N pulses (missing pulses filled with 0).

    allowed_channels : if given, only channels whose short name
        (last whitespace-separated token, uppercased) are included.
    """
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []

    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()

    if allowed_channels is not None:
        df = df[df['channel'].apply(lambda c: c.split()[-1].upper() in allowed_channels)]
        if df.empty:
            return []

    n_pulses = int(df['pulse_index'].max()) + 1

    results = []
    for ch, ch_df in df.groupby('channel'):
        per_pulse = ch_df.groupby('pulse_index')['amplitude_mV'].max()
        max_amp   = per_pulse.max()
        if pd.isna(max_amp):
            continue

        if pulse_limit is not None:
            amps = per_pulse.reindex(range(pulse_limit), fill_value=0).values
            efficiency = np.sum(amps / max_amp) / pulse_limit
        else:
            amps = per_pulse.reindex(range(n_pulses), fill_value=0).values
            last_resp = np.where(~np.isnan(amps))[0]
            if len(last_resp) > 0:
                amps[last_resp[-1] + 1:] = 0
            efficiency = np.nanmean(amps / max_amp)

        results.append((efficiency, max_amp))

    return results


def channel_norm_slope(csv_path, allowed_channels=None):
    """
    Returns list of (decay_pct, max_amp_mV) per qualifying channel.
    decay_pct = (1 - mean(last 3 detected amps) / mean(first 3 detected amps)) * 100,
    using only the first 100 pulses (indices 0–99).
    Channels with fewer than 99 pulses (max pulse_index < 98) are skipped.
    Channels with fewer than 3 detected pulses in the window are skipped.
    """
    df = pd.read_csv(csv_path)
    if df.empty or 'amplitude_mV' not in df.columns or 'pulse_index' not in df.columns:
        return []

    df = df.dropna(subset=['pulse_index']).copy()
    df['amplitude_mV'] = df['amplitude_mV'].abs()

    if allowed_channels is not None:
        df = df[df['channel'].apply(lambda c: c.split()[-1].upper() in allowed_channels)]
        if df.empty:
            return []

    results = []
    for ch, ch_df in df.groupby('channel'):
        per_pulse = ch_df.groupby('pulse_index')['amplitude_mV'].max()

        # require at least 99 pulses (indices 0–98)
        if per_pulse.index.max() < 98:
            continue

        per_pulse = per_pulse[per_pulse.index < 100]
        detected  = per_pulse.dropna()
        detected  = detected[detected > 0].sort_index()
        if len(detected) < 3:
            continue

        mean_first = float(detected.iloc[:3].mean())
        if mean_first == 0:
            continue
        mean_last = float(detected.iloc[-3:].mean())
        decay_pct = (1 - mean_last / mean_first) * 100
        results.append((decay_pct, float(per_pulse.max())))

    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*_direct_response.csv')))
    if not csv_files:
        print(f'No CSV files found in {RESULTS_DIR}')
        exit()

    # {retina_label: {(current_uA, freq_Hz): [(efficiency, max_amp_mV), ...]}}
    data_by_retina: dict[str, dict[tuple, list]] = {}
    # same structure but efficiency = sum(norm_amps[:PULSE_LIMIT]) / PULSE_LIMIT
    data_by_retina_norm100: dict[str, dict[tuple, list]] = {}
    # same structure but value = max(detected_amps) - min(detected_amps) in mV
    data_by_retina_cv: dict[str, dict[tuple, list]] = {}
    # {retina_label: {(current_uA, freq_Hz): [filename, ...]}}
    files_by_retina: dict[str, dict[tuple, list]] = {}
    # [(retina, short_channel, current, freq, pulse_df)] for scatter figure
    pulse_entries = []

    for path in csv_files:
        fname = os.path.basename(path)
        current, freq = parse_current_freq(fname)
        if current is None:
            print(f'  [skip] could not parse current/freq from: {fname}')
            continue

        label   = retina_label(fname) if SEPARATE_RETINAS else 'all'
        allowed = RETINA_CHANNELS.get(retina_label(fname))
        results = channel_efficiencies(path, allowed_channels=allowed)
        if not results:
            continue
        data_by_retina.setdefault(label, {}).setdefault((current, freq), []).extend(results)
        files_by_retina.setdefault(label, {}).setdefault((current, freq), []).append(fname)

        results_n100 = channel_efficiencies(path, allowed_channels=allowed, pulse_limit=PULSE_LIMIT)
        data_by_retina_norm100.setdefault(label, {}).setdefault((current, freq), []).extend(results_n100)

        results_cv = channel_norm_slope(path, allowed_channels=allowed)
        data_by_retina_cv.setdefault(label, {}).setdefault((current, freq), []).extend(results_cv)

        raw = collect_raw_pulses(path, allowed_channels=allowed)
        for ch, ch_df in raw.groupby('channel'):
            short_ch = ch.split()[-1].upper()
            pulse_entries.append((retina_label(fname), short_ch, current, freq, ch_df.copy()))

    if not data_by_retina:
        print('No qualifying channels found.')
        exit()

    # ── summary print ─────────────────────────────────────────────────────────
    for retina, combos in sorted(files_by_retina.items()):
        all_files   = {f for flist in combos.values() for f in flist}
        retina_set  = {retina_label(f) for f in all_files} if SEPARATE_RETINAS else {retina_label(f) for f in all_files}
        print(f'\nFigure: {retina}')
        print(f'  Files included : {len(all_files)}')
        print(f'  Retinas        : {len(retina_set)} ({", ".join(sorted(retina_set))})')
        for (cur, freq), fnames in sorted(combos.items()):
            print(f'  {cur:g} µA  {freq:g} Hz  ({len(fnames)} file{"s" if len(fnames) != 1 else ""}):')
            for f in sorted(fnames):
                print(f'    {f}')

    # ── figure: max amplitude distributions across all channels & files ───────
    amps_by_retina: dict[str, list] = {}
    for path in csv_files:
        fname = os.path.basename(path)
        if parse_current_freq(fname)[0] is None:
            continue
        rlabel = retina_label(fname)
        amps_by_retina.setdefault(rlabel, []).extend(channel_max_amplitudes(path))

    all_max_amps = [a for amps in amps_by_retina.values() for a in amps]

    if all_max_amps:
        all_max_amps = np.array(all_max_amps)

        pastel_colors = ['#AEC6CF', '#FFB7C5', '#B5EAD7', '#FFDAC1', '#D4B8E0']
        retina_order  = sorted(amps_by_retina.keys())
        color_map     = {r: pastel_colors[i % len(pastel_colors)]
                         for i, r in enumerate(retina_order)}

        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(12, 4))
        fig_dist.suptitle('Max amplitude distribution – all channels, all files', fontsize=11)

        positive_all = all_max_amps[all_max_amps > 0]
        bins_lin = np.linspace(0, positive_all.max(), 121)
        bins_log = np.logspace(np.log10(positive_all.min()), np.log10(positive_all.max()), 60)

        for ax, bins, xscale, xtitle in [
            (axes_dist[0], bins_lin, 'linear', 'Max amplitude (mV)'),
            (axes_dist[1], bins_log, 'log',    'Max amplitude (mV) – log scale'),
        ]:
            for rlabel in retina_order:
                vals = np.array(amps_by_retina[rlabel])
                vals = vals[vals > 0] if xscale == 'log' else vals
                ax.hist(vals, bins=bins, color=color_map[rlabel], edgecolor='white',
                        linewidth=0.3, alpha=0.75, label=rlabel)
            for lo, hi in [(0.05, 0.25), (0.25, 0.5), (1.0, None)]:
                ax.axvline(lo, color='tomato', lw=1, ls='--', alpha=0.8)
                if hi is not None:
                    ax.axvline(hi, color='tomato', lw=1, ls='--', alpha=0.8)
            if xscale == 'log':
                ax.set_xscale('log')
            ax.set_xlabel(xtitle, fontsize=10)
            ax.set_ylabel('Channel count', fontsize=10)
            ax.set_title('Linear scale' if xscale == 'linear' else 'Log scale', fontsize=9)

        axes_dist[1].legend(fontsize=7, frameon=False)

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
        (0.2, None,  ' '),
        # (0.5, 1.0,  '0.5 – 1 mV'),
        # (1.0, 1.5,  '1 – 1.5 mV'),
        # (1.5, 2.0,  '1.5 – 2 mV'),
        # (2.0, None, '> 2.0 mV'),
    ]

    def _plot_heatmap(data, agg_fn, agg_name, out_name, title_suffix='', vmax=1):
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

            _vmax = float(np.nanmax(matrix)) if vmax is None else vmax
            im = ax.imshow(matrix, aspect='auto', cmap='Purples',
                           vmin=0, vmax=_vmax, origin='lower')

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
                                color='white' if matrix[r, c] / _vmax > 0.6 else 'black')
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

    for retina, data in sorted(data_by_retina.items()):
        suffix   = retina if SEPARATE_RETINAS else ''
        tag      = f'_{retina.replace(" ", "_")}' if SEPARATE_RETINAS else ''
        # _plot_heatmap(data, np.median, 'median', f'response_efficiency_heatmap_median{tag}.png', suffix)
        # _plot_heatmap(data, np.mean, 'mean', f'response_efficiency_heatmap_mean{tag}.png', suffix)

    # normalise norm100 efficiencies to [0, global_max] before plotting
    all_norm100_effs = [e for rd in data_by_retina_norm100.values()
                        for entries in rd.values() for e, _ in entries]
    global_max_norm100 = max(all_norm100_effs) if all_norm100_effs else 1.0
    if global_max_norm100 > 0:
        data_by_retina_norm100 = {
            label: {
                combo: [(e / global_max_norm100, amp) for e, amp in entries]
                for combo, entries in rd.items()
            }
            for label, rd in data_by_retina_norm100.items()
        }

    for retina, data in sorted(data_by_retina_norm100.items()):
        suffix = retina if SEPARATE_RETINAS else ''
        tag    = f'_{retina.replace(" ", "_")}' if SEPARATE_RETINAS else ''
        # _plot_heatmap(data, np.mean, f'norm-sum-{PULSE_LIMIT}',
        #               f'response_efficiency_heatmap_norm{PULSE_LIMIT}{tag}.png', suffix)

    for retina, data in sorted(data_by_retina_cv.items()):
        suffix = retina if SEPARATE_RETINAS else ''
        tag    = f'_{retina.replace(" ", "_")}' if SEPARATE_RETINAS else ''
        _plot_heatmap(data, np.mean, 'amplitude decay (%)',
                      f'response_efficiency_heatmap_cv{tag}.png', suffix,
                      vmax=100)

    # ── amplitude decay line plot: x=freq, y=−decay_pct, colour=current ──────
    all_currents_cv = sorted({cur
                               for rd in data_by_retina_cv.values()
                               for (cur, _) in rd})
    _tab = plt.cm.tab10(np.linspace(0, 0.9, max(len(all_currents_cv), 1)))
    cur_color_cv = {c: _tab[i] for i, c in enumerate(all_currents_cv)}

    for retina, data in sorted(data_by_retina_cv.items()):
        suffix = retina if SEPARATE_RETINAS else ''
        tag    = f'_{retina.replace(" ", "_")}' if SEPARATE_RETINAS else ''

        freqs_cv    = sorted({freq for (_, freq) in data})
        currents_cv = sorted({cur  for (cur, _)  in data})

        fig_cv, ax_cv = plt.subplots(figsize=(6, 4))

        for cur in currents_cv:
            xs, ys, errs = [], [], []
            for freq in freqs_cv:
                vals = [-v for v, _ in data.get((cur, freq), [])]
                if not vals:
                    continue
                xs.append(freq)
                ys.append(float(np.mean(vals)))
                errs.append(float(np.std(vals)))
            if not xs:
                continue
            ax_cv.errorbar(xs, ys, yerr=errs,
                           color=cur_color_cv[cur], linewidth=2.5,
                           marker='o', markersize=7, capsize=5, capthick=1.5,
                           label=f'{cur:g} µA', zorder=3)

        ax_cv.axhline(0, color='grey', linewidth=0.8, linestyle='--')
        ax_cv.set_xticks(freqs_cv)
        ax_cv.set_xticklabels([f'{f:g}' for f in freqs_cv], fontsize=9)
        ax_cv.set_xlabel('Frequency (Hz)', fontsize=10)
        ax_cv.set_ylabel('−Amplitude decay (%)', fontsize=10)
        title = 'Amplitude stability (first 3 vs last 3 pulses)'
        if suffix:
            title += f'  [{suffix}]'
        ax_cv.set_title(title, fontsize=10)
        ax_cv.legend(fontsize=9, frameon=False)
        ax_cv.spines['top'].set_visible(False)
        ax_cv.spines['right'].set_visible(False)

        plt.tight_layout()
        out_cv = os.path.join(RESULTS_DIR, f'amplitude_decay_lineplot{tag}.png')
        fig_cv.savefig(out_cv, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_cv)
        print(f'Saved → {out_cv}')

    # ── pulse # vs amplitude scatter ──────────────────────────────────────────
    if pulse_entries:
        # sort: retina → channel → current → freq
        pulse_entries.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        pastel_colors = ['#AEC6CF', '#FFB7C5', '#B5EAD7', '#FFDAC1', '#D4B8E0']
        retina_order  = sorted({e[0] for e in pulse_entries})
        retina_color  = {r: pastel_colors[i % len(pastel_colors)]
                         for i, r in enumerate(retina_order)}

        n_sub  = len(pulse_entries)
        n_cols = min(n_sub, 10)
        n_rows = math.ceil(n_sub / n_cols)

        fig_sc, axes_sc = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 1.4, n_rows * 1.4),
            squeeze=False,
        )

        for idx, (retina, short_ch, cur, freq, ch_df) in enumerate(pulse_entries):
            ax = axes_sc[idx // n_cols][idx % n_cols]
            ax.set_facecolor(retina_color[retina])
            ch_sub    = ch_df[ch_df['pulse_index'] < PULSE_LIMIT].dropna(subset=['amplitude_mV'])
            ch_sub    = ch_sub[ch_sub['amplitude_mV'] > 0]
            if ch_sub.empty:
                continue
            n_pulses  = int(ch_sub['pulse_index'].max()) + 1
            per_pulse = (ch_sub.groupby('pulse_index')['amplitude_mV'].max()
                         .reindex(range(n_pulses), fill_value=0))
            first_amp = per_pulse[per_pulse > 0].iloc[0] if (per_pulse > 0).any() else None
            if first_amp is None or first_amp == 0:
                continue
            norm_vals = (per_pulse.values - first_amp) / first_amp
            ax.scatter(per_pulse.index, norm_vals,
                       s=2, color='black', linewidths=0)
            ax.set_title(f'{retina}\n{short_ch}  {cur:g}µA {freq:g}Hz',
                         fontsize=4.5, pad=2)
            ax.set_xticks([0, PULSE_LIMIT // 2, PULSE_LIMIT])
            ax.set_xticklabels([0, PULSE_LIMIT // 2, PULSE_LIMIT], fontsize=4)
            ax.set_ylim(-1, 1)
            ax.axhline(0, color='grey', linewidth=0.3, linestyle='--')
            ax.tick_params(axis='y', labelsize=4, length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

        for idx in range(n_sub, n_rows * n_cols):
            axes_sc[idx // n_cols][idx % n_cols].set_visible(False)

        fig_sc.text(0.5, 0.01, 'pulse #', ha='center', fontsize=7)
        fig_sc.text(0.01, 0.5, '(amp − first amp) / first amp', va='center', rotation='vertical', fontsize=7)

        plt.tight_layout(rect=[0.03, 0.03, 1, 1])
        plt.subplots_adjust(wspace=0.15, hspace=0.55)

        out_sc = os.path.join(RESULTS_DIR, 'pulse_vs_amplitude.png')
        fig_sc.savefig(out_sc, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_sc)
        print(f'Saved → {out_sc}')

    # ── pulse vs amplitude: 4 subplots (one per retina), color=current, style=freq ──
    if pulse_entries:
        retinas  = sorted({e[0] for e in pulse_entries})
        currents = sorted({e[2] for e in pulse_entries})
        freqs    = sorted({e[3] for e in pulse_entries})

        freq_colors = plt.cm.tab10(np.linspace(0, 0.9, len(freqs)))
        freq_color  = {f: freq_colors[i] for i, f in enumerate(freqs)}

        line_styles = ['-', '--', ':', '-.']
        cur_style   = {c: line_styles[i % len(line_styles)] for i, c in enumerate(currents)}

        fig_pf, axes_pf = plt.subplots(
            1, len(retinas),
            figsize=(len(retinas) * 3.2, 3.0),
            squeeze=False,
        )

        for col, retina in enumerate(retinas):
            ax = axes_pf[0][col]
            ax.set_title(retina, fontsize=8, pad=4)

            for cur in currents:
                for freq in freqs:
                    matching = [e for e in pulse_entries
                                if e[0] == retina and e[2] == cur and e[3] == freq]
                    for _, _, _, _, ch_df in matching:
                        srt = ch_df[ch_df['pulse_index'] < PULSE_LIMIT].sort_values('pulse_index')
                        ax.plot(srt['pulse_index'], srt['amplitude_mV'],
                                color=freq_color[freq], linestyle=cur_style[cur],
                                linewidth=1.0, alpha=0.85)

            ax.set_xlabel('pulse #', fontsize=7)
            if col == 0:
                ax.set_ylabel('amplitude (mV)', fontsize=7)
            ax.tick_params(labelsize=6, length=3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

        # legend: freqs (color) + currents (style) on last subplot
        freq_handles = [plt.Line2D([0], [0], color=freq_color[f], linewidth=1.5,
                                   label=f'{f:g} Hz') for f in freqs]
        cur_handles  = [plt.Line2D([0], [0], color='black', linewidth=1.0,
                                   linestyle=cur_style[c], label=f'{c:g} µA')
                        for c in currents]
        axes_pf[0][-1].legend(handles=cur_handles + freq_handles,
                               fontsize=6, frameon=False, loc='upper right')

        plt.tight_layout()
        out_pf = os.path.join(RESULTS_DIR, 'pulse_vs_amp_by_retina_current.png')
        fig_pf.savefig(out_pf, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_pf)
        print(f'Saved → {out_pf}')

    # ── first amplitude: rows=channels, cols=retinas, color=current ───────────
    if pulse_entries:
        # collect first detected amplitude per (retina, channel, current, freq)
        first_amp_rows = []
        for retina, short_ch, cur, freq, ch_df in pulse_entries:
            min_pulse = ch_df['pulse_index'].min()
            val = ch_df.loc[ch_df['pulse_index'] == min_pulse, 'amplitude_mV']
            if val.empty:
                continue
            first_amp_rows.append({
                'retina':  retina,
                'channel': short_ch,
                'current': cur,
                'freq':    freq,
                'amp':     float(val.iloc[0]),
            })

        if first_amp_rows:
            fa_df = pd.DataFrame(first_amp_rows)

            retinas  = sorted(fa_df['retina'].unique())
            currents = sorted(fa_df['current'].unique())
            freqs    = sorted(fa_df['freq'].unique())

            # two channel slots per retina (sorted alphabetically)
            ch_by_retina = {r: sorted(fa_df[fa_df['retina'] == r]['channel'].unique())
                            for r in retinas}
            n_ch_slots   = max(len(v) for v in ch_by_retina.values())

            cur_colors = plt.cm.tab10(np.linspace(0, 0.9, len(currents)))
            cur_color  = {c: cur_colors[i] for i, c in enumerate(currents)}

            fig_fa, axes_fa = plt.subplots(
                n_ch_slots, len(retinas),
                figsize=(len(retinas) * 2.8, n_ch_slots * 2.4),
                squeeze=False,
            )

            for col, retina in enumerate(retinas):
                axes_fa[0][col].set_title(retina, fontsize=8, pad=4)
                channels = ch_by_retina[retina]

                for row, ch in enumerate(channels):
                    ax = axes_fa[row][col]
                    sub = fa_df[(fa_df['retina'] == retina) & (fa_df['channel'] == ch)]

                    for cur in currents:
                        csub = sub[sub['current'] == cur].sort_values('freq')
                        if csub.empty:
                            continue
                        ax.scatter(csub['freq'], csub['amp'],
                                   color=cur_color[cur], s=20, zorder=3)
                        ax.plot(csub['freq'], csub['amp'],
                                color=cur_color[cur], linewidth=0.8, zorder=2)

                    if col == 0:
                        ax.set_ylabel(f'{ch}\namp (mV)', fontsize=7)
                    else:
                        ax.set_ylabel(ch, fontsize=7)
                    ax.set_xticks(freqs)
                    ax.set_xticklabels([f'{f:g}' for f in freqs], fontsize=6)
                    ax.tick_params(axis='y', labelsize=6)
                    if row == n_ch_slots - 1:
                        ax.set_xlabel('Frequency (Hz)', fontsize=7)
                    ax.set_ylim(0, 8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)

                # hide unused rows for retinas with fewer channels
                for row in range(len(channels), n_ch_slots):
                    axes_fa[row][col].set_visible(False)

            handles = [plt.Line2D([0], [0], color=cur_color[c], marker='o',
                                  markersize=4, linewidth=1, label=f'{c:g} µA')
                       for c in currents]
            axes_fa[0][-1].legend(handles=handles, fontsize=6, frameon=False,
                                  loc='upper right')

            plt.tight_layout()
            out_fa = os.path.join(RESULTS_DIR, 'first_amplitude_vs_freq.png')
            fig_fa.savefig(out_fa, dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig_fa)
            print(f'Saved → {out_fa}')

    # ── amplitude vs pulse # per frequency ───────────────────────────────────
    if pulse_entries:
        freqs_all = sorted({e[3] for e in pulse_entries})
        fig_fq, axes_fq = plt.subplots(
            1, len(freqs_all),
            figsize=(len(freqs_all) * 3.0, 2.8),
            squeeze=False,
        )
        for col, freq in enumerate(freqs_all):
            ax = axes_fq[0][col]
            for retina, short_ch, cur, f, ch_df in pulse_entries:
                if f != freq:
                    continue
                detected = ch_df[ch_df['amplitude_mV'] > 0]
                if detected.empty:
                    continue
                per_pulse = (detected.groupby('pulse_index')['amplitude_mV'].max()
                             .reindex(range(PULSE_LIMIT), fill_value=0))
                first_amp = per_pulse[per_pulse > 0].iloc[0] if (per_pulse > 0).any() else 0
                if first_amp == 0:
                    continue
                ax.plot(per_pulse.index, per_pulse.values / first_amp,
                        linewidth=0.4, alpha=0.5, color='black')
            ax.set_title(f'{freq:g} Hz', fontsize=8)
            ax.set_xlabel('pulse #', fontsize=7)
            ax.set_xlim(0, PULSE_LIMIT)
            if col == 0:
                ax.set_ylabel('amplitude / first amplitude', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out_fq = os.path.join(RESULTS_DIR, 'amp_vs_pulse_per_freq.png')
        fig_fq.savefig(out_fq, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig_fq)
        print(f'Saved → {out_fq}')
