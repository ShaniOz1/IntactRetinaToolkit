"""
IntactRetinaToolkit — main.py
==============================
Loads one Intan (.rhs) and one MEA (.edf) recording and runs analysis.
Edit the params below and run:
    python main.py
"""

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

from dataobj import load_rhs, load_edf
from dataviz.viz import *
from datahelper.statistics import compare_direct_responses
from dataobj.analysis.direct import _compute_threshold

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Results', 'ex_vivo_all')

# ============================================================
#  PARAMS
# ============================================================

# --- Direct response ---
DIRECT_WIN_MS       = 10.0
BLANK_MS            = 1.5
DIRECT_THRESHOLD_MV = 0.1  # set to None to compute threshold from data
INTERACTIVE         = True  # set to True to review/adjust threshold per file before saving
selected_channels =  None
# =============================== =============================
#  FILES TO RUN  (path, stim_electrode)
# ============================================================

FILES = [
    # ── Group 2 · 2024.11.17 Direct Response - Fading · G6 ──────────────────
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T12-53-10No_noise_stimulation_4uA_1Hz_200pulses_B-00071.edf',            'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T12-57-14No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',           'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-02-36No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',           'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-05-52Noise_1-10Hz_0.2mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-07-07Noise_1-10Hz_0.1mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-08-03No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',           'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-09-21Noise_1-10Hz_0.05mA_stimulation_4uA_10Hz_200pulses_B-00071.edf','G6'),
    #
    # #
    # # ── Group 4 · 2024.11.20 Ieva · G6 ─────────────────────────────────────
    # (r'C:\Shani\MEA mini1200\2024.11.20 e14_Ieva\2024-11-20T14-45-08_40uA _g6_10Hz_100Pulses_B-00071.edf', 'G6'),
    #
    # # ── Group 6 · 2025.11.02 Retina2 phase1-normal · J6 ────────────────────
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf',   'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-59-46J6_7uA_300us_60us_10Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-01-53J6_7uA_300us_60us_20Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-05-56J6_7uA_300us_60us_20Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-06-13J6_7uA_300us_60us_20Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-07-37J6_10uA_300us_60us_1Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-10-01J6_10uA_300us_60us_10Hz_100pulses.edf', 'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-12-26J6_10uA_300us_60us_20Hz_100pulses.edf', 'J6'),
    #
    # # ── Group 10 · 2025.11.02 Retina3 phase1-normal · J9 ───────────────────
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-55-12J9_10uA_300us_60us_1Hz_100pulses.edf',    'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-57-46J9_10uA_300us_60us_10Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-00-18J9_10uA_300us_60us_20Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-02-50J9_10uA_300us_60us_50Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-03-33J9_20uA_300us_60us_1Hz_100pulses.edf',    'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-06-09J9_20uA_300us_60us_10Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-07-34J9_20uA_300us_60us_20Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-08-27J9_20uA_300us_60us_50Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-13-18J9_20uA_300us_60us_100Hz_1000pulses.edf', 'J9'),

    # # ── Group 12 · 2025.11.12 Retina1 Phase1-Normal · G10 ──────────────────
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-19-197uA_300us_60us_1Hz_100pulse_B-00071.edf',  'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-22-347uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-25-147uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-26-217uA_300us_60us_50Hz_100pulse_B-00071.edf', 'G10'),


]

# ============================================================
#  CHANNEL SELECTION  (interactive helper)
# ============================================================

def _select_channels_interactive(rec, win_size_ms, blank_ms, threshold, data_type='blanked'):
    """
    Show the MEA grid and let the user click subplots to select channels.
    A clicked subplot turns light-pink (selected); clicking again deselects it.
    Close the window to confirm the selection.

    Returns a list of selected channel names, or all channel names when nothing
    is selected.
    """
    _attr_map = {'filtered': 'filtered_data', 'blanked': 'blanked_data', 'raw': 'recording_data'}
    data = getattr(rec, _attr_map[data_type])

    sample_rate  = rec.sample_rate
    stim_indices = rec.stim_indices
    locations    = rec.channel_locations
    win_samples  = int(win_size_ms / 1000 * sample_rate)
    xlim = (0, win_size_ms)
    ylim = (-1.0, 1.0)

    pos_to_ch: dict = {}
    for ch_idx, ch_name in enumerate(rec.channel_names):
        loc = locations[ch_idx] if ch_idx < len(locations) else None
        if loc is None:
            continue
        row, col = int(loc[0]), int(loc[1])
        if 0 <= row < 12 and 0 <= col < 12:
            pos_to_ch[(row, col)] = (ch_idx, ch_name)

    stim_pos = None
    if rec.stim_channel_name:
        for ch_idx, ch_name in enumerate(rec.channel_names):
            if rec.stim_channel_name in ch_name:
                loc = locations[ch_idx] if ch_idx < len(locations) else None
                if loc is not None:
                    stim_pos = (int(loc[0]), int(loc[1]))
                break

    selected: set = set()

    fig, axes = plt.subplots(12, 12, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    used_positions: set = set()
    ax_to_ch: dict = {}

    for (row, col), (ch_idx, ch_name) in pos_to_ch.items():
        used_positions.add((row, col))
        ax = axes[row, col]
        ax_to_ch[ax] = ch_name
        ch_data = data[ch_idx, :]

        for stim_idx in stim_indices:
            start  = int(stim_idx)
            end    = int(min(stim_idx + win_samples, len(ch_data)))
            window = ch_data[start:end]
            if len(window) == 0:
                continue
            times = np.arange(len(window)) / sample_rate * 1000
            ax.plot(times, window, color='black', linewidth=0.2)

        if threshold is not None:
            ax.axhline(-threshold, color='grey', linewidth=0.5, linestyle='--')
        if blank_ms is not None:
            ax.axvline(blank_ms, color='grey', linewidth=0.5, linestyle='--')

        short_label = ch_name.split()[-1].upper() if ' ' in ch_name else ch_name.lower()
        ax.text(0.5, 0.95, short_label, transform=ax.transAxes,
                fontsize=6, va='top', ha='center', color='dimgrey')

        if stim_pos is not None and (row, col) == stim_pos:
            cx = (xlim[0] + xlim[1]) / 2
            cy = (ylim[0] + ylim[1]) / 2
            ax.plot(cx, cy, 'o', color='red', markersize=4, zorder=5)

        for side, spine in ax.spines.items():
            spine.set_visible(side in ('bottom', 'left'))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    for r in range(12):
        for c in range(12):
            if (r, c) not in used_positions:
                axes[r, c].set_visible(False)

    def _on_click(event):
        if event.inaxes is None:
            return
        ch_name = ax_to_ch.get(event.inaxes)
        if ch_name is None:
            return
        if ch_name in selected:
            selected.discard(ch_name)
            event.inaxes.set_facecolor('white')
        else:
            selected.add(ch_name)
            event.inaxes.set_facecolor('lightpink')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _on_click)

    plt.suptitle(
        f'{rec.file_name}  —  Click channels to include in analysis. Close window when done.',
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show(block=False)
    print('    Channel selection open — click subplots to toggle (pink = selected). Close window to confirm.')
    while plt.fignum_exists(fig.number):
        plt.pause(0.05)

    if not selected:
        print('    No channels selected — using all channels.')
        return list(rec.channel_names)
    print(f'    Selected {len(selected)} channel(s): {sorted(selected)}')
    return list(selected)


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    failed = []

    for i, (EDF_FILE, EDF_STIM_ELECTRODE) in enumerate(FILES, 1):
        print(f'\n[{i}/{len(FILES)}] {os.path.basename(EDF_FILE)}  (stim={EDF_STIM_ELECTRODE})')
        try:
            edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
            edf_rec.filter()
            edf_rec.blank(duration_ms=BLANK_MS, source='filtered_data')

            threshold = DIRECT_THRESHOLD_MV
            if INTERACTIVE:
                if threshold is None:
                    # Compute per-channel auto-thresholds and take the median so the
                    # dashed line is visible on the first plot.
                    per_ch = [
                        _compute_threshold(edf_rec.blanked_data[idx],
                                           edf_rec.stim_indices,
                                           edf_rec.sample_rate)
                        for idx in range(edf_rec.blanked_data.shape[0])
                    ]
                    threshold = float(np.median(per_ch))
                    print(f'    Auto threshold (median across channels): {threshold:.4f} mV')

                # Step 1 — channel selection (once per group, first file only)
                if selected_channels is None:
                    selected_channels = _select_channels_interactive(
                        edf_rec, DIRECT_WIN_MS, BLANK_MS, threshold
                    )
                else:
                    print(f'    Using previously selected channels: {sorted(selected_channels)}')

                # Step 2 — threshold review (every file)
                while True:
                    plot_spikes_layout_mea(rec=edf_rec,
                                           win_size_ms=DIRECT_WIN_MS,
                                           data_type='blanked',
                                           threshold=threshold,
                                           blank_ms=BLANK_MS,
                                           output_folder=None)

                    thr_str = f'{threshold} mV'
                    print(f'    Threshold: {thr_str}')
                    ans = input('    Enter new threshold (mV) to re-plot, or press Enter to approve: ').strip()
                    plt.close('all')

                    if ans == '':
                        break
                    try:
                        threshold = float(ans)
                    except ValueError:
                        print('    Invalid — enter a number or press Enter to approve.')

            edf_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, threshold=threshold)

            if selected_channels is not None:
                dr = edf_rec.direct_response
                if dr is not None and not dr.empty and 'channel' in dr.columns:
                    edf_rec.direct_response = (
                        dr[dr['channel'].isin(selected_channels)].reset_index(drop=True)
                    )

            out_path = os.path.join(RESULTS_DIR, f'{edf_rec.file_name}_direct_response.csv')
            edf_rec.direct_response.to_csv(out_path, index=False)
            print(f'    saved → {out_path}')
        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(EDF_FILE)

    print(f'\n{"=" * 60}')
    print(f'Done. {len(FILES) - len(failed)}/{len(FILES)} succeeded.')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')
