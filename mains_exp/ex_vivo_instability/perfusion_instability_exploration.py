"""
perfusion_instability_exploration.py
=====================================
Loops over all .edf files under the phase* subfolders of Retina3
(2025.11.02 experiment), tags each file as 'normal' or 'high_perfusion'
based on the subfolder name, and saves a CSV per file with that tag
prepended to the filename.

Subfolders expected under SOURCE_DIR:
    phase1-normal          → tag 'normal'
    phase2-high perfusion  → tag 'high_perfusion'
"""

import os
import glob
import traceback

import numpy as np
import matplotlib.pyplot as plt

from dataobj import load_edf

SOURCE_DIR   = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3'
RESULTS_DIR  = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_perfusion'
STIM_ELECTRODE      = 'J9'

# ============================================================
#  PARAMS
# ============================================================
DIRECT_WIN_MS       = 10.0
BLANK_MS            = 1.0
DIRECT_THRESHOLD_MV = 0.2
INTERACTIVE         = True   # set to True to pick channels interactively before processing
selected_channels   = None


# ── helpers ──────────────────────────────────────────────────────────────────

def perfusion_tag(folder_name: str) -> str:
    """Return 'high_perfusion' or 'normal' based on the phase subfolder name."""
    name = folder_name.lower()
    if 'high' in name:
        return 'high_perfusion'
    return 'normal'


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
        f'{rec.file_name}  -  Click channels to include in analysis. Close window when done.',
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show(block=False)
    print('    Channel selection open - click subplots to toggle (pink = selected). Close window to confirm.')
    while plt.fignum_exists(fig.number):
        plt.pause(0.05)

    if not selected:
        print('    No channels selected - using all channels.')
        return list(rec.channel_names)
    print(f'    Selected {len(selected)} channel(s): {sorted(selected)}')
    return list(selected)


def find_edf_entries():
    """Return sorted list of (edf_path, tag) for all phase* subfolders."""
    entries = []
    for phase_dir in sorted(glob.glob(os.path.join(SOURCE_DIR, 'phase*'))):
        tag = perfusion_tag(os.path.basename(phase_dir))
        for path in sorted(glob.glob(os.path.join(phase_dir, '*.edf'))):
            entries.append((path, tag))
    return entries


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    entries = find_edf_entries()
    if not entries:
        print(f'No .edf files found under {SOURCE_DIR}')
        exit()

    print(f'Found {len(entries)} .edf files.\n')

    failed = []

    for i, (path, tag) in enumerate(entries, 1):
        fname    = os.path.basename(path)
        out_path = os.path.join(RESULTS_DIR, f'{tag}_{fname}_direct_response.csv')

        # if os.path.exists(out_path):
        #     print(f'[{i}/{len(entries)}] already exists, skipping: {fname}')
        #     continue

        print(f'[{i}/{len(entries)}] [{tag}] {fname}')
        try:
            rec = load_edf(path, stim_electrode=STIM_ELECTRODE)
            rec.filter()
            rec.blank(duration_ms=BLANK_MS, source='filtered_data')

            if INTERACTIVE and selected_channels is None:
                selected_channels = _select_channels_interactive(
                    rec, DIRECT_WIN_MS, BLANK_MS, DIRECT_THRESHOLD_MV
                )

            rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                       threshold=DIRECT_THRESHOLD_MV,
                                       plot=True)

            if selected_channels is not None:
                dr = rec.direct_response
                if dr is not None and not dr.empty and 'channel' in dr.columns:
                    rec.direct_response = (
                        dr[dr['channel'].isin(selected_channels)].reset_index(drop=True)
                    )

            rec.direct_response.to_csv(out_path, index=False)
            print(f'    saved → {out_path}')
        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(path)

    print(f'\n{"=" * 60}')
    print(f'Done. {len(entries) - len(failed)}/{len(entries)} succeeded.')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')
