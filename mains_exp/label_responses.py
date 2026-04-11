"""
IntactRetinaToolkit — label_responses.py
=========================================
Interactive tool for manually labelling per-pulse responses in RHS recordings.

Workflow
--------
1. Iterate every file listed in RHS_FILES (skips missing paths).
2. Load each recording (no blanking applied).
3. For every channel (stim channel excluded), show an interactive figure
   with one subplot per stimulation pulse — same thin-black-line style as
   plot_spikes_layout_probe16.
4. Annotate using:
     • "All Response"    button → every pulse labelled 1
     • "All No Response" button → every pulse labelled 0
     • Click a subplot          → toggle that pulse (1 ↔ 0)
     • "Done" button / close    → accept current labels and advance
5. After each channel figure is closed, save a compressed NumPy archive:
     signals : (n_pulses, n_samples)  float32
     labels  : (n_pulses,)            int8  (1 = response, 0 = no response)
   File: <OUTPUT_DIR>/<rhs_stem>_ch<channel_name>.npz
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from dataobj import load_rhs

# ============================================================
#  PARAMS — fill in the actual paths before running
# ============================================================

RHS_FILES: list[str] = [
    r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina1\Ch05_300us_50us_7uA_1Hz_250528_092146\Ch05_300us_50us_7uA_1Hz_250528_092146.rhs',
    r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina1\Ch05_300us_50us_7uA_10Hz_250528_092403\Ch05_300us_50us_7uA_10Hz_250528_092403',
    r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch01_300us_50us_7uA_1Hz_250528_113143\Ch01_300us_50us_7uA_1Hz_250528_113143.rhs',
    r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch01_300us_50us_7uA_10Hz_250528_113243\Ch01_300us_50us_7uA_10Hz_250528_113243',


]

WIN_SIZE_MS    = 10.0    # window extracted after each stim pulse (ms)
STIM_THRESHOLD = 470     # threshold passed to load_rhs for stim detection
OUTPUT_DIR     = 'Labels'


# ============================================================
#  VISUAL CONSTANTS  (matching plot_spikes_layout_probe16 style)
# ============================================================

_RESP_BG      = '#ffffff'   # subplot background when labelled as response
_NO_RESP_BG   = '#ffe5e5'   # subplot background when labelled as no-response
_RESP_COLOR   = 'black'     # trace colour for response
_NO_RESP_COLOR = '#bbbbbb'  # trace colour for no-response
_LINE_WIDTH   = 0.3         # same thin line as in the layout plot


# ============================================================
#  ANNOTATION FIGURE
# ============================================================

def _annotate_channel(
    signals: np.ndarray,
    channel_name: str,
    file_name: str,
    win_size_ms: float,
) -> np.ndarray:
    """
    Show an interactive pulse-grid figure for one channel and return labels.

    Parameters
    ----------
    signals : (n_pulses, n_samples) array
    channel_name : str
    file_name : str   — used in the figure title only
    win_size_ms : float

    Returns
    -------
    labels : (n_pulses,) int8 array  (1 = response, 0 = no response)
    """
    n_pulses, n_samples = signals.shape

    # Grid layout — roughly square
    ncols  = max(1, math.ceil(math.sqrt(n_pulses)))
    nrows  = math.ceil(n_pulses / ncols)
    labels = np.ones(n_pulses, dtype=np.int8)   # start: all = response

    fig, axes_2d = plt.subplots(
        nrows, ncols,
        figsize=(max(8, ncols * 1.1), max(5, nrows * 0.85) + 0.8),
        squeeze=False,
    )
    axes = axes_2d.flatten()

    times = np.linspace(0, win_size_ms, n_samples)

    ylim = (-400, 400)

    line_objs: list = []
    ax_to_idx: dict = {}

    for i, ax in enumerate(axes):
        if i < n_pulses:
            ln, = ax.plot(times, signals[i], color=_RESP_COLOR,
                          linewidth=_LINE_WIDTH)
            line_objs.append(ln)
            ax_to_idx[ax] = i

            ax.text(0.05, 0.95, str(i + 1), transform=ax.transAxes,
                    fontsize=5, va='top', ha='left', color='dimgrey')
            ax.set_xlim(0, win_size_ms)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([-400, 0, 400])
            ax.tick_params(axis='y', labelsize=4, pad=1, length=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.set_facecolor(_RESP_BG)
        else:
            ax.set_visible(False)
            line_objs.append(None)

    # ── Update helpers ────────────────────────────────────────
    def _refresh(idx: int) -> None:
        if labels[idx] == 1:
            axes[idx].set_facecolor(_RESP_BG)
            line_objs[idx].set_color(_RESP_COLOR)
        else:
            axes[idx].set_facecolor(_NO_RESP_BG)
            line_objs[idx].set_color(_NO_RESP_COLOR)
        fig.canvas.draw()

    def _on_click(event) -> None:
        # Ignore clicks on button axes or outside any axes
        if event.inaxes not in ax_to_idx:
            return
        if event.button != 1:
            return
        idx = ax_to_idx[event.inaxes]
        labels[idx] ^= 1
        _refresh(idx)

    def _all_response(_event) -> None:
        labels[:] = 1
        for i in range(n_pulses):
            _refresh(i)

    def _all_no_response(_event) -> None:
        labels[:] = 0
        for i in range(n_pulses):
            _refresh(i)

    def _done(_event) -> None:
        plt.close(fig)

    # ── Buttons ───────────────────────────────────────────────
    fig.subplots_adjust(bottom=0.10, top=0.93, left=0.03, right=0.98,
                        hspace=0.15, wspace=0.10)
    ax_b_all  = fig.add_axes([0.12, 0.02, 0.22, 0.055])
    ax_b_none = fig.add_axes([0.39, 0.02, 0.22, 0.055])
    ax_b_done = fig.add_axes([0.66, 0.02, 0.14, 0.055])

    btn_all  = Button(ax_b_all,  'All Response',
                      color='#e8f5e9', hovercolor='#c8e6c9')
    btn_none = Button(ax_b_none, 'All No Response',
                      color='#ffebee', hovercolor='#ffcdd2')
    btn_done = Button(ax_b_done, 'Done',
                      color='#e3f2fd', hovercolor='#bbdefb')

    for btn in (btn_all, btn_none, btn_done):
        btn.label.set_fontsize(8)

    btn_all.on_clicked(_all_response)
    btn_none.on_clicked(_all_no_response)
    btn_done.on_clicked(_done)

    # ── Title & legend ────────────────────────────────────────
    resp_patch    = plt.Line2D([0], [0], color=_RESP_COLOR,
                                linewidth=1.2, label='Response (1)')
    no_resp_patch = plt.Line2D([0], [0], color=_NO_RESP_COLOR,
                                linewidth=1.2, label='No response (0)')
    fig.legend(handles=[resp_patch, no_resp_patch],
               loc='upper right', fontsize=7, frameon=False,
               bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(
        f'{file_name}  —  Channel {channel_name}  '
        f'({n_pulses} pulses)\n'
        f'Click subplot to toggle  |  White = response  |  Pink = no response',
        fontsize=8,
    )

    cid = fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.show(block=True)
    fig.canvas.mpl_disconnect(cid)

    return labels


# ============================================================
#  SAVE
# ============================================================

def _save_labels(
    signals: np.ndarray,
    labels: np.ndarray,
    rhs_stem: str,
    channel_name: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fname    = f'{rhs_stem}_ch{channel_name}.npz'
    out_path = os.path.join(output_dir, fname)
    np.savez_compressed(
        out_path,
        signals=signals.astype(np.float32),
        labels=labels.astype(np.int8),
    )
    n_resp = int(labels.sum())
    print(f'    Saved → {out_path}  '
          f'({n_resp}/{len(labels)} pulses labelled as response)')


# ============================================================
#  LOAD ALL LABELS → UNIFIED DATAFRAME
# ============================================================

def load_label_folder(folder: str):
    """
    Load every .npz file in *folder* (written by _save_labels) and return
    a single tidy DataFrame.

    Each row is one pulse.  Columns:
        file         : str   — recording stem parsed from the filename
        channel      : str   — channel name parsed from the filename
        pulse_index  : int   — 0-based index within that file/channel
        label        : int8  — 1 = response, 0 = no response
        signal       : object — 1-D float32 numpy array (n_samples,)

    The file naming convention expected is: <rhs_stem>_ch<channel>.npz
    """
    import pandas as pd

    abs_folder = os.path.abspath(folder)
    print(f'  [load_label_folder] scanning: {abs_folder}')

    if not os.path.isdir(abs_folder):
        print(f'  [load_label_folder] folder not found — returning empty DataFrame')
        return pd.DataFrame(columns=['file', 'channel', 'pulse_index',
                                     'label', 'signal'])

    npz_files = [f for f in sorted(os.listdir(abs_folder)) if f.endswith('.npz')]
    print(f'  [load_label_folder] found {len(npz_files)} .npz file(s)')

    rows = []
    for fname in npz_files:
        if not fname.endswith('.npz'):
            continue

        # Parse stem and channel from filename  e.g. "rec01_ch5.npz"
        stem = fname[:-4]                          # strip .npz
        if '_ch' in stem:
            rhs_stem, ch_name = stem.rsplit('_ch', maxsplit=1)
        else:
            rhs_stem, ch_name = stem, 'unknown'

        npz = np.load(os.path.join(abs_folder, fname), allow_pickle=False)
        signals_arr = npz['signals']   # (n_pulses, n_samples)
        labels_arr  = npz['labels']    # (n_pulses,)

        for pulse_idx, (sig, lbl) in enumerate(zip(signals_arr, labels_arr)):
            rows.append({
                'file':        rhs_stem,
                'channel':     ch_name,
                'pulse_index': pulse_idx,
                'label':       int(lbl),
                'signal':      sig,
            })

    if not rows:
        return pd.DataFrame(columns=['file', 'channel', 'pulse_index',
                                     'label', 'signal'])

    return pd.DataFrame(rows)


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    for rhs_path in RHS_FILES:
        if not os.path.isfile(rhs_path):
            print(f'[skip] Not found: {rhs_path}')
            continue

        print(f'\n{"=" * 60}')
        print(f'Loading: {rhs_path}')
        print('=' * 60)

        rec = load_rhs(rhs_path, stim_threshold=STIM_THRESHOLD)

        sample_rate  = rec.sample_rate
        stim_indices = rec.stim_indices
        win_samples  = int(WIN_SIZE_MS / 1000 * sample_rate)
        data         = rec.recording_data        # (n_channels, n_samples)
        rhs_stem     = os.path.splitext(rec.file_name)[0]

        print(f'  Channels : {len(rec.channel_names)}')
        print(f'  Pulses   : {len(stim_indices)}')

        for ch_idx, ch_name in enumerate(rec.channel_names):
            if ch_name == rec.stim_channel_name:
                continue   # skip stimulation channel

            # Extract complete pulse windows for this channel
            windows: list[np.ndarray] = []
            for si in stim_indices:
                start = int(si)
                end   = start + win_samples
                if end <= data.shape[1]:
                    windows.append(data[ch_idx, start:end])

            if not windows:
                print(f'  [skip] ch {ch_name}: no complete windows')
                continue

            out_path = os.path.join(OUTPUT_DIR, f'{rhs_stem}_ch{ch_name}.npz')
            if os.path.isfile(out_path):
                print(f'  [skip] ch {ch_name}: already labelled ({out_path})')
                continue

            signals = np.stack(windows)   # (n_pulses, n_samples)

            print(f'  Annotating ch {ch_name}  ({len(windows)} pulses)...')
            labels = _annotate_channel(
                signals      = signals,
                channel_name = ch_name,
                file_name    = rec.file_name,
                win_size_ms  = WIN_SIZE_MS,
            )

            _save_labels(signals, labels, rhs_stem, ch_name, OUTPUT_DIR)

    # ── Build unified DataFrame from all saved .npz files ────
    df = load_label_folder(OUTPUT_DIR)
    if not df.empty:
        print(f'\nUnified dataset: {len(df)} pulses from {df["file"].nunique()} '
              f'file(s), {df["channel"].nunique()} channel(s)')
        print(df.groupby(['file', 'channel'])['label']
                .value_counts().rename('count').to_string())

print('done')
