"""
single_pulse_example.py
========================
Overlaid single-pulse waveforms for ex vivo (row 0) and intact (row 1).
Color encodes pulse order: red = first → violet = last (rainbow).
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataobj import load_edf, load_rhs
from dataobj.analysis.direct import suppress_stim_artifact

# ── ex vivo ────────────────────────────────────────────────────────────────────
EV_EDF_FILE       = (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani'
                     r'\Direct Response - Fading'
                     r'\2024-11-17T13-08-03No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf')
EV_STIM_ELECTRODE = 'G6'
EV_CHANNELS       = ['G7', 'F1']

EV_WIN_MS         = 20.0
EV_BLANK_MS       = 1.5
EV_THRESHOLD_MV   = 0.1

# ── intact ─────────────────────────────────────────────────────────────────────
IN_RHS_DIR    = (r'C:\Shani\SoftC prob\16Ch prob experiments'
                 r'\2025.05.28 E14\Retina3\Ch01_300us_50us_7uA_10Hz_250528_113243')
IN_CHANNELS   = ['5', '26']       # real channel numbers (strings)

IN_WIN_MS         = 20.0
IN_STIM_THRESHOLD = 470
IN_THRESHOLD_MV   = 15


# ── helpers ────────────────────────────────────────────────────────────────────

def _ch_index_map(rec):
    """Return {short_name: array_index} for a recording."""
    idx_map = {}
    for i, name in enumerate(rec.channel_names):
        short = name.split()[-1].upper() if ' ' in name else name.upper()
        idx_map[short] = i
    return idx_map


def _snippets(data, stim_indices, ch_idx, win_samples, sr,
              artifact_removal=False, uv_to_mv=False):
    """Return list of (pulse_i, snippet_mV) for one channel."""
    ch_data = data[ch_idx]
    _k = max(1, int(0.4 * sr / 1000))   # smoothing kernel (same as detect_direct_response)
    out = []
    for pulse_i, si in enumerate(stim_indices):
        start = int(si)
        end   = start + win_samples
        if end > len(ch_data):
            continue
        seg = ch_data[start:end].astype(float)
        if artifact_removal:
            seg = suppress_stim_artifact(seg, sample_rate=sr)
            seg = np.convolve(seg, np.ones(_k) / _k, mode='same')
        if uv_to_mv:
            seg = seg / 1000.0
        out.append((pulse_i, seg))
    return out


# ── load ex vivo ───────────────────────────────────────────────────────────────
print('Loading ex vivo …')
ev_rec = load_edf(EV_EDF_FILE, stim_electrode=EV_STIM_ELECTRODE)
ev_rec.filter()
ev_rec.blank(duration_ms=EV_BLANK_MS, source='filtered_data')
ev_rec.detect_direct_response(win_size_ms=EV_WIN_MS,
                               threshold=EV_THRESHOLD_MV,
                               plot=False)

ev_data   = ev_rec.filtered_data
ev_sr     = ev_rec.sample_rate
ev_stims  = ev_rec.stim_indices
ev_win_s  = int(EV_WIN_MS / 1000 * ev_sr)
ev_t_ms   = np.arange(ev_win_s) / ev_sr * 1000
ev_ch_map = _ch_index_map(ev_rec)
ev_n      = len(ev_stims)

# ── load intact ────────────────────────────────────────────────────────────────
print('Loading intact …')
rhs_files = sorted(glob.glob(os.path.join(IN_RHS_DIR, '*.rhs')))
if not rhs_files:
    raise FileNotFoundError(f'No .rhs file found in {IN_RHS_DIR}')
in_rec = load_rhs(rhs_files[0], stim_threshold=IN_STIM_THRESHOLD)
in_rec.detect_direct_response(win_size_ms=IN_WIN_MS,
                               threshold=IN_THRESHOLD_MV,
                               data_type='raw',
                               plot=False)

in_data   = in_rec.recording_data
in_sr     = in_rec.sample_rate
in_stims  = in_rec.stim_indices
in_win_s  = int(IN_WIN_MS / 1000 * in_sr)
in_t_ms   = np.arange(in_win_s) / in_sr * 1000
in_ch_map = _ch_index_map(in_rec)
in_n      = len(in_stims)

# ── figure: 2 rows × 2 cols ───────────────────────────────────────────────────
cmap = cm.rainbow

#                  label       channels     data       stims      win_s    t_ms      ch_map      sr       n        artifact  uv2mv
rows = [
    ('ex vivo', EV_CHANNELS, ev_data, ev_stims, ev_win_s, ev_t_ms, ev_ch_map, ev_sr, ev_n, False, False),
    ('intact',  IN_CHANNELS, in_data, in_stims, in_win_s, in_t_ms, in_ch_map, in_sr, in_n, True,  True ),
]

_ZERO_BEFORE_MS    = 2.5   # applied to all channels
_ZERO_BEFORE_G7_MS = 1.1   # applied to G7 (ex vivo) instead
MAX_PULSES         = 100   # only plot the first N pulses

fig, axes = plt.subplots(2, 2, figsize=(4.5, 4.5), sharey=False)
fig.subplots_adjust(right=0.87)   # leave room for the colorbar

for row_i, (label, channels, data, stims, win_s, t_ms, ch_map, sr, n_pulses, artifact, uv2mv) in enumerate(rows):
    n_plot = min(n_pulses, MAX_PULSES)
    for col_i, ch_name in enumerate(channels):
        ax = axes[row_i, col_i]
        idx = ch_map.get(ch_name.upper())

        is_g7_ev = (label == 'ex vivo' and ch_name.upper() == 'G7')
        zero_ms  = _ZERO_BEFORE_G7_MS if is_g7_ev else _ZERO_BEFORE_MS

        if idx is None:
            ax.text(0.5, 0.5, f'{ch_name}\nnot found',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')
        else:
            for pulse_i, snippet in _snippets(data, stims, idx, win_s, sr,
                                              artifact_removal=artifact, uv_to_mv=uv2mv):
                if pulse_i >= MAX_PULSES:
                    break
                seg = snippet.copy()
                seg[t_ms < zero_ms] = 0.0
                color = cmap(pulse_i / max(n_plot - 1, 1))
                ax.plot(t_ms, seg, color=color, linewidth=0.4, alpha=0.55)

        ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_title(ch_name, fontsize=9, pad=3)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                fontsize=7, va='top', ha='left', color='dimgrey',
                style='italic')

    axes[row_i, 0].set_ylabel('amplitude (mV)', fontsize=7)

for ax in axes[-1]:
    ax.set_xlabel('time after stim onset (ms)', fontsize=7)

# ── single colorbar on the right ──────────────────────────────────────────────
cbar_ax = fig.add_axes([0.89, 0.1, 0.025, 0.8])   # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=1, vmax=MAX_PULSES))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('pulse #', fontsize=7)
cbar.ax.tick_params(labelsize=6)

fig.suptitle(
    f'Single-pulse waveforms  ·  red = first, violet = last',
    fontsize=9,
)
plt.tight_layout()
plt.show()
plt.close(fig)