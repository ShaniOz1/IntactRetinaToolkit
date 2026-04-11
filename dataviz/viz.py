"""
IntactRetinaToolkit.dataviz.viz
=================================
Visualisation functions for retinal recordings.

All functions accept a RetinalRecording object (or the data structures
derived from it) and produce matplotlib figures.

Functions
---------
plot_probe_schematic   — draw the Intan probe layout, colour-coding stim/response channels
plot_direct_spikes     — overlay + average of spike waveforms per responding channel
plot_artifacts_vs_signals — raw pulses (row 1) vs ICA-cleaned signals (row 2)
plot_spike_amps_vs_time   — spike amplitude over the course of the recording
plot_indirect_response    — raster + single-channel trace + response metrics
plot_overlay_pulses              — raw pulse overlays (single obj) or averages (multi-obj comparison)
plot_direct_response_summary     — MEA heatmaps of mean amp/latency/width + decay heatmap + decay trace
plot_spikes_layout_mea           — overlay all pulse windows on the 12×12 MEA grid (EDF)
plot_spikes_layout_probe16       — overlay all pulse windows on the prob16 ring layout (RHS)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Probe schematic
# ─────────────────────────────────────────────────────────────────────────────

def plot_probe_schematic(
    ax: plt.Axes,
    stim_channels: list[int],
    response_channels: list[int],
    probe: str = 'prob16',
) -> None:
    """
    Draw an Intan probe schematic on ax, colour-coding stimulation and
    response channels.

    Parameters
    ----------
    ax : plt.Axes
    stim_channels : list[int]
        Channel indices/numbers to highlight in red (stimulation).
    response_channels : list[int]
        Channel indices/numbers to highlight in dodgerblue (response).
    probe : str
        Probe layout to draw: 'prob16' (default) or 'prob32'.
    """
    if probe == 'prob16':
        _draw_probe_ring(
            ax,
            outer_numbers=[26, 5, 25, 6, 24, 7, 28, 2, 29, 1, 30, 0, 31, 99, 99, 99, 3, 27, 4],
            stim_channels=stim_channels,
            response_channels=response_channels,
            radius=2,
        )

    elif probe == 'prob32':
        inner = [13, 2, 12, 3, 11, 4, 10, 5, 9, 6, 8, 7, 99, 99, 99, 99, 15, 0, 14, 1]
        outer = [25, 22, 26, 21, 27, 20, 28, 19, 29, 18, 30, 17, 31, 16, 99, 99, 99, 99, 24, 23]

        for r, numbers in [(1, inner), (2, outer)]:
            circle = plt.Circle((0, 0), r, color='grey', alpha=0.3,
                                fill=False, linewidth=20)
            ax.add_artist(circle)
            angles = np.linspace(0, 2 * np.pi, len(numbers), endpoint=False)
            for num, angle in zip(numbers, angles):
                if num == 99:
                    continue
                px, py = r * np.cos(angle), r * np.sin(angle)
                color = ('red' if num in stim_channels
                         else 'dodgerblue' if num in response_channels
                         else 'black')
                ax.text(px, py, str(num), color=color,
                        fontsize=8 if r == 1 else 10,
                        ha='center', va='center')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])

    red_patch  = mpatches.Patch(color='red',        label='Stimulation')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Response')
    ax.legend(handles=[red_patch, blue_patch], loc='upper left',
              fontsize='x-small', ncol=2, bbox_to_anchor=(0, 1.15))


def _draw_probe_ring(
    ax: plt.Axes,
    outer_numbers: list[int],
    stim_channels: list[int],
    response_channels: list[int],
    radius: float = 2,
) -> None:
    circle = plt.Circle((0, 0), radius, color='grey', alpha=0.3,
                         fill=False, linewidth=20)
    ax.add_artist(circle)
    angles = np.linspace(0, 2 * np.pi, len(outer_numbers), endpoint=False)
    for num, angle in zip(outer_numbers, angles):
        if num == 99:
            continue
        px, py = radius * np.cos(angle), radius * np.sin(angle)
        color = ('red'        if num in stim_channels
                 else 'dodgerblue' if num in response_channels
                 else 'black')
        ax.text(px, py, str(num), color=color, fontsize=10,
                ha='center', va='center')


# ─────────────────────────────────────────────────────────────────────────────
# Direct response — spike overlays
# ─────────────────────────────────────────────────────────────────────────────

def plot_direct_spikes(
    rec,
    spikes_dict: dict,
    save: bool = True,
) -> None:
    """
    Plot overlaid spike waveforms (+ mean) for each responding channel,
    alongside a probe schematic.

    Parameters
    ----------
    rec : RetinalRecording
    spikes_dict : dict
        {channel_name: np.ndarray (n_spikes, n_samples)} as produced by
        the spike-detection analysis step.
    save : bool
        If True and rec.output_folder is set, saves the figure to disk.
    """
    response_channels = [int(k[-3:]) for k in spikes_dict.keys()]
    if not response_channels:
        print("[plot_direct_spikes] No responding channels found.")
        return

    fs_ms = rec.sample_rate / 1000
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    stim_ch_num = (int(rec.stim_channel_name[-3:])
                   if rec.stim_channel_name else None)
    stim_list = [stim_ch_num] if stim_ch_num is not None else []
    plot_probe_schematic(axs[0, 0], stim_list, response_channels, probe='prob16')

    for (title, signals), ax in zip(spikes_dict.items(), axs.flatten()[1:]):
        t = (np.arange(signals.shape[1]) - 5 * fs_ms) / fs_ms
        for row in signals:
            ax.plot(t, row, color='lightgrey', alpha=0.8)
        ax.plot(t, np.mean(signals, axis=0), 'k', linewidth=1.5)
        ax.set_title(f'Ch – {title}')
        ax.set_xlabel('Time after stimulation [ms]')
        ax.set_ylabel('Amplitude [µV]')
        ax.set_ylim(-500, 500)

    plt.suptitle(rec.file_name)
    plt.tight_layout()

    if save and rec.output_folder:
        fig.savefig(f'{rec.output_folder}/Direct_response.png')
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Artifacts vs ICA-cleaned signals
# ─────────────────────────────────────────────────────────────────────────────

def plot_artifacts_vs_signals(
    rec,
    signals_mat_3d: np.ndarray,
    artifacts_mat_3d: np.ndarray,
    spikes_dict: dict,
    save: bool = True,
) -> None:
    """
    Two-row figure: top row = ICA-cleaned neural signals, bottom = artifacts.

    Parameters
    ----------
    rec : RetinalRecording
    signals_mat_3d : np.ndarray
        Shape (n_pulses, n_channels, n_samples) — cleaned signals.
    artifacts_mat_3d : np.ndarray
        Shape (n_pulses, n_channels, n_samples) — artifact component.
    spikes_dict : dict
        Used only to check whether responding channels exist.
    save : bool
    """
    if not spikes_dict:
        return

    n_ch = signals_mat_3d.shape[1]
    ch_names = rec.channel_names[:n_ch]

    fig, axs = plt.subplots(2, n_ch, figsize=(max(18, n_ch * 1.2), 4))

    for i in range(n_ch):
        # Row 0: cleaned signals
        for p in range(signals_mat_3d.shape[0]):
            axs[0, i].plot(signals_mat_3d[p, i, :], color='grey', linewidth=0.5)
        axs[0, i].set_ylim(-200, 200)
        axs[0, i].set_title(ch_names[i], fontsize=6)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        # Row 1: artifacts
        for p in range(artifacts_mat_3d.shape[0]):
            axs[1, i].plot(artifacts_mat_3d[p, i, :], color='k', linewidth=0.5)
        axs[1, i].set_ylim(-5000, 5000)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    plt.tight_layout()

    if save and rec.output_folder:
        fig.savefig(f'{rec.output_folder}/Artifacts_vs_signals.png')
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Spike amplitude over time
# ─────────────────────────────────────────────────────────────────────────────

def plot_spike_amps_vs_time(
    rec,
    signals_mat_3d: np.ndarray,
    channel_index: int,
    save: bool = True,
) -> None:
    """
    Two-panel figure: left = overlaid spike waveforms for one channel,
    right = minimum amplitude (peak negativity) vs pulse number.

    Parameters
    ----------
    rec : RetinalRecording
    signals_mat_3d : np.ndarray
        Shape (n_pulses, n_channels, n_samples).
    channel_index : int
        Which channel (axis-1 index) to plot.
    save : bool
    """
    mat = signals_mat_3d[:, channel_index, :]
    fs  = rec.sample_rate
    min_vals = [np.min(row) for row in mat]

    fig  = plt.figure(figsize=(15, 4))
    gs   = GridSpec(1, 4, figure=fig)
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1:4])

    t = (np.arange(mat.shape[1]) - 125) / (fs / 1000)
    for row in mat:
        ax1.plot(t, row / 1000, color='k', linewidth=0.1)
    ax1.plot(t, np.mean(mat, axis=0) / 1000, color='k', linewidth=2)
    ax1.set_xlim(0, 10)
    ax1.set_xlabel('Time [ms]', fontsize=14)
    ax1.set_ylabel('Amplitude [mV]', fontsize=14)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.tick_params(labelsize=14)

    ax2.scatter(np.arange(len(min_vals)), np.array(min_vals) / 1000,
                color='k', marker='.', s=30)
    ax2.set_xlabel('# Pulse', fontsize=14)
    ax2.set_xlim(0, len(min_vals))
    ax2.tick_params(labelsize=14)

    arr = np.abs(np.array(min_vals) / 1000)
    decrease = round(100 * (np.mean(arr[:10]) - np.mean(arr[-10:])), 2)
    plt.suptitle(f'decrease={decrease}% — {rec.file_name} — ch{channel_index}',
                 fontsize=8)
    plt.tight_layout()

    if save and rec.output_folder:
        fig.savefig(f'{rec.output_folder}/Spike_amps_ch{channel_index}.png')
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Indirect response — raster + metrics
# ─────────────────────────────────────────────────────────────────────────────

def plot_indirect_response(
    rec,
    spikes_indices: list[np.ndarray],
    selected_ch_ind: int = 15,
    blanking_win_msec: float = 15.0,
    save: bool = True,
) -> None:
    """
    Four-panel figure for indirect (network-driven) response:
    raster plot, single-channel trace, spike count, ISI, latency,
    peak firing rate — all vs pulse number.

    Parameters
    ----------
    rec : RetinalRecording
    spikes_indices : list[np.ndarray]
        One array per channel, each containing spike sample indices.
    selected_ch_ind : int
        Channel index to detail in the lower panels. Default: 15.
    blanking_win_msec : float
        Blanking window used (for axis label). Default: 15 ms.
    save : bool
    """
    times      = np.arange(rec.n_samples) / rec.sample_rate
    stim_idx   = rec.stim_indices
    spike_idx  = spikes_indices[selected_ch_ind]

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(4, 4, figure=fig)

    # ── Raster ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(times, 0.5 + selected_ch_ind, 1 + selected_ch_ind,
                     color='lightgrey', alpha=0.5)
    for ch, ind_list in enumerate(spikes_indices):
        for ind in ind_list:
            ax1.vlines(times[ind], ymin=0.5 + ch, ymax=1 + ch,
                       color='black', linewidth=0.7)
    ax1.scatter(times[stim_idx],
                rec.n_channels + np.zeros_like(stim_idx),
                color='#800020', marker='v', s=25)
    ax1.set_ylabel('# electrode')
    ax1.set_title('Raster plot (all electrodes)')
    ax1.set_xlabel('Time [s]')
    ax1.set_xlim(0, times[stim_idx].max() + 1)

    # ── Single-channel trace (blanked) ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    blanking_samples = int(blanking_win_msec / 1000 * rec.sample_rate)
    mask = np.ones(rec.n_samples, dtype=bool)
    for idx in sorted(stim_idx):
        start = int(max(0, idx - int(0.005 * rec.sample_rate)))
        end   = int(min(rec.n_samples, idx + blanking_samples + 1))
        mask[start:end] = False
    blanked = np.copy(rec.recording_data[selected_ch_ind, :])
    blanked[~mask] = 0

    lim = 200
    ax2.plot(times, blanked, color='k', linewidth=0.05)
    ax2.scatter(times[stim_idx],
                np.zeros_like(stim_idx) + lim - 100,
                color='#800020', marker='v', s=20)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude [µV]')
    ax2.set_xlim(0, times[stim_idx].max() + 1)
    ax2.set_title(f'Blanked ({blanking_win_msec} ms) — electrode {selected_ch_ind}')

    # ── Per-pulse metrics ────────────────────────────────────────────────
    spike_count, isi_mean, latency, peak_fr = _compute_response_metrics(
        stim_idx, spike_idx, rec.sample_rate
    )

    ax3 = fig.add_subplot(gs[2, 0:2])
    ax3.scatter(np.arange(len(spike_count)), spike_count,
                color='k', marker='.', s=30)
    ax3.set_ylabel('Spike count')
    ax3.set_xlabel('# pulses')

    ax4 = fig.add_subplot(gs[2, 2:4])
    ax4.scatter(np.arange(len(isi_mean)), isi_mean,
                color='k', marker='.', s=30)
    ax4.set_ylabel('Inter-spike interval [ms]')
    ax4.set_xlabel('# pulses')

    ax5 = fig.add_subplot(gs[3, 0:2])
    ax5.scatter(np.arange(len(latency)), latency,
                color='k', marker='.', s=30)
    ax5.set_ylabel('Latency [ms]')
    ax5.set_xlabel('# pulses')

    ax6 = fig.add_subplot(gs[3, 2:4])
    ax6.scatter(np.arange(len(peak_fr)), peak_fr,
                color='k', marker='.', s=30)
    ax6.set_ylabel('Peak firing rate [1/ms]')
    ax6.set_xlabel('# pulses')

    plt.suptitle(rec.file_name)
    plt.tight_layout()

    if save and rec.output_folder:
        fig.savefig(
            f'{rec.output_folder}/Indirect_{rec.file_name}_ch{selected_ch_ind}.png'
        )
        plt.close(fig)
    else:
        plt.show()


def _compute_response_metrics(
    stim_indices: np.ndarray,
    spike_indices: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pulse spike count, mean ISI, latency, and peak firing rate.
    Returns four arrays of length len(stim_indices).
    """
    spike_count, isi_mean, latency, peak_fr = [], [], [], []

    for i, stim_start in enumerate(stim_indices):
        stim_end = (stim_indices[i + 1] if i < len(stim_indices) - 1
                    else stim_start + sample_rate)
        in_win = spike_indices[
            (spike_indices > stim_start) & (spike_indices < stim_end)
        ]

        spike_count.append(len(in_win))

        if len(in_win) > 1:
            isi = np.diff(in_win) / sample_rate * 1000
            isi_mean.append(np.mean(isi))
            peak_fr.append(1 / np.min(isi))
        else:
            isi_mean.append(np.nan)
            peak_fr.append(np.nan)

        latency.append(
            (in_win[0] - stim_start) / sample_rate * 1000
            if len(in_win) > 0 else np.nan
        )

    return (np.array(spike_count), np.array(isi_mean),
            np.array(latency), np.array(peak_fr))


# ─────────────────────────────────────────────────────────────────────────────
# Overlay pulses (single recording or multi-recording comparison)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overlay_pulses(
    rec_list: list,
    output_path: str | None = None,
    ylim: int = 7000,
    average: bool = True,
    colors: list[str] | None = None,
) -> None:
    """
    Plot overlaid pulse waveforms across all channels in a 4×4 grid.

    Parameters
    ----------
    rec_list : list[RetinalRecording]
        One or more recordings. If average=False, only rec_list[0] is used
        and every individual pulse is plotted. If average=True, the mean
        pulse per channel is plotted for each recording (colour-coded).
    output_path : str | None
        Full path (including filename) to save the figure. If None, shows
        interactively.
    ylim : int
        Y-axis symmetric limit in µV. Default: 7000.
    average : bool
        False → plot all individual pulses for rec_list[0].
        True  → plot per-recording averages, one colour each.
    colors : list[str] | None
        Colours for each recording in average mode. Defaults to
        ['#5f8fb7', '#6d9d74', '#9c5a61'].
    """
    if colors is None:
        colors = ['#5f8fb7', '#6d9d74', '#9c5a61']

    fig, axes = plt.subplots(4, 4, figsize=(10, 15))
    axes = axes.flatten()

    if not average:
        rec   = rec_list[0]
        data  = rec.pulses                  # (n_pulses, n_ch, n_samples)
        fs    = rec.sample_rate
        stim_ch = getattr(rec, 'stim_channel_index', None)
        title = (f'Stimulating ch {rec.stim_channel_name}'
                 if rec.stim_channel_name else rec.file_name)

        for i, ax in enumerate(axes):
            if i >= data.shape[1]:
                ax.axis('off')
                continue
            for j in range(data.shape[0]):
                t = -3 + np.arange(data.shape[2]) / fs * 1000
                ax.plot(t, data[j, i, :], color='grey', linewidth=0.5, alpha=0.3)
            ax.set_title(
                rec.channel_names[i],
                color='red' if i == stim_ch else 'black',
            )
            ax.set_ylim(-ylim, ylim)
            if i < 12:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Time [ms]')
            if i % 4 != 0:
                ax.set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.93])

    else:
        category = []
        for rec, color in zip(rec_list, colors):
            data    = rec.pulses
            fs      = rec.sample_rate
            avg     = np.mean(data, axis=0)
            stim_ch = getattr(rec, 'stim_channel_index', None)
            category.append(getattr(rec, 'parent_folder', rec.file_name))

            for i, ax in enumerate(axes):
                if i >= data.shape[1]:
                    ax.axis('off')
                    continue
                t = -3 + np.arange(data.shape[2]) / fs * 1000
                ax.plot(t, avg[i, :], color=color, linewidth=1)
                ax.set_title(
                    rec.channel_names[i],
                    color='red' if i == stim_ch else 'black',
                )
                ax.set_ylim(-ylim, ylim)
                if i < 12:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('Time [ms]')
                if i % 4 != 0:
                    ax.set_yticks([])

        legend_handles = [
            plt.Line2D([0], [0], color=c, lw=2, label=cat)
            for c, cat in zip(colors, category)
        ]
        plt.figlegend(handles=legend_handles, loc='upper center',
                      ncol=len(rec_list), frameon=False)
        title = (f'Stimulating ch {rec_list[0].stim_channel_name}'
                 if rec_list[0].stim_channel_name else rec_list[0].file_name)
        plt.suptitle(title, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Overlay spike windows on MEA grid
# ─────────────────────────────────────────────────────────────────────────────

_VIZ_DATA_ATTRS = {
    'filtered': 'filtered_data',
    'blanked':  'blanked_data',
    'raw':      'recording_data',
}


def plot_spikes_layout_mea(
    rec,
    win_size_ms: float = 10.0,
    data_type: str = 'blanked',
    threshold: float | None = None,
    ylim: tuple[float, float] = (-1.0, 1.0),
    save: bool = True,
    output_folder: str | None = None,
) -> None:
    """
    Overlay all pulse windows for every channel on a 12×12 MEA grid (EDF).

    Each subplot position corresponds to the electrode's physical location.
    All pulse traces are plotted in black with a thin line.
    An optional horizontal grey dashed threshold line is drawn per subplot.
    Unused grid positions are hidden.

    Parameters
    ----------
    rec : RetinalRecording
        Must be an EDF recording (rec.source == 'edf').
    win_size_ms : float
        Window length in ms to extract after each stim onset. Default: 10.
    data_type : {'blanked', 'filtered', 'raw'}
        Which data array to plot. Default: 'blanked'.
    threshold : float | None
        If provided, draw a horizontal grey dashed line at -threshold.
    ylim : tuple[float, float]
        Y-axis limits for every subplot. Default: (-1, 1).
    save : bool
        If True and output_folder is set, saves the figure to disk.
    output_folder : str | None
        Directory in which to save the figure. If None, shows interactively.
    """
    if rec.source != 'edf':
        raise ValueError("plot_spikes_layout_mea requires an EDF recording.")

    # ── EDF ─────────────────────────────────────────────────────────────────
    attr = _VIZ_DATA_ATTRS.get(data_type)
    if attr is None:
        raise ValueError(
            f"data_type must be 'blanked', 'filtered', or 'raw'; got {data_type!r}"
        )
    data = getattr(rec, attr, None)
    if data is None:
        raise ValueError(f"rec.{attr} is None — run the appropriate preprocessing step first.")

    sample_rate  = rec.sample_rate
    stim_indices = rec.stim_indices
    locations    = rec.channel_locations   # list[tuple[int,int] | None]

    win_samples = int(win_size_ms / 1000 * sample_rate)
    xlim        = (0, win_size_ms)

    # Identify stimulation electrode grid position
    stim_pos = None
    if rec.stim_channel_name:
        for ch_idx, ch_name in enumerate(rec.channel_names):
            if rec.stim_channel_name in ch_name:
                loc = locations[ch_idx] if ch_idx < len(locations) else None
                if loc is not None:
                    stim_pos = (loc[0], loc[1])
                break

    fig, axes = plt.subplots(12, 12, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    used_positions = set()

    for ch_idx, ch_name in enumerate(rec.channel_names):
        loc = locations[ch_idx] if ch_idx < len(locations) else None
        if loc is None:
            continue
        row, col = loc
        if not (0 <= row < 12 and 0 <= col < 12):
            continue

        used_positions.add((row, col))
        ax      = axes[row, col]
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

        # Channel label — top-left corner
        short_label = ch_name.split()[-1].upper() if ' ' in ch_name else ch_name.lower()
        ax.text(0.05, 0.95, short_label, transform=ax.transAxes,
                fontsize=6, va='top', ha='left', color='dimgrey')

        # Red dot for stimulation electrode
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

    # Hide unused subplots
    for r in range(12):
        for c in range(12):
            if (r, c) not in used_positions:
                axes[r, c].set_visible(False)

    # Bottom-most right used subplot: add axis ticks with tiny font
    if used_positions:
        max_row = max(r for r, c in used_positions)
        max_col = max(c for r, c in used_positions if r == max_row)
        br_ax = axes[max_row, max_col]
        x_ticks = [0, win_size_ms / 2, win_size_ms]
        br_ax.set_xticks(x_ticks)
        br_ax.set_xticklabels([f'{v:.0f}' for v in x_ticks], fontsize=6)
        y_ticks = [ylim[0], (ylim[0] + ylim[1]) / 2, ylim[1]]
        br_ax.set_yticks(y_ticks)
        br_ax.set_yticklabels([f'{v:.0f}' for v in y_ticks], fontsize=6)
        br_ax.tick_params(axis='both', length=2, pad=1)
        br_ax.set_xlabel('ms', fontsize=6, labelpad=1)
        br_ax.set_ylabel('mV', fontsize=6, labelpad=1)

    data_label = data_type.capitalize()
    plt.suptitle(f'{rec.file_name}  [{data_label}]', fontsize=9)

    stim_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                             markersize=4, label='Stimulation\nelectrode')
    thresh_handle = plt.Line2D([0], [0], color='grey', linewidth=0.8,
                               linestyle='--', label='Threshold')
    fig.legend(handles=[stim_handle, thresh_handle], loc='upper right',
               fontsize='x-small', frameon=False, borderaxespad=1)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save and output_folder:
        fig.savefig(f'{output_folder}/spikes_layout_mea_{data_type}.png', dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Spike overlay — prob16 ring layout (RHS)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spikes_layout_probe16(
    rec,
    win_size_ms: float = 10.0,
    data_type: str = 'blanked',
    threshold: float | None = None,
    save: bool = True,
    output_folder: str | None = None,
) -> None:
    """
    Overlay all pulse windows for every channel on the prob16 ring layout (RHS).

    Subplot positions follow the physical circular arrangement of the probe
    (9×9 grid, derived from the ring schematic).  All pulse traces are
    plotted in black with a thin line.  An optional grey dashed threshold
    line is drawn per subplot.  Unused grid positions are hidden.

    Y-axis limits are set automatically from the max absolute value across
    all channels in the selected data array, with a 0.5 % margin.

    Parameters
    ----------
    rec : RetinalRecording
        Must be an RHS recording (rec.source == 'rhs').
    win_size_ms : float
        Window length in ms to extract after each stim onset. Default: 10.
    data_type : {'blanked', 'filtered', 'raw'}
        Which data array to plot. Default: 'blanked'.
    threshold : float | None
        If provided, draw a horizontal grey dashed line at -threshold.
    save : bool
        If True and output_folder is set, saves the figure to disk.
    output_folder : str | None
        Directory in which to save the figure. If None, shows interactively.
    """
    if rec.source != 'rhs':
        raise ValueError("plot_spikes_layout_probe16 requires an RHS recording.")

    attr = _VIZ_DATA_ATTRS.get(data_type)
    if attr is None:
        raise ValueError(
            f"data_type must be 'blanked', 'filtered', or 'raw'; got {data_type!r}"
        )
    data = getattr(rec, attr, None)
    if data is None:
        raise ValueError(f"rec.{attr} is None — run the appropriate preprocessing step first.")

    sample_rate  = rec.sample_rate
    stim_indices = rec.stim_indices
    win_samples  = int(win_size_ms / 1000 * sample_rate)
    xlim         = (0, win_size_ms)

    # Y limits: computed from the plotted windows only (not entire signal)
    import math
    window_vals = []
    for ch_idx in range(data.shape[0]):
        for stim_idx in stim_indices:
            start = int(stim_idx)
            end   = int(min(stim_idx + win_samples, data.shape[1]))
            if start < end:
                window_vals.append(data[ch_idx, start:end])
    if window_vals:
        all_wins = np.concatenate(window_vals)
        ymin = math.floor(float(all_wins.min()) / 50) * 50
        ymax = math.ceil( float(all_wins.max()) / 50) * 50
    else:
        ymin, ymax = -50, 50
    ylim = (ymin, ymax)

    # Physical ring layout — channel order around the circle starting at
    # 3 o'clock, going counter-clockwise.  99 = empty slot.
    outer_numbers = [26, 5, 25, 6, 24, 7, 28, 2, 29, 1, 30, 0, 31, 99, 99, 99, 3, 27, 4]
    n_slots  = len(outer_numbers)
    angles   = np.linspace(0, 2 * np.pi, n_slots, endpoint=False)

    # Channel name → recording index
    ch_name_to_idx = {ch: i for i, ch in enumerate(rec.channel_names)}
    stim_ch_name   = rec.stim_channel_name

    # ── Figure: axes placed manually on a circle ──────────────────
    fig = plt.figure(figsize=(12, 12))

    # Ring geometry in figure-fraction coordinates
    cx, cy = 0.5, 0.5    # centre of the ring
    radius = 0.36         # ring radius
    ax_w   = 0.12         # subplot width  (figure fraction)
    ax_h   = 0.075        # subplot height (figure fraction)

    # ch 3 (slot 16, ≈303°) is the bottom-right channel — used as scale reference
    ref_ch_name = '3'
    ref_ax      = None

    for slot_idx, ch_num in enumerate(outer_numbers):
        if ch_num == 99:
            continue
        ch_name = str(ch_num)
        ch_idx  = ch_name_to_idx.get(ch_name)
        if ch_idx is None:
            continue

        angle  = angles[slot_idx]
        ax_cx  = cx + radius * np.cos(angle)
        ax_cy  = cy + radius * np.sin(angle)
        ax = fig.add_axes([ax_cx - ax_w / 2, ax_cy - ax_h / 2, ax_w, ax_h])

        if ch_name == ref_ch_name:
            ref_ax = ax

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

        # Red dot for stimulation electrode
        if stim_ch_name and ch_name == stim_ch_name:
            ax.plot((xlim[0] + xlim[1]) / 2, 0, 'o',
                    color='red', markersize=4, zorder=5)

        # Channel label
        ax.text(0.05, 0.95, ch_name, transform=ax.transAxes,
                fontsize=6, va='top', ha='left', color='dimgrey')

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        # Transparent background so the right spine of this axis is never
        # hidden by the patch of a neighbouring axis
        ax.patch.set_visible(False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    # Reference subplot (ch 3): also show bottom and left spines + tick labels
    if ref_ax is not None:
        ref_ax.spines['bottom'].set_visible(True)
        ref_ax.spines['left'].set_visible(True)
        x_ticks = [0, win_size_ms / 2, win_size_ms]
        ref_ax.set_xticks(x_ticks)
        ref_ax.set_xticklabels([f'{v:.0f}' for v in x_ticks], fontsize=6)
        y_ticks = [ymin, 0, ymax]
        ref_ax.set_yticks(y_ticks)
        ref_ax.set_yticklabels([f'{v:.0f}' for v in y_ticks], fontsize=6)
        ref_ax.tick_params(axis='both', length=2, pad=1)
        ref_ax.set_xlabel('ms', fontsize=6, labelpad=1)
        ref_ax.set_ylabel('mV', fontsize=6, labelpad=1)

    # ── Legend and title ──────────────────────────────────────────
    data_label   = data_type.capitalize()
    stim_handle  = plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='red', markersize=4,
                              label='Stimulation electrode')
    handles = [stim_handle]
    if threshold is not None:
        handles.append(plt.Line2D([0], [0], color='grey', linewidth=0.8,
                                  linestyle='--', label='Threshold'))
    fig.legend(handles=handles, loc='upper right', fontsize='x-small',
               frameon=False, borderaxespad=1)

    fig.suptitle(f'{rec.file_name}  [Prob16 – {data_label}]', fontsize=9,
                 y=0.98)

    if save and output_folder:
        fig.savefig(f'{output_folder}/spikes_layout_probe16_{data_type}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Direct-response summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_direct_response_summary(
    rec,
    save: bool = True,
    output_folder: str | None = None,
) -> None:
    """
    Two-row summary figure for direct-response results.

    Row 1 — three MEA heatmaps (12×12 grid):
        mean amplitude (µV), mean latency (ms), mean width (ms),
        each averaged across all detected pulses per channel.

    Row 2:
        Left  (1/3 width): MEA heatmap of the per-channel amplitude-decay
            constant *k* from an exponential fit  A·exp(−k·pulse_index).
        Right (2/3 width): |amplitude| vs pulse number for the channel
            that shows the steepest decay, with the fitted curve overlaid.

    All heatmaps use the ``Purples`` colormap.

    Parameters
    ----------
    rec : RetinalRecording
        Must have ``rec.direct_response`` populated (call
        ``rec.detect_direct_response()`` first).  The DataFrame must
        contain an ``amplitude_decay`` column; this is added automatically
        by ``detect_direct_response`` starting from the version that ships
        ``add_amplitude_decay``.
    save : bool
        Save to *output_folder* when True (default). Show interactively
        otherwise.
    output_folder : str | None
        Directory for the saved figure. Ignored when *save* is False.
    """
    df = rec.direct_response
    if df is None or df.empty:
        raise ValueError(
            "rec.direct_response is None or empty — call "
            "rec.detect_direct_response() first."
        )
    if 'amplitude_decay_pct' not in df.columns:
        from dataobj.analysis.direct import add_amplitude_decay
        df = add_amplitude_decay(df)
        rec.direct_response = df

    # ── Channel → grid-location map ──────────────────────────────
    ch_to_loc: dict[str, tuple[int, int]] = {}
    for ch_idx, ch_name in enumerate(rec.channel_names):
        loc = (rec.channel_locations[ch_idx]
               if ch_idx < len(rec.channel_locations) else None)
        if loc is not None:
            ch_to_loc[ch_name] = (int(loc[0]), int(loc[1]))

    # ── Per-channel summary stats ─────────────────────────────────
    df_work = df.copy()
    df_work['amplitude_mV'] = df_work['amplitude_mV'].abs()

    ch_stats = (
        df_work
        .groupby('channel', sort=False)
        .agg(
            mean_amplitude= ('amplitude_mV',      'mean'),
            mean_latency=   ('latency_ms',        'mean'),
            mean_width=     ('width_ms',          'mean'),
            amplitude_decay=('amplitude_decay_pct', 'first'),
        )
        .reset_index()
    )

    # ── Build 12×12 grids ─────────────────────────────────────────
    def _build_grid(col: str) -> np.ndarray:
        grid = np.full((12, 12), np.nan)
        for _, row in ch_stats.iterrows():
            loc = ch_to_loc.get(row['channel'])
            if loc is not None:
                r, c = loc
                if 0 <= r < 12 and 0 <= c < 12:
                    grid[r, c] = row[col]
        return grid

    grid_amp   = _build_grid('mean_amplitude')
    grid_lat   = _build_grid('mean_latency')
    grid_wid   = _build_grid('mean_width')
    grid_decay = _build_grid('amplitude_decay')

    # ── Channel with the highest decay constant ───────────────────
    valid = ch_stats.dropna(subset=['amplitude_decay'])
    highest_decay_ch = (
        valid.loc[valid['amplitude_decay'].idxmax(), 'channel']  # col renamed via agg alias
        if not valid.empty else None
    )

    # ── Stimulation electrode grid position ───────────────────────
    stim_pos: tuple[int, int] | None = None
    if rec.stim_channel_name:
        for ch_idx, ch_name in enumerate(rec.channel_names):
            if rec.stim_channel_name in ch_name:
                loc = ch_to_loc.get(ch_name)
                if loc is not None:
                    stim_pos = loc
                break

    # ── Colour scale limits ───────────────────────────────────────
    amp_max  = float(np.nanmax(grid_amp))  if np.any(~np.isnan(grid_amp))   else 1.0
    lat_min  = float(np.nanmin(grid_lat))  if np.any(~np.isnan(grid_lat))   else 0.0
    lat_max  = float(np.nanmax(grid_lat))  if np.any(~np.isnan(grid_lat))   else 1.0
    dec_max  = float(np.nanmax(grid_decay)) if np.any(~np.isnan(grid_decay)) else 5.0

    lat_vmin = int(np.floor(lat_min))
    lat_vmax = int(np.ceil(lat_max))

    import math
    dec_vmax = math.ceil(dec_max / 5) * 5   # round up to nearest 5

    # ── Figure / GridSpec ─────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_amp   = fig.add_subplot(gs[0, 0])
    ax_lat   = fig.add_subplot(gs[0, 1])
    ax_wid   = fig.add_subplot(gs[0, 2])
    ax_decay = fig.add_subplot(gs[1, 0])          # 1/3 width
    ax_trace = fig.add_subplot(gs[1, 1:])         # 2/3 width

    cmap = 'Purples'

    def _heatmap(ax, grid, title, unit, vmin, vmax):
        im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='upper',
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cbar.set_label(unit, fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        # Red dot at stimulation electrode
        if stim_pos is not None:
            r, c = stim_pos
            ax.plot(c, r, 'o', color='red', markersize=5, zorder=5,
                    markeredgewidth=0)

    _heatmap(ax_amp,   grid_amp,   'Mean Amplitude', 'µV', vmin=0,       vmax=amp_max)
    _heatmap(ax_lat,   grid_lat,   'Mean Latency',   'ms', vmin=lat_vmin, vmax=lat_vmax)
    _heatmap(ax_wid,   grid_wid,   'Mean Width',     'ms', vmin=0.2,     vmax=1.0)
    _heatmap(ax_decay, grid_decay, 'Amplitude Decay','%',  vmin=0,       vmax=dec_vmax)

    # ── Amplitude vs pulse — highest-decay channel ────────────────
    if highest_decay_ch is not None:
        ch_df      = (df[df['channel'] == highest_decay_ch]
                      .sort_values('pulse_index'))
        pulse_nums = ch_df['pulse_index'].values
        amps       = ch_df['amplitude_mV'].abs().values
        pct_val    = float(
            ch_stats.loc[
                ch_stats['channel'] == highest_decay_ch,
                'amplitude_decay'
            ].values[0]
        )

        ax_trace.scatter(pulse_nums, amps,
                         color='#b39ddb', s=18, zorder=3)
        ax_trace.set_xlabel('Pulse #', fontsize=9)
        ax_trace.set_ylabel('|Amplitude| (µV)', fontsize=9)
        ax_trace.set_title(
            f'Amplitude decay — {highest_decay_ch}  '
            f'(highest decay: {pct_val:.1f}%)',
            fontsize=9,
        )
        ax_trace.spines[['top', 'right']].set_visible(False)
    else:
        ax_trace.text(0.5, 0.5, 'No decay data available',
                      ha='center', va='center',
                      transform=ax_trace.transAxes, fontsize=10)
        ax_trace.set_axis_off()

    plt.suptitle(
        f'{rec.file_name} — Direct Response Summary', fontsize=11
    )

    if save and output_folder:
        fig.savefig(
            f'{output_folder}/direct_response_summary.png',
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)
    else:
        plt.show()