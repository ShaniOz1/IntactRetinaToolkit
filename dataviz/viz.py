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
plot_overlay_pulses       — raw pulse overlays (single obj) or averages (multi-obj comparison)
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