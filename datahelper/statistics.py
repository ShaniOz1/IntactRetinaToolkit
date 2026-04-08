"""
IntactRetinaToolkit.datahelper.statistics
==========================================
Statistical comparison utilities across recordings.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Parameters to compare: (column, y-axis label, take abs value)
_PARAMS: list[tuple[str, str, bool]] = [
    ('amplitude_mV',       '|Amplitude| [mV]',     True),
    ('latency_ms',         'Latency [ms]',          False),
    ('width_ms',           'Width [ms]',            False),
    ('amplitude_decay_pct','Amplitude Decay [%]',   False),
]


def compare_direct_responses(
    dfs: list[pd.DataFrame],
    labels: list[str] | None = None,
    channels: list[list[str] | None] | None = None,
    save: bool = True,
    output_folder: str | None = None,
) -> None:
    """
    Compare direct-response parameters across 1–4 recordings.

    Generates a 1×4 figure with one box plot per parameter
    (amplitude, latency, width, amplitude_decay), each showing the
    distribution for every supplied recording side by side.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        1–4 direct-response DataFrames (from rec.direct_response).
    labels : list[str] | None
        Display names for each recording.  Defaults to 'Recording 1', etc.
    channels : list[list[str] | None] | None
        Per-recording channel filter.  Must have the same length as ``dfs``
        when provided.  Each entry is either a list of channel names to keep,
        or None to include all channels for that recording.
    save : bool
        Save to output_folder when True.  Show interactively otherwise.
    output_folder : str | None
        Directory for the saved figure.
    """
    if not 1 <= len(dfs) <= 4:
        raise ValueError(f"Expected 1–4 DataFrames, got {len(dfs)}.")

    n = len(dfs)
    if labels is None:
        labels = [f'Recording {i + 1}' for i in range(n)]
    if len(labels) != n:
        raise ValueError("len(labels) must match len(dfs).")

    if channels is not None:
        if len(channels) != n:
            raise ValueError("len(channels) must match len(dfs).")
        dfs = [
            df[df['channel'].isin(ch)] if (ch is not None and 'channel' in df.columns) else df
            for df, ch in zip(dfs, channels)
        ]

    import matplotlib.cm as cm
    _purples = cm.get_cmap('Purples')
    # Spread colors evenly across the useful part of the Purples ramp
    color_stops = [0.35 + 0.55 * i / max(n - 1, 1) for i in range(n)]
    colors = [_purples(s) for s in color_stops]

    positions = list(range(1, n + 1))
    box_width = max(0.25, 0.6 - 0.08 * n)

    fig, axes = plt.subplots(1, len(_PARAMS), figsize=(3 * len(_PARAMS), 4))
    plt.subplots_adjust(wspace=0.5, left=0.09, right=0.97,
                        top=0.86, bottom=0.15)

    for ax, (col, ylabel, use_abs) in zip(axes, _PARAMS):
        groups = []
        for df in dfs:
            if col not in df.columns or df.empty:
                groups.append(np.array([np.nan]))
            else:
                v = df[col].dropna().values
                groups.append(np.abs(v) if use_abs else v)

        bp = ax.boxplot(
            groups,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=0.8, color='#555555'),
            capprops=dict(linewidth=0),          # hide caps
            flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                            markeredgewidth=0, linestyle='none'),
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_linewidth(0.8)
        for flier, color in zip(bp['fliers'], colors):
            flier.set_markerfacecolor(color)

        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=20, ha='right')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', length=0)
        ax.yaxis.grid(True, linewidth=0.4, color='#dddddd', zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle('Direct Response — Recording Comparison', fontsize=10, y=0.97)

    patches = [mpatches.Patch(color=c, alpha=0.85, label=lbl)
               for c, lbl in zip(colors, labels)]
    fig.legend(handles=patches, loc='upper right', fontsize=7,
               frameon=False, borderaxespad=0.5)

    if save and output_folder:
        fig.savefig(f'{output_folder}/direct_response_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
