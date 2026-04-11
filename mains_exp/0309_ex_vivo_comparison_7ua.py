"""
IntactRetinaToolkit — main.py
==============================
Loads one Intan (.rhs) and one MEA (.edf) recording and runs analysis.
Edit the params below and run:
    python main.py
"""

import os

import matplotlib.cm as cm

from dataobj import load_rhs, load_edf
from dataviz.viz import *
from datahelper.statistics import compare_direct_responses


# ============================================================
#  COMPARISON FUNCTION
# ============================================================

def response_comparison_between_mea_and_probe(
    edf_rec,
    rhs_rec,
    mea_channel: str = 'E11',
    probe_channel: str = '0',
    save: bool = True,
    output_folder: str = 'Results',
) -> None:
    """
    Compare direct-response properties of one MEA channel vs one probe channel.

    Values are normalized to each group's own mean (relative normalization)
    so the two recording systems can be compared on the same scale.
    A dashed reference line marks mean = 1.

    Generates a 1×3 figure with purple boxplots:
        • Normalized |Amplitude|
        • Normalized Width
        • Normalized Latency

    Parameters
    ----------
    edf_rec : RetinalRecording (source='edf')
    rhs_rec : RetinalRecording (source='rhs')
    mea_channel : str
        Channel label to select from edf_rec.direct_response (default 'E11').
    probe_channel : str
        Channel label to select from rhs_rec.direct_response (default '0').
    save : bool
        Save to output_folder when True, show interactively otherwise.
    output_folder : str
        Directory for the saved figure.
    """
    mea_df   = edf_rec.direct_response
    probe_df = rhs_rec.direct_response

    # Filter to the requested channels
    if 'channel' in mea_df.columns:
        mea_df = mea_df[mea_df['channel'] == mea_channel]
    if 'channel' in probe_df.columns:
        probe_df = probe_df[probe_df['channel'] == probe_channel]

    purples = cm.get_cmap('Purples')
    colors  = [purples(0.45), purples(0.75)]   # MEA = lighter, Probe = darker
    labels  = [f'MEA ({mea_channel})', f'Probe (ch{probe_channel})']

    params = [
        ('amplitude_mV', 'Normalized |Amplitude|', True),
        ('width_ms',     'Normalized Width',        False),
        ('latency_ms',   'Normalized Latency',      False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    plt.subplots_adjust(wspace=0.45, left=0.09, right=0.97,
                        top=0.86, bottom=0.18)

    for ax, (col, ylabel, use_abs) in zip(axes, params):
        groups = []
        for df in (mea_df, probe_df):
            if col not in df.columns or df.empty:
                groups.append(np.array([np.nan]))
            else:
                v = df[col].dropna().values
                v = np.abs(v) if use_abs else v
                mean = v.mean()
                groups.append(v / mean if mean != 0 else v)

        bp = ax.boxplot(
            groups,
            positions=[1, 2],
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=0.8, color='#555555'),
            capprops=dict(linewidth=0),
            flierprops=dict(marker='o', markersize=2.5, alpha=0.35,
                            markeredgewidth=0, linestyle='none'),
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_linewidth(0.8)
        for flier, color in zip(bp['fliers'], colors):
            flier.set_markerfacecolor(color)

        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axhline(1.0, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', length=0)
        ax.yaxis.grid(True, linewidth=0.4, color='#dddddd', zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        f'MEA ({mea_channel}) vs Probe (ch{probe_channel}) — Normalized Response Comparison',
        fontsize=10, y=0.97,
    )

    patches = [mpatches.Patch(color=c, alpha=0.85, label=lbl)
               for c, lbl in zip(colors, labels)]
    fig.legend(handles=patches, loc='upper right', fontsize=7,
               frameon=False, borderaxespad=0.5)

    if save and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(f'{output_folder}/mea_vs_probe_response_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

RESULTS_DIR = 'Results'

# ============================================================
#  PARAMS
# ============================================================

RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\For figures\Stim with MEA\New folder\ChE11_20uA_300us_50us_1Hz_250309_172137.rhs'
RHS_FILE2           = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\Ch27_7uA_300us_50us_1Hz_256pulses_250309_190326.rhs'
EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf'
EDF_STIM_ELECTRODE  = 'J6'

# --- Direct response ---
DIRECT_WIN_MS       = 10.0
BLANK_MS     = 1.5
DIRECT_THRESHOLD_MV    = 0.2  # set to None to compute threshold from data

# --- Indirect response ---
INDIRECT_BLANK_MS   = 15.0
INDIRECT_THRESH_STD = 4.0

# --- Spontaneous ---
SPONT_MIN_DUR_MS    = 5.0
SPONT_THRESH_STD    = 4.0

# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── MEA EDF ──────────────────────────────────────────────
    print()
    print('=' * 60)
    print('MEA EDF')
    print('=' * 60)
    edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
    edf_rec.filter()
    edf_rec.blank(duration_ms=BLANK_MS, source='filtered_data')

    edf_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, threshold=DIRECT_THRESHOLD_MV)
    plot_spikes_layout_mea(rec=edf_rec,
                        win_size_ms=DIRECT_WIN_MS,
                        data_type = 'blanked',
                        threshold=DIRECT_THRESHOLD_MV,
                        output_folder=RESULTS_DIR)

    plot_direct_response_summary(rec=edf_rec, output_folder=RESULTS_DIR)

    # edf_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS, threshold_std=INDIRECT_THRESH_STD)

    # edf_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS, threshold_std=SPONT_THRESH_STD)

    # print(edf_rec)


    # ── Intan RHS ────────────────────────────────────────────
    print('=' * 60)
    print('Intan RHS')
    print('=' * 60)
    rhs_rec1 = load_rhs(RHS_FILE1, stim_threshold=470)

    rhs_rec1.blank(duration_ms=BLANK_MS, source='raw_data')
    rhs_rec1.detect_direct_response(win_size_ms=10,
                                   threshold=15,
                                   data_type='blanked')

    plot_spikes_layout_probe16(rec=rhs_rec1,
                               win_size_ms=20,
                               data_type='blanked',
                               threshold=15,
                               output_folder=RESULTS_DIR)

    # rhs_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS,
    #                                  threshold_std=INDIRECT_THRESH_STD)
    #
    # rhs_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS,
    #                            threshold_std=SPONT_THRESH_STD)



    rhs_rec2 = load_rhs(RHS_FILE2, stim_threshold=470)

    rhs_rec2.blank(duration_ms=BLANK_MS, source='raw_data')
    rhs_rec2.detect_direct_response(win_size_ms=10,
                                   threshold=15,
                                   data_type='blanked')

    # ── Cross-recording comparison ────────────────────────────
    compare_direct_responses(
        dfs=[edf_rec.direct_response, rhs_rec2.direct_response],
        labels=['MEA', 'Probe'],
        channels=[['E_B-00071 F11'], ['0']],
        output_folder=RESULTS_DIR,
    )

    response_comparison_between_mea_and_probe(
        edf_rec=edf_rec,
        rhs_rec=rhs_rec2,
        mea_channel='E11',
        probe_channel='0',
        output_folder=RESULTS_DIR,
    )