"""
IntactRetinaToolkit — main.py
==============================
Loads one Intan (.rhs) and one MEA (.edf) recording and runs analysis.
Edit the params below and run:
    python main.py
"""

import os

from dataobj import load_rhs, load_edf
from dataviz.viz import *
from datahelper.statistics import compare_direct_responses

RESULTS_DIR = 'Results_glucose'

# ============================================================
#  PARAMS
# ============================================================
# With retina and response:
#phase1
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase1\Ch4_300us_50us_7uA_1Hz_100pulses_251105_155444\Ch4_300us_50us_7uA_1Hz_100pulses_251105_155444.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase1\Ch4_300us_50us_7uA_10Hz_100pulses_251105_160302\Ch4_300us_50us_7uA_10Hz_100pulses_251105_160302.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase1\Ch4_300us_50us_7uA_20Hz_100pulses_251105_155959\Ch4_300us_50us_7uA_20Hz_100pulses_251105_155959.rhs'
# phase2
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase2\Ch4_300us_50us_7uA_1Hz_100pulses_251105_165652\Ch4_300us_50us_7uA_1Hz_100pulses_251105_165652.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase2\Ch4_300us_50us_7uA_10Hz_100pulses_251105_165850\Ch4_300us_50us_7uA_10Hz_100pulses_251105_165850.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase2\Ch4_300us_50us_7uA_20Hz_100pulses_251105_170036\Ch4_300us_50us_7uA_20Hz_100pulses_251105_170036.rhs'
#phase3
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase3\Ch4_300us_50us_7uA_1Hz_100pulses_251105_174506\Ch4_300us_50us_7uA_1Hz_100pulses_251105_174506.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase3\Ch4_300us_50us_7uA_10Hz_100pulses_251105_174700\Ch4_300us_50us_7uA_10Hz_100pulses_251105_174700.rhs'
RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2\phase3\Ch4_300us_50us_7uA_20Hz_100pulses_251105_174841\Ch4_300us_50us_7uA_20Hz_100pulses_251105_174841.rhs'

# Fading examples
# EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf'
# EDF_STIM_ELECTRODE  = 'J6'


# --- Direct response ---
DIRECT_WIN_MS       = 10.0
BLANK_MS     = 2.5
DIRECT_THRESHOLD_MV    = 0.3  # set to None to compute threshold from data

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
    # print()
    # print('=' * 60)
    # print('MEA EDF')
    # print('=' * 60)
    # edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
    # edf_rec.filter()
    # edf_rec.blank(duration_ms=BLANK_MS, source='filtered_data')
    #
    # edf_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, threshold=DIRECT_THRESHOLD_MV)
    # edf_rec.direct_response.to_csv(os.path.join(RESULTS_DIR, f'{edf_rec.file_name}_direct_response.csv'), index=False)

    # plot_spikes_layout_mea(rec=edf_rec,
    #                     win_size_ms=DIRECT_WIN_MS,
    #                     data_type = 'raw',
    #                     threshold=DIRECT_THRESHOLD_MV,
    #                     output_folder=RESULTS_DIR)

    # plot_direct_response_summary(rec=edf_rec, output_folder=RESULTS_DIR)

    # edf_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS, threshold_std=INDIRECT_THRESH_STD)
    #
    # edf_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS, threshold_std=SPONT_THRESH_STD)
    #
    # print(edf_rec)


    # ── Intan RHS ────────────────────────────────────────────
    print('=' * 60)
    print('Intan RHS')
    print('=' * 60)
    rhs_rec1 = load_rhs(RHS_FILE1, stim_threshold=470)

    # rhs_rec1.blank(duration_ms=BLANK_MS, source='raw_data')
    rhs_rec1.detect_direct_response(win_size_ms=10,
                                   threshold=15,
                                   data_type='raw',
                                   plot=True,
                                   output_folder=RESULTS_DIR)
    rhs_rec1.direct_response.to_csv(os.path.join(RESULTS_DIR, f'{rhs_rec1.file_name}_direct_response.csv'), index=False)

    # plot_spikes_layout_probe16(rec=rhs_rec1,
    #                            win_size_ms=20,
    #                            data_type='raw',
    #                            threshold=15,
    #                            output_folder=RESULTS_DIR)

    # rhs_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS,
    #                                  threshold_std=INDIRECT_THRESH_STD)

    # rhs_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS,
    #                            threshold_std=SPONT_THRESH_STD)

    #
    #
    # rhs_rec2 = load_rhs(RHS_FILE2, stim_threshold=470)
    #
    # rhs_rec2.blank(duration_ms=BLANK_MS, source='raw_data')
    # rhs_rec2.detect_direct_response(win_size_ms=10,
    #                                threshold=15,
    #                                data_type='blanked')
    #
    # # ── Cross-recording comparison ────────────────────────────
    # compare_direct_responses(
    #     dfs=[rhs_rec1.direct_response, rhs_rec2.direct_response],
    #     labels=['MEA stim', 'Probe stim'],
    #     channels=[None, ['0', '1', '2', '30', '31']],
    #     output_folder=RESULTS_DIR,
    # )

    # ── Glucose experiment: amplitude vs pulse by phase and frequency ──
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    GLUCOSE_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\mains_exp\Results\Results_glucose'
    phases      = ['phase1', 'phase2', 'phase3']
    freq_keys   = ['1Hz', '10Hz', '20Hz']
    freq_colors = {
        '1Hz':  {'scatter': '#404040', 'line': 'black',   'shade': '#b0b0b0'},
        '10Hz': {'scatter': '#9b59b6', 'line': '#7b2d8b', 'shade': '#c39bd3'},
        '20Hz': {'scatter': '#5b9bd5', 'line': '#1f77b4', 'shade': '#aec7e8'},
    }

    def _freq_from_name(fname):
        f = fname.lower()
        for hz in ('20hz', '10hz', '1hz'):
            if f'_{hz}_' in f:
                return {'20hz': '20Hz', '10hz': '10Hz', '1hz': '1Hz'}[hz]
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, phase in zip(axes, phases):
        phase_dir = os.path.join(GLUCOSE_DIR, phase)
        csv_files = sorted(glob.glob(os.path.join(phase_dir, '*_direct_response.csv')))

        data_by_freq = {f: [] for f in freq_keys}

        for path in csv_files:
            freq = _freq_from_name(os.path.basename(path))
            if freq is None:
                continue
            df    = pd.read_csv(path)
            valid = df.dropna(subset=['amplitude_mV', 'pulse_index']).copy()
            if valid.empty:
                continue
            valid['amplitude_uV'] = valid['amplitude_mV'].abs() * 1000
            ch6 = valid[valid['channel'].astype(str) == '6']
            if ch6.empty:
                continue
            data_by_freq[freq].append(ch6[['pulse_index', 'amplitude_uV']])

        for freq in freq_keys:
            if not data_by_freq[freq]:
                continue
            clr  = freq_colors[freq]
            data = pd.concat(data_by_freq[freq], ignore_index=True)

            ax.scatter(data['pulse_index'].values, data['amplitude_uV'].values,
                       color=clr['scatter'], s=12, alpha=0.6, linewidths=0, label=freq)

        ax.set_title(phase, fontsize=10)
        ax.set_xlabel('Pulse #', fontsize=9)
        ax.set_xlim(left=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel('|Amplitude| (µV)', fontsize=10)
    axes[-1].legend(fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(GLUCOSE_DIR, 'glucose_amplitude_vs_pulse_by_phase.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved glucose figure → {GLUCOSE_DIR}/glucose_amplitude_vs_pulse_by_phase.png")