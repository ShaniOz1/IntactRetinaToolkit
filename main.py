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

RESULTS_DIR = 'Results'

# ============================================================
#  PARAMS
# ============================================================

RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\For figures\Stim with MEA\New folder\ChE11_20uA_300us_50us_1Hz_250309_172137.rhs'
RHS_FILE2           = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\Ch27_7uA_300us_50us_1Hz_256pulses_250309_190326.rhs'
EDF_FILE            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\For figures\Stim with MEA\New folder\id1 ChE11_20uA_300us_50us_1Hz_100pulsesB-00071.edf'
EDF_STIM_ELECTRODE  = 'E11'

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
    # plot_spikes_layout_mea(rec=edf_rec,
    #                     win_size_ms=DIRECT_WIN_MS,
    #                     data_type = 'blanked',
    #                     threshold=DIRECT_THRESHOLD_MV,
    #                     output_folder=RESULTS_DIR)

    # plot_direct_response_summary(rec=edf_rec, output_folder=RESULTS_DIR)

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

    # rhs_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS,
    #                            threshold_std=SPONT_THRESH_STD)



    rhs_rec2 = load_rhs(RHS_FILE2, stim_threshold=470)

    rhs_rec2.blank(duration_ms=BLANK_MS, source='raw_data')
    rhs_rec2.detect_direct_response(win_size_ms=10,
                                   threshold=15,
                                   data_type='blanked')

    # ── Cross-recording comparison ────────────────────────────
    compare_direct_responses(
        dfs=[rhs_rec1.direct_response, rhs_rec2.direct_response],
        labels=['MEA stim', 'Probe stim'],
        channels=[None, ['0', '1', '2', '30', '31']],
        output_folder=RESULTS_DIR,
    )