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
# With retina and response:
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina1\Ch05_300us_50us_7uA_1Hz_250528_092146\Ch05_300us_50us_7uA_1Hz_250528_092146.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch01_300us_50us_7uA_1Hz_250528_113143\Ch01_300us_50us_7uA_1Hz_250528_113143.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch04_300us_50us_6uA_1Hz_250528_112626\Ch04_300us_50us_6uA_1Hz_250528_112626.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch04_300us_50us_7uA_1Hz_250528_112207\Ch04_300us_50us_7uA_1Hz_250528_112207.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch04_300us_50us_7uA_1Hz_250528_121450\Ch04_300us_50us_7uA_1Hz_250528_121450.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina5\Ch04_300us_50us_7uA_1Hz_250528_142150\Ch04_300us_50us_7uA_1Hz_250528_142150.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.25 E14\Retina1\7uA\Ch04_300us_50us_1Hz_250525_095035\Ch04_300us_50us_1Hz_250525_095135.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.25 E14\Retina4\Ch04_300us_50us_7uA_1Hz_250525_131107\Ch04_300us_50us_7uA_1Hz_250525_131107.rhs'

# 10 Hz
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina1\Ch05_300us_50us_7uA_10Hz_250528_092403\Ch05_300us_50us_7uA_10Hz_250528_092403.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch01_300us_50us_7uA_10Hz_250528_113243\Ch01_300us_50us_7uA_10Hz_250528_113243.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina5\Ch04_300us_50us_7uA_10Hz_250528_142229\Ch04_300us_50us_7uA_10Hz_250528_142229.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.25 E14\Retina4\Ch04_300us_50us_7uA_10Hz_250525_131156\Ch04_300us_50us_7uA_10Hz_250525_131156.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.25 E14\Retina1\7uA\Ch04_300us_50us_10Hz_250525_094851\Ch04_300us_50us_10Hz_250525_094851.rhs'

# 20 Hz
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina1\Ch05_300us_50us_7uA_20Hz_250528_092432\Ch05_300us_50us_7uA_20Hz_250528_092432.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina3\Ch01_300us_50us_7uA_10Hz_250528_113243\Ch01_300us_50us_7uA_10Hz_250528_113243.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.28 E14\Retina5\Ch04_300us_50us_7uA_20Hz_250528_142312\Ch04_300us_50us_7uA_20Hz_250528_142312.rhs'
# RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.25 E14\Retina4\Ch04_300us_50us_7uA_20Hz_250525_131217\Ch04_300us_50us_7uA_20Hz_250525_131217.rhs'
RHS_FILE1            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.05.25 E14\Retina1\7uA\Ch04_300us_50us_20Hz_250525_095009\Ch04_300us_50us_20Hz_250525_095009.rhs'

#
# No retina:
# RHS_FILE1            = r'S:\shani_data\Intact\2025.11.18 No retina\Stim_ch04\1Hz_251118_161301\1Hz_251118_161301.rhs' # probe is floating
# RHS_FILE1            = r'S:\shani_data\Intact\2025.11.18 No retina\Stim_ch01\1Hz_251118_161942\1Hz_251118_161942.rhs' # probe is floating
# RHS_FILE1            = r'S:\shani_data\Intact\2025.11.19 E14\No retina\Ch04_300us_50us_7uA_1Hz_251119_123706\Ch04_300us_50us_7uA_1Hz_251119_123706.rhs' # On choroid with magnet


# Fading examples
# EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf'
# EDF_STIM_ELECTRODE  = 'J6'
# EDF_FILE            = r'S:\shani_data\Ex-vivo\2024.10.26 e14_Shani\Fading test\No noise\2024-10-27T13-33-16No_noise_stimulation_3uA_B-00071.edf'
# EDF_STIM_ELECTRODE  = 'G6'

#10Hz
# EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-57-46J9_10uA_300us_60us_10Hz_100pulses.edf'
# EDF_STIM_ELECTRODE  = 'J9'
# EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-22-027uA_300us_60us_10Hz_100pulse_B-00071.edf'
# EDF_STIM_ELECTRODE  = 'G10'
# EDF_FILE            = r'C:\Shani\MEA mini1200\2024.11.20 e14_Ieva\2024-11-20T14-45-08_40uA _g6_10Hz_100Pulses_B-00071.edf'
# EDF_STIM_ELECTRODE  = 'G6'

#20Hz
EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-07-34J9_20uA_300us_60us_20Hz_100pulses.edf'
EDF_STIM_ELECTRODE  = 'J9'

# Indirect response
# EDF_FILE            = r'C:\Shani\MEA mini1200\2024.11.12 e14_Shani\D5\No Noise 2\2024-11-12T12-11-02No_noise_stimulation_20uA_800us_1Hz_B-00071.edf'
# EDF_STIM_ELECTRODE  = 'D5'

# --- Direct response ---
DIRECT_WIN_MS       = 10.0
BLANK_MS     = 2.5
DIRECT_THRESHOLD_MV    = None  # set to None to compute threshold from data

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
    edf_rec.direct_response.to_csv(os.path.join(RESULTS_DIR, f'{edf_rec.file_name}_direct_response.csv'), index=False)

    # plot_spikes_layout_mea(rec=edf_rec,
    #                     win_size_ms=DIRECT_WIN_MS,
    #                     data_type = 'raw',
    #                     threshold=DIRECT_THRESHOLD_MV,
    #                     output_folder=RESULTS_DIR)

    # plot_direct_response_summary(rec=edf_rec, output_folder=RESULTS_DIR)

    # edf_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS, threshold_std=INDIRECT_THRESH_STD)
    # plot_indirect_response_raster(rec=edf_rec, save=True, output_folder=RESULTS_DIR)

    # edf_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS, threshold_std=SPONT_THRESH_STD)
    #
    # print(edf_rec)


    # ── Intan RHS ────────────────────────────────────────────
    # print('=' * 60)
    # print('Intan RHS')
    # print('=' * 60)
    # rhs_rec1 = load_rhs(RHS_FILE1, stim_threshold=470)
    #
    # # rhs_rec1.blank(duration_ms=BLANK_MS, source='raw_data')
    # rhs_rec1.detect_direct_response(win_size_ms=10,
    #                                threshold=15,
    #                                data_type='raw',
    #                                plot=True,
    #                                output_folder=RESULTS_DIR)
    # rhs_rec1.direct_response.to_csv(os.path.join(RESULTS_DIR, f'{rhs_rec1.file_name}_direct_response.csv'), index=False)

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