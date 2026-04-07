"""
IntactRetinaToolkit — main.py
==============================
Loads one Intan (.rhs) and one MEA (.edf) recording and runs analysis.
Edit the params below and run:
    python main.py
"""

from dataobj import load_rhs, load_edf

# ============================================================
#  PARAMS
# ============================================================

RHS_FILE            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\For figures\Stim with MEA\New folder\ChE11_20uA_300us_50us_1Hz_250309_172137.rhs'
EDF_FILE            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.03.09 E14\For figures\Stim with MEA\New folder\id1 ChE11_20uA_300us_50us_1Hz_100pulsesB-00071.edf'
EDF_STIM_ELECTRODE  = 'E11'

# --- Direct response ---
DIRECT_WIN_MS       = 10.0
DIRECT_BLANK_MS     = 1.0

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

    # ── MEA EDF ──────────────────────────────────────────────
    print()
    print('=' * 60)
    print('MEA EDF')
    print('=' * 60)
    edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
    edf_rec.filter()
    edf_rec.blank(source='filtered_data')

    edf_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, blank_ms=DIRECT_BLANK_MS)

    edf_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS, threshold_std=INDIRECT_THRESH_STD)

    # edf_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS, threshold_std=SPONT_THRESH_STD)

    print(edf_rec)


    # ── Intan RHS ────────────────────────────────────────────
    print('=' * 60)
    print('Intan RHS')
    print('=' * 60)
    rhs_rec = load_rhs(RHS_FILE)
    print(rhs_rec)

    rhs_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                   blank_ms=DIRECT_BLANK_MS)
    print(rhs_rec.direct_response)

    rhs_rec.detect_indirect_response(blanking_ms=INDIRECT_BLANK_MS,
                                     threshold_std=INDIRECT_THRESH_STD)
    print(rhs_rec.indirect_response)

    # rhs_rec.detect_spontaneous(min_duration_ms=SPONT_MIN_DUR_MS,
    #                            threshold_std=SPONT_THRESH_STD)
    # print(rhs_rec.spontaneous)

    print(rhs_rec)