"""
IntactRetinaToolkit — main.py
==============================
Loads one Intan (.rhs) and one MEA (.edf) recording and prints a summary
of each. Edit the params below and run:
    python main.py
"""

from dataobj import load_rhs, load_edf

# ============================================================
#  PARAMS
# ============================================================

RHS_FILE            = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.01.08 E14\Retina2\Recordings on Retina\Ch2_300us_50us_15uA_1Hz_20pulse_250108_155301\Ch2_300us_50us_15uA_1Hz_20pulse_250108_155301.rhs'
EDF_FILE            = r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-12-26J6_10uA_300us_60us_20Hz_100pulses.edf'
EDF_STIM_ELECTRODE  = 'G6'     # grid label of the stim electrode, or None

Apply_filter        = True
Impedance_check     = False
Direct_response     = True
Indirect_response   = False
Spontaneous         = False

# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':

    print('Loading Intan RHS...')
    rhs_rec = load_rhs(RHS_FILE)

    print('Loading MEA EDF...')
    edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
    print(edf_rec)
