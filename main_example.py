"""
IntactRetinaToolkit — main_example.py
=======================================
Minimal example: load one Intan (.rhs) and one MEA (.edf) recording,
run direct-response detection on each, and compare the results.

Edit the file paths and parameters below, then run:
    python main_example.py
"""

import os
from dataobj import load_rhs, load_edf
from dataviz.viz import plot_spikes_layout_probe16, plot_spikes_layout_mea, plot_direct_response_summary
from datahelper.statistics import compare_direct_responses

# ── Output ───────────────────────────────────────────────────────────────────

RESULTS_DIR = 'Results'

# ── File paths ────────────────────────────────────────────────────────────────

RHS_FILE = r'C:\path\to\your_file.rhs'

EDF_FILE           = r'C:\path\to\your_file.edf'
EDF_STIM_ELECTRODE = 'J6'           # electrode label used for stimulation

# ── Shared parameters ─────────────────────────────────────────────────────────

DIRECT_WIN_MS  = 10.0   # analysis window per pulse (ms)
BLANK_MS       = 2.5    # blanking duration after stim onset (ms)

# ── RHS-specific ──────────────────────────────────────────────────────────────

RHS_STIM_THRESHOLD = 470   # fallback threshold if no stim signal found in file

# ── EDF-specific ──────────────────────────────────────────────────────────────

EDF_DIRECT_THRESHOLD_MV = 0.3   # detection threshold (mV); None = auto

# ── Indirect / spontaneous ────────────────────────────────────────────────────

INDIRECT_BLANK_MS   = 15.0
INDIRECT_THRESH_STD = 4.0
SPONT_MIN_DUR_MS    = 5.0
SPONT_THRESH_STD    = 4.0


# =============================================================================
if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Intan RHS ─────────────────────────────────────────────────────────────
    print('=' * 60)
    print('Intan RHS')
    print('=' * 60)

    rhs_rec = load_rhs(RHS_FILE, stim_threshold=RHS_STIM_THRESHOLD)
    rhs_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, data_type='raw')

    plot_spikes_layout_probe16(
        rec=rhs_rec,
        win_size_ms=DIRECT_WIN_MS,
        data_type='raw',
        output_folder=RESULTS_DIR,
    )
    plot_direct_response_summary(rec=rhs_rec, output_folder=RESULTS_DIR)

    rhs_rec.detect_indirect_response(
        blanking_ms=INDIRECT_BLANK_MS,
        threshold_std=INDIRECT_THRESH_STD,
    )
    rhs_rec.detect_spontaneous(
        min_duration_ms=SPONT_MIN_DUR_MS,
        threshold_std=SPONT_THRESH_STD,
    )
    print(rhs_rec)

    # ── MEA EDF ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('MEA EDF')
    print('=' * 60)

    edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
    edf_rec.filter()
    edf_rec.blank(duration_ms=BLANK_MS, source='filtered_data')
    edf_rec.detect_direct_response(
        win_size_ms=DIRECT_WIN_MS,
        threshold=EDF_DIRECT_THRESHOLD_MV,
    )

    plot_spikes_layout_mea(
        rec=edf_rec,
        win_size_ms=DIRECT_WIN_MS,
        data_type='raw',
        threshold=EDF_DIRECT_THRESHOLD_MV,
        output_folder=RESULTS_DIR,
    )
    plot_direct_response_summary(rec=edf_rec, output_folder=RESULTS_DIR)

    edf_rec.detect_indirect_response(
        blanking_ms=INDIRECT_BLANK_MS,
        threshold_std=INDIRECT_THRESH_STD,
    )
    edf_rec.detect_spontaneous(
        min_duration_ms=SPONT_MIN_DUR_MS,
        threshold_std=SPONT_THRESH_STD,
    )
    print(edf_rec)

    # ── Comparison ────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('Comparison')
    print('=' * 60)

    compare_direct_responses(
        dfs=[rhs_rec.direct_response, edf_rec.direct_response],
        labels=['Probe (RHS)', 'MEA (EDF)'],
        output_folder=RESULTS_DIR,
    )