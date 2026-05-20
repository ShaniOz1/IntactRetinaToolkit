"""
IntactRetinaToolkit — main.py
==============================
Loads one Intan (.rhs) and one MEA (.edf) recording and runs analysis.
Edit the params below and run:
    python main.py
"""

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

from dataobj import load_rhs, load_edf
from dataviz.viz import *
from datahelper.statistics import compare_direct_responses
from dataobj.analysis.direct import _compute_threshold

RESULTS_DIR = r'/Results/ex_vivo_all'

# ============================================================
#  PARAMS
# ============================================================

# --- Direct response ---
DIRECT_WIN_MS       = 10.0
BLANK_MS            = 1.0
DIRECT_THRESHOLD_MV = 0.2  # set to None to compute threshold from data
INTERACTIVE         = True  # set to True to review/adjust threshold per file before saving

# ============================================================
#  FILES TO RUN  (path, stim_electrode)
# ============================================================

FILES = [
    # ── Group 2 · 2024.11.17 Direct Response - Fading · G6 ──────────────────
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T12-53-10No_noise_stimulation_4uA_1Hz_200pulses_B-00071.edf',            'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T12-57-14No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',           'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-02-36No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',           'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-05-52Noise_1-10Hz_0.2mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-07-07Noise_1-10Hz_0.1mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-08-03No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',           'G6'),
    # (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-09-21Noise_1-10Hz_0.05mA_stimulation_4uA_10Hz_200pulses_B-00071.edf','G6'),
    #
    # #
    # # ── Group 4 · 2024.11.20 Ieva · G6 ─────────────────────────────────────
    # (r'C:\Shani\MEA mini1200\2024.11.20 e14_Ieva\2024-11-20T14-45-08_40uA _g6_10Hz_100Pulses_B-00071.edf', 'G6'),
    #
    # # ── Group 6 · 2025.11.02 Retina2 phase1-normal · J6 ────────────────────
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf',   'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-59-46J6_7uA_300us_60us_10Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-01-53J6_7uA_300us_60us_20Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-05-56J6_7uA_300us_60us_20Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-06-13J6_7uA_300us_60us_20Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-07-37J6_10uA_300us_60us_1Hz_100pulses.edf',  'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-10-01J6_10uA_300us_60us_10Hz_100pulses.edf', 'J6'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-12-26J6_10uA_300us_60us_20Hz_100pulses.edf', 'J6'),
    #
    # # ── Group 10 · 2025.11.02 Retina3 phase1-normal · J9 ───────────────────
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-55-12J9_10uA_300us_60us_1Hz_100pulses.edf',    'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-57-46J9_10uA_300us_60us_10Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-00-18J9_10uA_300us_60us_20Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-02-50J9_10uA_300us_60us_50Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-03-33J9_20uA_300us_60us_1Hz_100pulses.edf',    'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-06-09J9_20uA_300us_60us_10Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-07-34J9_20uA_300us_60us_20Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-08-27J9_20uA_300us_60us_50Hz_100pulses.edf',   'J9'),
    # (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-13-18J9_20uA_300us_60us_100Hz_1000pulses.edf', 'J9'),
    #
    # # ── Group 12 · 2025.11.12 Retina1 Phase1-Normal · G10 ──────────────────
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-19-197uA_300us_60us_1Hz_100pulse_B-00071.edf',  'G10'),
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-22-027uA_300us_60us_10Hz_100pulse_B-00071.edf', 'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-22-347uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-23-087uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-25-147uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    # (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-26-217uA_300us_60us_50Hz_100pulse_B-00071.edf', 'G10'),
]

# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    failed = []

    for i, (EDF_FILE, EDF_STIM_ELECTRODE) in enumerate(FILES, 1):
        print(f'\n[{i}/{len(FILES)}] {os.path.basename(EDF_FILE)}  (stim={EDF_STIM_ELECTRODE})')
        try:
            edf_rec = load_edf(EDF_FILE, stim_electrode=EDF_STIM_ELECTRODE)
            edf_rec.filter()
            edf_rec.blank(duration_ms=BLANK_MS, source='filtered_data')

            threshold = DIRECT_THRESHOLD_MV
            if INTERACTIVE:
                if threshold is None:
                    # Compute per-channel auto-thresholds and take the median so the
                    # dashed line is visible on the first plot.
                    per_ch = [
                        _compute_threshold(edf_rec.blanked_data[idx],
                                           edf_rec.stim_indices,
                                           edf_rec.sample_rate)
                        for idx in range(edf_rec.blanked_data.shape[0])
                    ]
                    threshold = float(np.median(per_ch))
                    print(f'    Auto threshold (median across channels): {threshold:.4f} mV')

                while True:
                    plot_spikes_layout_mea(rec=edf_rec,
                                           win_size_ms=DIRECT_WIN_MS,
                                           data_type='raw',
                                           threshold=threshold,
                                           output_folder=None)

                    thr_str = f'{threshold} mV'
                    print(f'    Threshold: {thr_str}')
                    ans = input('    Enter new threshold (mV) to re-plot, or press Enter to approve: ').strip()
                    plt.close('all')

                    if ans == '':
                        break
                    try:
                        threshold = float(ans)
                    except ValueError:
                        print('    Invalid — enter a number or press Enter to approve.')

            edf_rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, threshold=threshold)
            out_path = os.path.join(RESULTS_DIR, f'{edf_rec.file_name}_direct_response.csv')
            edf_rec.direct_response.to_csv(out_path, index=False)
            print(f'    saved → {out_path}')
        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(EDF_FILE)

    print(f'\n{"=" * 60}')
    print(f'Done. {len(FILES) - len(failed)}/{len(FILES)} succeeded.')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')
