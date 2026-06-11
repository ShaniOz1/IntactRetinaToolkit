"""
Batch runner: ex-vivo direct response analysis
================================================
Runs detect_direct_response on a fixed list of EDF files and saves
each result as a CSV in RESULTS_DIR.
"""

import os
import traceback

from dataobj import load_edf

RESULTS_DIR = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_all'

DIRECT_WIN_MS       = 10.0
BLANK_MS            = 2.5
DIRECT_THRESHOLD_MV = None  # None → compute from data

# (path, stim_electrode)
FILES = [
    # ── Group 2 · 2024.11.17 Direct Response - Fading · G6 ──────────────────
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T12-53-10No_noise_stimulation_4uA_1Hz_200pulses_B-00071.edf',          'G6'),
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T12-57-14No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',         'G6'),
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-02-36No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',         'G6'),
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-05-52Noise_1-10Hz_0.2mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-07-07Noise_1-10Hz_0.1mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-08-03No_noise_stimulation_4uA_10Hz_200pulses_B-00071.edf',         'G6'),
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\Direct Response - Fading\2024-11-17T13-09-21Noise_1-10Hz_0.05mA_stimulation_4uA_10Hz_200pulses_B-00071.edf', 'G6'),

    # ── Group 3 · 2024.11.17 SR · G6 · no-noise only ────────────────────────
    (r'C:\Shani\MEA mini1200\2024.11.17 e14_Shani\SR\2024-11-17T13-16-40No_noise_stimulation_2uA_1Hz_20pulses_B-00071.edf', 'G6'),

    # ── Group 4 · 2024.11.20 Ieva · G6 ─────────────────────────────────────
    (r'C:\Shani\MEA mini1200\2024.11.20 e14_Ieva\2024-11-20T14-45-08_40uA _g6_10Hz_100Pulses_B-00071.edf', 'G6'),

    # ── Group 6 · 2025.11.02 Retina2 phase1-normal · J6 ────────────────────
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-56-11J6_7uA_300us_60us_1Hz_100pulses.edf',  'J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T10-59-46J6_7uA_300us_60us_10Hz_100pulses.edf', 'J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-01-53J6_7uA_300us_60us_20Hz_100pulses.edf', 'J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-05-56J6_7uA_300us_60us_20Hz_100pulses.edf', 'J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-06-13J6_7uA_300us_60us_20Hz_100pulses.edf', 'J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-07-37J6_10uA_300us_60us_1Hz_100pulses.edf', 'J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-10-01J6_10uA_300us_60us_10Hz_100pulses.edf','J6'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina2\phase1-normal\2025-11-02T11-12-26J6_10uA_300us_60us_20Hz_100pulses.edf','J6'),

    # ── Group 10 · 2025.11.02 Retina3 phase1-normal · J9 ───────────────────
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-55-12J9_10uA_300us_60us_1Hz_100pulses.edf',      'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T15-57-46J9_10uA_300us_60us_10Hz_100pulses.edf',     'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-00-18J9_10uA_300us_60us_20Hz_100pulses.edf',     'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-02-50J9_10uA_300us_60us_50Hz_100pulses.edf',     'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-03-33J9_20uA_300us_60us_1Hz_100pulses.edf',      'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-06-09J9_20uA_300us_60us_10Hz_100pulses.edf',     'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-07-34J9_20uA_300us_60us_20Hz_100pulses.edf',     'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-08-27J9_20uA_300us_60us_50Hz_100pulses.edf',     'J9'),
    (r'C:\Shani\MEA mini1200\2025.11.02 e14_Shani\Retina3\phase1-normal\2025-11-02T16-13-18J9_20uA_300us_60us_100Hz_1000pulses.edf',   'J9'),

    # ── Group 12 · 2025.11.12 Retina1 Phase1-Normal · G10 ──────────────────
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-19-197uA_300us_60us_1Hz_100pulse_B-00071.edf',  'G10'),
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-22-027uA_300us_60us_10Hz_100pulse_B-00071.edf', 'G10'),
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-22-347uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-23-087uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-25-147uA_300us_60us_20Hz_100pulse_B-00071.edf', 'G10'),
    (r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1\Phase1 - Normal\2025-11-12T11-26-217uA_300us_60us_50Hz_100pulse_B-00071.edf', 'G10'),
]


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    failed = []

    for i, (edf_path, stim_electrode) in enumerate(FILES, 1):
        print(f'\n[{i}/{len(FILES)}] {os.path.basename(edf_path)}  (stim={stim_electrode})')
        try:
            rec = load_edf(edf_path, stim_electrode=stim_electrode)
            rec.filter()
            rec.blank(duration_ms=BLANK_MS, source='filtered_data')
            rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS, threshold=DIRECT_THRESHOLD_MV)
            out_path = os.path.join(RESULTS_DIR, f'{rec.file_name}_direct_response.csv')
            rec.direct_response.to_csv(out_path, index=False)
            print(f'    saved → {out_path}')
        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(edf_path)

    print(f'\n{"="*60}')
    print(f'Done. {len(FILES) - len(failed)}/{len(FILES)} succeeded.')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')
