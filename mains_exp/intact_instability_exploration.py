"""
intact_instability_exploration.py
===================================
Loops over all .rhs files found under the Retina* subfolders of the
configured experiment dates, detects direct responses, and saves a CSV
per file into Results/<freq>hz/ so the analyse script can pick them up.
"""

import os
import re
import glob
import traceback

from dataobj import load_rhs

BASE_DIR     = r'C:\Shani\SoftC prob\16Ch prob experiments'
RESULTS_BASE = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_all2'

# Experiment date folders to include (must start with one of these prefixes)
EXPERIMENT_DATES = [
    '2025.05.25 E14',
    '2025.05.28 E14',
]

# ============================================================
#  PARAMS
# ============================================================
STIM_THRESHOLD      = 470
DIRECT_WIN_MS       = 10.0
DIRECT_THRESHOLD_MV = 15


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_freq_hz(filename):
    """Return frequency in Hz parsed from filename, or None."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', filename, re.IGNORECASE)
    return float(m.group(1)) if m else None


def find_rhs_files():
    """Return sorted list of all .rhs paths under Retina* folders of the selected dates."""
    paths = []
    for date in EXPERIMENT_DATES:
        date_dir = os.path.join(BASE_DIR, date)
        for retina_dir in sorted(glob.glob(os.path.join(date_dir, 'Retina*'))):
            paths.extend(sorted(glob.glob(os.path.join(retina_dir, '**', '*.rhs'), recursive=True)))
    return paths


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rhs_files = find_rhs_files()

    if not rhs_files:
        print(f'No .rhs files found under {EXPERIMENT_DATES}')
        exit()

    print(f'Found {len(rhs_files)} .rhs files.\n')

    failed  = []
    skipped = []

    for i, path in enumerate(rhs_files, 1):
        fname = os.path.basename(path)
        freq  = parse_freq_hz(fname)

        if freq is None:
            print(f'[{i}/{len(rhs_files)}] SKIP (no freq in name): {fname}')
            skipped.append(path)
            continue

        os.makedirs(RESULTS_BASE, exist_ok=True)
        out_path = os.path.join(RESULTS_BASE, f'{fname}_direct_response.csv')
        if os.path.exists(out_path):
            print(f'[{i}/{len(rhs_files)}] already exists, skipping: {fname}')
            continue

        print(f'[{i}/{len(rhs_files)}] {fname}  (freq={freq} Hz)')
        try:
            rec = load_rhs(path, stim_threshold=STIM_THRESHOLD)
            rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                       threshold=DIRECT_THRESHOLD_MV,
                                       data_type='raw',
                                       plot=True,
                                       output_folder=RESULTS_BASE)
            rec.direct_response.to_csv(out_path, index=False)
            print(f'    saved → {out_path}')
        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(path)

    print(f'\n{"=" * 60}')
    print(f'Done. {len(rhs_files) - len(failed) - len(skipped)} processed, '
          f'{len(skipped)} skipped (no freq), {len(failed)} failed.')
    if failed:
        print('Failed:')
        for f in failed:
            print(f'  {f}')
