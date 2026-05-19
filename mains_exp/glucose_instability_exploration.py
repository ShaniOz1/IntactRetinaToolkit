"""
glucose_instability_exploration.py
====================================
Loops over all .edf files under the Phase* subfolders of Retina1
(2025.11.12 experiment), tags each file as 'low' or 'normal' glucose
based on the subfolder name, and saves a CSV per file with that tag
prepended to the filename.
"""

import os
import glob
import traceback

from dataobj import load_edf

SOURCE_DIR   = r'C:\Shani\MEA mini1200\2025.11.12 e14_Shani\Retina1'
RESULTS_DIR  = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\ex_vivo_glucose'
STIM_ELECTRODE      = 'G10'

# ============================================================
#  PARAMS
# ============================================================
DIRECT_WIN_MS       = 10.0
BLANK_MS            = 1.0
DIRECT_THRESHOLD_MV = 0.2


# ── helpers ──────────────────────────────────────────────────────────────────

def glucose_tag(folder_name):
    """Return 'low' or 'normal' based on the phase subfolder name."""
    name = folder_name.lower()
    if 'low' in name:
        return 'low'
    return 'normal'


def find_edf_entries():
    """Return sorted list of (edf_path, glucose_tag) for all Phase* subfolders."""
    entries = []
    for phase_dir in sorted(glob.glob(os.path.join(SOURCE_DIR, 'Phase*'))):
        tag = glucose_tag(os.path.basename(phase_dir))
        for path in sorted(glob.glob(os.path.join(phase_dir, '*.edf'))):
            entries.append((path, tag))
    return entries


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    entries = find_edf_entries()
    if not entries:
        print(f'No .edf files found under {SOURCE_DIR}')
        exit()

    print(f'Found {len(entries)} .edf files.\n')

    failed  = []

    for i, (path, tag) in enumerate(entries, 1):
        fname    = os.path.basename(path)
        out_path = os.path.join(RESULTS_DIR, f'{tag}_{fname}_direct_response.csv')

        if os.path.exists(out_path):
            print(f'[{i}/{len(entries)}] already exists, skipping: {fname}')
            continue

        print(f'[{i}/{len(entries)}] [{tag}] {fname}')
        try:
            rec = load_edf(path, stim_electrode=STIM_ELECTRODE)
            rec.filter()
            rec.blank(duration_ms=BLANK_MS, source='filtered_data')
            rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                       threshold=DIRECT_THRESHOLD_MV)
            rec.direct_response.to_csv(out_path, index=False)
            print(f'    saved → {out_path}')
        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(path)

    print(f'\n{"=" * 60}')
    print(f'Done. {len(entries) - len(failed)}/{len(entries)} succeeded.')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')
