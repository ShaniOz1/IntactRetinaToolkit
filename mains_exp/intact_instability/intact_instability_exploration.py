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
RESULTS_BASE = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_all4_sorted'

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


def retina_from_path(path):
    """Return the Retina* folder name from anywhere in the path, or 'unknown'."""
    for part in path.replace('\\', '/').split('/'):
        if re.match(r'Retina\d+', part, re.IGNORECASE):
            return part
    return 'unknown'


def find_rhs_files():
    """Return sorted list of (path, retina_name) under Retina* folders of the selected dates."""
    entries = []
    for date in EXPERIMENT_DATES:
        date_dir = os.path.join(BASE_DIR, date)
        for retina_dir in sorted(glob.glob(os.path.join(date_dir, 'Retina*'))):
            for path in sorted(glob.glob(os.path.join(retina_dir, '**', '*.rhs'), recursive=True)):
                entries.append((path, retina_from_path(path)))
    return entries


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rhs_entries = find_rhs_files()

    if not rhs_entries:
        print(f'No .rhs files found under {EXPERIMENT_DATES}')
        exit()

    print(f'Found {len(rhs_entries)} .rhs files.\n')

    failed  = []
    skipped = []

    for i, (path, retina) in enumerate(rhs_entries, 1):
        fname = os.path.basename(path)
        freq  = parse_freq_hz(fname)

        if freq is None:
            print(f'[{i}/{len(rhs_entries)}] SKIP (no freq in name): {fname}')
            skipped.append(path)
            continue

        os.makedirs(RESULTS_BASE, exist_ok=True)
        out_path = os.path.join(RESULTS_BASE, f'{retina}_{fname}_direct_response.csv')
        if os.path.exists(out_path):
            print(f'[{i}/{len(rhs_entries)}] already exists, skipping: {fname}')
            continue

        print(f'[{i}/{len(rhs_entries)}] [{retina}] {fname}  (freq={freq} Hz)')
        try:
            rec = load_rhs(path, stim_threshold=STIM_THRESHOLD)
            rec.detect_direct_response(win_size_ms=DIRECT_WIN_MS,
                                       threshold=DIRECT_THRESHOLD_MV,
                                       data_type='raw',
                                       plot=True,
                                       output_folder=None)

            df = rec.direct_response
            if df is None or df.empty:
                print('    No responses detected — skipping.')
                skipped.append(path)
                continue

            available = sorted(df['channel'].apply(lambda c: c.split()[-1].upper()).unique())
            for idx, ch in enumerate(available):
                print(f'      [{idx}] {ch}')
            raw_input_str = input('    Indices to include (space-separated), or Enter to skip: ').strip()

            if not raw_input_str:
                print('    No channels specified — skipping.')
                skipped.append(path)
                continue

            try:
                indices  = [int(x) for x in raw_input_str.split()]
                selected = {available[i] for i in indices if 0 <= i < len(available)}
            except ValueError:
                print('    Invalid input — skipping.')
                skipped.append(path)
                continue

            if not selected:
                print('    No valid indices — skipping.')
                skipped.append(path)
                continue

            df_filtered = df[df['channel'].apply(lambda c: c.split()[-1].upper() in selected)]

            if df_filtered.empty:
                print('    No matching channels found — skipping.')
                skipped.append(path)
                continue

            df_filtered.to_csv(out_path, index=False)
            print(f'    saved → {out_path}  (channels: {", ".join(sorted(selected))})')

        except Exception:
            print(f'    ERROR — skipping')
            traceback.print_exc()
            failed.append(path)

    print(f'\n{"=" * 60}')
    print(f'Done. {len(rhs_entries) - len(failed) - len(skipped)} processed, '
          f'{len(skipped)} skipped (no freq), {len(failed)} failed.')
    if failed:
        print('Failed:')
        for f in failed:
            print(f'  {f}')
