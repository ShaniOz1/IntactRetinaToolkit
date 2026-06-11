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

BASE_DIR     = r'C:\Shani\SoftC prob\16Ch prob experiments\2025.11.05 E14\Retina2'
RESULTS_BASE = r'C:\Users\YHLab\PycharmProjects\IntactRetinaToolkit\Results\Intact_glucose'


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


_PHASE_RE = re.compile(r'^phase\d+$', re.IGNORECASE)


def find_rhs_files():
    """Return sorted (path, retina_name) for .rhs files anywhere inside phase1–phase5 folders."""
    entries = []
    for root, dirs, files in os.walk(BASE_DIR):
        # Accept if the top-level folder relative to BASE_DIR is a phase folder
        top_folder = os.path.relpath(root, BASE_DIR).split(os.sep)[0]
        if _PHASE_RE.match(top_folder):
            retina = retina_from_path(root)
            for fname in sorted(files):
                if fname.lower().endswith('.rhs'):
                    entries.append((os.path.join(root, fname), retina))
    return sorted(entries)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rhs_entries = find_rhs_files()

    if not rhs_entries:
        print(f'No .rhs files found under {BASE_DIR}')
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
                                       output_folder=RESULTS_BASE)

            df = rec.direct_response
            if df is None or df.empty:
                print('    No responses detected — skipping.')
                skipped.append(path)
                continue

            available = sorted(df['channel'].apply(lambda c: c.split()[-1].upper()).unique())
            print(f'    Available channels: {" ".join(available)}')
            raw_input_str = input('    Channel names to save (space-separated), or Enter to skip: ').strip()

            if not raw_input_str:
                print('    No channels specified — skipping.')
                skipped.append(path)
                continue

            typed    = {n.upper() for n in raw_input_str.split()}
            selected = typed & set(available)
            unknown  = typed - set(available)
            if unknown:
                print(f'    Unknown channel names ignored: {", ".join(sorted(unknown))}')

            if not selected:
                print('    No valid channel names — skipping.')
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
