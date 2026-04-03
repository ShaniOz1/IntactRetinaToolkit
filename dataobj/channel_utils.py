"""
IntactRetinaToolkit.dataobj.channel_utils
==========================================
Parses channel names into (row, col) grid locations for both MEA (EDF)
and Intan probe (RHS) recordings, and provides a MEA index↔grid-ref
converter.

MEA channel names:   'E_B-00071 F7'  →  grid ref 'F7'  →  (row, col)
Intan channel names: 'B-000', 'B-025', etc.
"""

from __future__ import annotations
from typing import Optional


# ───────────────────────────────────────────────
# MEA (EDF) — name → (row, col)
# ───────────────────────────────────────────────

def mea_name_to_location(channel_name: str) -> Optional[tuple[int, int]]:
    """
    Parse a MEA channel label into a (row, col) grid tuple (0-based).

    The MEA labels follow the pattern 'E_B-00071 XN' where X is a letter
    column (A–M, skipping I) and N is a 1-based integer row.

    Parameters
    ----------
    channel_name : str
        Full channel label from the EDF file, or bare grid ref (e.g. 'F7').

    Returns
    -------
    tuple[int, int] | None
        (row, col) zero-based, or None if the name cannot be parsed.

    Examples
    --------
    >>> mea_name_to_location('E_B-00071 F7')
    (6, 5)
    >>> mea_name_to_location('F7')
    (6, 5)
    """
    grid_ref = channel_name.strip()
    if ' ' in grid_ref:
        grid_ref = grid_ref.split(' ')[-1].strip()

    if not grid_ref or len(grid_ref) < 2:
        return None

    try:
        col_letter = grid_ref[0].upper()
        row_1based = int(grid_ref[1:])
    except (ValueError, IndexError):
        return None

    row = row_1based - 1
    col_ord = ord(col_letter) - ord('A')
    if col_ord > (ord('I') - ord('A')):   # skip 'I'
        col_ord -= 1

    return (row, col_ord)


def mea_channel_names_to_locations(
    channel_names: list[str],
) -> list[Optional[tuple[int, int]]]:
    """
    Map a list of MEA channel names to (row, col) tuples.

    Parameters
    ----------
    channel_names : list[str]

    Returns
    -------
    list[tuple[int, int] | None]
        Same length as input.
    """
    return [mea_name_to_location(n) for n in channel_names]


# ───────────────────────────────────────────────
# MEA — index ↔ grid-ref converter
# ───────────────────────────────────────────────

_MEA_CHANNEL_LIST: list[str] = [
    'E_B-00071 F7',  'E_B-00071 F8',  'E_B-00071 F12', 'E_B-00071 F11',
    'E_B-00071 F10', 'E_B-00071 F9',  'E_B-00071 E12', 'E_B-00071 E11',
    'E_B-00071 E10', 'E_B-00071 E9',  'E_B-00071 D12', 'E_B-00071 D11',
    'E_B-00071 D10', 'E_B-00071 D9',  'E_B-00071 C11', 'E_B-00071 C10',
    'E_B-00071 B10', 'E_B-00071 E8',  'E_B-00071 C9',  'E_B-00071 B9',
    'E_B-00071 A9',  'E_B-00071 D8',  'E_B-00071 C8',  'E_B-00071 B8',
    'E_B-00071 A8',  'E_B-00071 D7',  'E_B-00071 C7',  'E_B-00071 B7',
    'E_B-00071 A7',  'E_B-00071 E7',  'E_B-00071 F6',  'E_B-00071 E6',
    'E_B-00071 A6',  'E_B-00071 B6',  'E_B-00071 C6',  'E_B-00071 D6',
    'E_B-00071 A5',  'E_B-00071 B5',  'E_B-00071 C5',  'E_B-00071 D5',
    'E_B-00071 A4',  'E_B-00071 B4',  'E_B-00071 C4',  'E_B-00071 D4',
    'E_B-00071 B3',  'E_B-00071 C3',  'E_B-00071 C2',  'E_B-00071 E5',
    'E_B-00071 D3',  'E_B-00071 D2',  'E_B-00071 D1',  'E_B-00071 E4',
    'E_B-00071 E3',  'E_B-00071 E2',  'E_B-00071 E1',  'E_B-00071 F4',
    'E_B-00071 F3',  'E_B-00071 F2',  'E_B-00071 F1',  'E_B-00071 F5',
    'E_B-00071 G6',  'E_B-00071 G5',  'E_B-00071 G1',  'E_B-00071 G2',
    'E_B-00071 G3',  'E_B-00071 G4',  'E_B-00071 H1',  'E_B-00071 H2',
    'E_B-00071 H3',  'E_B-00071 H4',  'E_B-00071 J1',  'E_B-00071 J2',
    'E_B-00071 J3',  'E_B-00071 J4',  'E_B-00071 K2',  'E_B-00071 K3',
    'E_B-00071 L3',  'E_B-00071 H5',  'E_B-00071 K4',  'E_B-00071 L4',
    'E_B-00071 M4',  'E_B-00071 J5',  'E_B-00071 K5',  'E_B-00071 L5',
    'E_B-00071 M5',  'E_B-00071 J6',  'E_B-00071 K6',  'E_B-00071 L6',
    'E_B-00071 M6',  'E_B-00071 H6',  'E_B-00071 G7',  'E_B-00071 H7',
    'E_B-00071 M7',  'E_B-00071 L7',  'E_B-00071 K7',  'E_B-00071 J7',
    'E_B-00071 M8',  'E_B-00071 L8',  'E_B-00071 K8',  'E_B-00071 J8',
    'E_B-00071 M9',  'E_B-00071 L9',  'E_B-00071 K9',  'E_B-00071 J9',
    'E_B-00071 L10', 'E_B-00071 K10', 'E_B-00071 K11', 'E_B-00071 H8',
    'E_B-00071 J10', 'E_B-00071 J11', 'E_B-00071 J12', 'E_B-00071 H9',
    'E_B-00071 H10', 'E_B-00071 H11', 'E_B-00071 H12', 'E_B-00071 G9',
    'E_B-00071 G10', 'E_B-00071 G11', 'E_B-00071 G12', 'E_B-00071 G8',
]


def mea_convert_channel(channel_input: int | str) -> str | int:
    """
    Convert between MEA channel index (int) and grid reference (str).

    Parameters
    ----------
    channel_input : int or str
        - int or numeric str → returns grid reference string (e.g. 'B9')
        - grid reference str (e.g. 'B9') → returns int index

    Returns
    -------
    str or int

    Raises
    ------
    ValueError, TypeError
    """
    if isinstance(channel_input, int):
        if 0 <= channel_input < len(_MEA_CHANNEL_LIST):
            return _MEA_CHANNEL_LIST[channel_input].split(' ')[-1]
        raise ValueError(f"Index {channel_input} out of range "
                         f"(0–{len(_MEA_CHANNEL_LIST) - 1}).")

    if isinstance(channel_input, str):
        s = channel_input.strip()
        if s.isdigit():
            return mea_convert_channel(int(s))
        has_letter = any(c.isalpha() for c in s)
        has_digit  = any(c.isdigit() for c in s)
        if has_letter and has_digit:
            for idx, ch in enumerate(_MEA_CHANNEL_LIST):
                if ch.endswith(' ' + s):
                    return idx
            raise ValueError(f"Grid reference '{s}' not found.")
        raise ValueError(f"Cannot interpret channel input '{s}'.")

    raise TypeError(f"Expected str or int, got {type(channel_input).__name__}.")


# ───────────────────────────────────────────────
# Intan probe (RHS) — name → (row, col)
# ───────────────────────────────────────────────

# Real channel numbers for each RHS recording index (0–15).
# Indices 0–7  → real channels 0–7
# Indices 8–15 → real channels 24–31
_RHS_INDEX_TO_REAL_CH: list[int] = [0, 1, 2, 3, 4, 5, 6, 7,
                                     24, 25, 26, 27, 28, 29, 30, 31]

# Physical (row, col) location keyed by real channel number.
# col 0 = left shank, col 1 = right shank (matches the prob16 schematic).
_PROB16_LAYOUT: dict[int, tuple[int, int]] = {
    0:  (0, 0),  1:  (1, 0),  2:  (2, 0),  3:  (3, 0),
    4:  (4, 0),  5:  (5, 0),  6:  (6, 0),  7:  (7, 0),
    24: (0, 1),  25: (1, 1),  26: (2, 1),  27: (3, 1),
    28: (4, 1),  29: (5, 1),  30: (6, 1),  31: (7, 1),
}


def intan_get_locations(
    channel_names: list[str],
) -> list[Optional[tuple[int, int]]]:
    """
    Convert a list of real Intan channel number strings to (row, col) locations.

    channel_names are already real channel numbers ('0'–'7', '24'–'31')
    as set by the rhs_loader.

    Parameters
    ----------
    channel_names : list[str]
        e.g. ['0', '1', ..., '7', '24', ..., '31']

    Returns
    -------
    list[tuple[int, int] | None]
    """
    locations = []
    for name in channel_names:
        try:
            real_ch = int(name)
            locations.append(_PROB16_LAYOUT.get(real_ch))
        except ValueError:
            locations.append(None)
    return locations