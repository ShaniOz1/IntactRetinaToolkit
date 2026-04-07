"""
IntactRetinaToolkit.dataobj.edf_loader
========================================
Loads a single MEA .edf file (+ optional companion .txt) into a
RetinalRecording.
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import pyedflib
from tqdm import tqdm

from dataobj.recording import RetinalRecording
from dataobj.channel_utils import mea_channel_names_to_locations


def load_edf(
    file_path: str,
    stim_electrode: str | None = None,
    create_output_folder: bool = False,
) -> RetinalRecording:
    """
    Load a MEA .edf file (and its companion .txt) into a RetinalRecording.

    The companion .txt must share the same base name as the .edf and live
    in the same directory. If absent, stim_indices will be None.

    Parameters
    ----------
    file_path : str
        Path to the .edf file.
    stim_electrode : str | None
        Grid label of the stimulation electrode (e.g. 'G6'). If given,
        that channel's signal is zeroed out after loading. Default: None.
    create_output_folder : bool
        If True, creates a folder named after the file in the current
        working directory. Default: False.

    Returns
    -------
    RetinalRecording

    Raises
    ------
    FileNotFoundError
    ValueError
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.lower().endswith('.edf'):
        raise ValueError(f"Expected an .edf file, got: {file_path}")

    file_name = os.path.basename(file_path)

    f              = pyedflib.EdfReader(file_path)
    sample_rate    = int(f.getSampleFrequency(0))
    channel_names  = list(f.getSignalLabels())
    file_info      = f.file_info_long()
    num_signals    = f.signals_in_file
    recording_data = _edf_to_numpy(f, channel_names)

    channel_locations = mea_channel_names_to_locations(channel_names)
    stim_indices      = _load_stim_from_txt(file_path, sample_rate)

    if stim_electrode is not None:
        _zero_stim_electrode(recording_data, channel_names, stim_electrode)

    return RetinalRecording(
        source='edf',
        file_path=file_path,
        file_name=file_name,
        sample_rate=sample_rate,
        recording_data=recording_data,
        channel_names=channel_names,
        channel_locations=channel_locations,
        stim_indices=stim_indices,
        stim_data=None,
        stim_current=None,
        stim_channel_name=stim_electrode,
        metadata={
            'file_info':           file_info,
            'num_signals_in_file': num_signals,
            'n_channels':          recording_data.shape[0],
            'n_samples':           recording_data.shape[1],
        },
    )


# ── internal ────────────────────────────────────────────────────────────────

def _edf_to_numpy(
    f: pyedflib.EdfReader,
    channel_names: list[str],
) -> np.ndarray:
    """Read all signals into a (n_channels, n_samples) array."""
    signals = []
    for i in tqdm(range(len(channel_names)), desc="Reading EDF signals"):
        signals.append(f.readSignal(i))
    return np.array(signals)


def _load_stim_from_txt(
    edf_file_path: str,
    sample_rate: int,
) -> np.ndarray | None:
    """
    Parse the companion .txt file to extract stimulation onset indices.

    Lines 0–2: ignored header
    Line 3:    tab-separated column names
    Line 4+:   tab-separated floats (values are stimulus times in µs,
               with a trailing unit suffix that is stripped before parsing)

    Returns the first column converted to sample indices, or None on failure.
    """
    directory = os.path.dirname(edf_file_path)
    base_name = os.path.splitext(os.path.basename(edf_file_path))[0]
    txt_path  = os.path.join(directory, base_name + '.txt')

    if not os.path.isfile(txt_path):
        warnings.warn(
            f"Companion .txt not found: '{txt_path}'. stim_indices=None.",
            UserWarning, stacklevel=3,
        )
        return None

    try:
        with open(txt_path, 'r') as fh:
            lines = fh.readlines()

        if len(lines) < 5:
            warnings.warn(
                f"'{txt_path}' has fewer than 5 lines — cannot parse. "
                "stim_indices=None.",
                UserWarning, stacklevel=3,
            )
            return None

        headers     = lines[3].strip().split('\t')
        data_matrix = []
        for line in lines[4:]:
            values = line.strip().split('\t')
            if not values or values == ['']:
                continue
            try:
                row = [float(s[:-2]) for s in values if len(s) > 2]
            except ValueError:
                continue
            if row:
                data_matrix.append(row)

        if not data_matrix:
            warnings.warn(
                f"No numeric data in '{txt_path}'. stim_indices=None.",
                UserWarning, stacklevel=3,
            )
            return None

        df = pd.DataFrame(data_matrix, columns=headers[:len(data_matrix[0])])

        # First column = onset times in µs → sample indices
        scaling_factor = (1 / sample_rate) * 1e6
        stim_indices   = (df.iloc[:, 0].to_numpy() / scaling_factor).astype(int)
        return stim_indices

    except Exception as exc:
        warnings.warn(
            f"Failed to parse '{txt_path}': {exc}. stim_indices=None.",
            UserWarning, stacklevel=3,
        )
        return None


def _zero_stim_electrode(
    recording_data: np.ndarray,
    channel_names: list[str],
    stim_electrode: str,
) -> None:
    """Zero out the stimulation electrode channel in-place."""
    matches = [i for i, n in enumerate(channel_names) if stim_electrode in n]
    if not matches:
        warnings.warn(
            f"stim_electrode '{stim_electrode}' not found — no channel zeroed.",
            UserWarning, stacklevel=3,
        )
        return
    recording_data[matches[0], :] = 0.0