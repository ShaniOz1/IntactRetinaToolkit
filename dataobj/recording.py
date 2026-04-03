"""
IntactRetinaToolkit.dataobj.recording
=======================================
Defines RetinalRecording — the single unified data object produced by
both the RHS (Intan) and EDF (MEA) loaders.

Do not instantiate this directly. Use the loaders:
    from IntactRetinaToolkit.dataobj import load_rhs, load_edf
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetinalRecording:
    """
    Unified data container for a single retinal recording session.

    Attributes
    ----------
    source : str
        Origin of the data: 'rhs' or 'edf'.
    file_path : str
        Absolute path to the loaded file.
    file_name : str
        Basename of the loaded file (no directory).
    sample_rate : int
        Sampling frequency in Hz.
    recording_data : np.ndarray
        Raw voltage traces, shape (n_channels, n_samples), in µV.
    channel_names : list[str]
        String label for each channel, length == n_channels.
    channel_locations : list[tuple[int,int] | None]
        (row, col) grid position for each channel (0-based), or None if
        the channel name could not be parsed to a grid position.
    stim_indices : np.ndarray | None
        Sample indices where stimulation pulses begin. None if not found.
    stim_data : np.ndarray | None
        Raw stimulation signal (1-D), same length as recording axis-1.
        Available for RHS only; None for EDF.
    stim_current : float | None
        Peak stimulation current in µA. RHS only; None for EDF.
    stim_pulse_duration : int | None
        Stimulation pulse phase duration in µs. RHS only; None for EDF.
    stim_channel_name : str | None
        Name/label of the channel used for stimulation.
    stim_electrode : str | None
        Grid label of the stimulation electrode (e.g. 'G6').
    metadata : dict
        Miscellaneous recording metadata (start_time, file_info, etc.).
    output_folder : str | None
        Path to the output folder. None unless create_output_folder=True
        was passed to the loader.
    """

    # Identity
    source: str
    file_path: str
    file_name: str

    # Signal
    sample_rate: int
    recording_data: np.ndarray              # (n_channels, n_samples)

    # Channels
    channel_names: list                     # list[str]
    channel_locations: list                 # list[tuple[int,int] | None]

    # Stimulation
    stim_indices: Optional[np.ndarray]
    stim_data: Optional[np.ndarray]
    stim_current: Optional[float]
    stim_channel_name: Optional[str]

    # Extras
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        """Number of recording channels."""
        return self.recording_data.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return self.recording_data.shape[1]

    @property
    def duration_sec(self) -> float:
        """Total recording duration in seconds."""
        return self.n_samples / self.sample_rate

    def get_channel_index(self, name: str) -> int:
        """
        Return the index of a channel by (partial) name match.

        Parameters
        ----------
        name : str
            Full or partial channel name to search for.

        Returns
        -------
        int
            Index of the first matching channel.

        Raises
        ------
        ValueError
            If no channel matches the given name.
        """
        matches = [i for i, ch in enumerate(self.channel_names) if name in ch]
        if not matches:
            raise ValueError(
                f"No channel found matching '{name}'.\n"
                f"Available channels: {self.channel_names}"
            )
        return matches[0]

    def get_channel_data(self, name: str) -> np.ndarray:
        """
        Return the raw signal for a channel identified by (partial) name.

        Parameters
        ----------
        name : str
            Full or partial channel name.

        Returns
        -------
        np.ndarray
            1-D array of shape (n_samples,).
        """
        return self.recording_data[self.get_channel_index(name)]

    def __repr__(self) -> str:
        stim_info = (
            f"stim_indices={len(self.stim_indices)}"
            if self.stim_indices is not None else "stim_indices=None"
        )
        return (
            f"RetinalRecording("
            f"source='{self.source}', "
            f"file='{self.file_name}', "
            f"n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, "
            f"sample_rate={self.sample_rate} Hz, "
            f"duration={self.duration_sec:.1f} s, "
            f"{stim_info})"
        )