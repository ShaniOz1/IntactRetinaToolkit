"""
IntactRetinaToolkit.dataobj.recording
=======================================
Defines RetinalRecording — the single unified data object produced by
both the RHS (Intan) and EDF (MEA) loaders.

Do not instantiate this directly. Use the loaders:
    from dataobj import load_rhs, load_edf
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetinalRecording:
    """
    Unified data container for a single retinal recording session.

    Loading attributes (set by loaders)
    ------------------------------------
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
        Raw stimulation signal (1-D). RHS only; None for EDF.
    stim_current : float | None
        Peak stimulation current in µA. RHS only; None for EDF.
    stim_channel_name : str | None
        Name/label of the channel used for stimulation.
    metadata : dict
        Miscellaneous recording metadata (start_time, file_info, etc.).

    Preprocessing attributes (set by rec.filter() / rec.blank())
    -------------------------------------------------------------
    filtered_data : np.ndarray | None
        Bandpass-filtered copy of recording_data. None until rec.filter()
        is called.
    blanked_data : np.ndarray | None
        Copy of the data with zeros around each stim pulse. None until
        rec.blank() is called. May be applied to recording_data or
        filtered_data — user decides the order.

    Analysis result attributes (set by rec.detect_*())
    ---------------------------------------------------
    direct_response : pd.DataFrame | None
        One row per detected direct spike. Columns:
        channel, pulse_index, amplitude (µV), latency (ms), width (ms).
        None until rec.detect_direct_response() is called.

    indirect_response : pd.DataFrame | None
        One row per detected indirect spike. Columns:
        channel, spike_index, amplitude (µV), latency (ms), width (ms).
        None until rec.detect_indirect_response() is called.

    spontaneous : pd.DataFrame | None
        One row per detected spontaneous wave. Columns:
        channel, start_index, stop_index.
        None until rec.detect_spontaneous() is called.
    """

    # --- Identity ---
    source: str
    file_path: str
    file_name: str

    # --- Signal ---
    sample_rate: int
    recording_data: np.ndarray              # (n_channels, n_samples)

    # --- Channels ---
    channel_names: list                     # list[str]
    channel_locations: list                 # list[tuple[int,int] | None]

    # --- Stimulation ---
    stim_indices: Optional[np.ndarray]
    stim_data: Optional[np.ndarray]
    stim_current: Optional[float]
    stim_channel_name: Optional[str]

    # --- Metadata ---
    metadata: dict = field(default_factory=dict)

    # --- Preprocessing results ---
    filtered_data: Optional[np.ndarray] = field(default=None, init=False)
    blanked_data: Optional[np.ndarray]  = field(default=None, init=False)

    # --- Analysis results ---
    direct_response: Optional[pd.DataFrame]   = field(default=None, init=False)
    indirect_response: Optional[pd.DataFrame] = field(default=None, init=False)
    spontaneous: Optional[pd.DataFrame]       = field(default=None, init=False)

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

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Preprocessing methods
    # ------------------------------------------------------------------

    def filter(
        self,
        lowcut: int = 300,
        highcut: int = 3000,
        order: int = 2,
    ) -> None:
        """
        Apply a Butterworth bandpass filter and store result in filtered_data.

        Parameters
        ----------
        lowcut : int
            Lower cutoff frequency in Hz. Default: 300.
        highcut : int
            Upper cutoff frequency in Hz. Default: 3000.
        order : int
            Filter order. Default: 2.
        """
        from scipy.signal import butter, sosfiltfilt
        sos = butter(order, [lowcut, highcut], btype='bandpass',
                     fs=self.sample_rate, output='sos')
        self.filtered_data = sosfiltfilt(sos, self.recording_data)
        print(f"[filter] Bandpass {lowcut}–{highcut} Hz applied → rec.filtered_data")

    def blank(
        self,
        duration_ms: float = 2.0,
        pre_ms: float = 0.0,
        source: str = 'recording_data',
    ) -> None:
        """
        Zero out a window around each stimulation pulse and store result
        in blanked_data.

        Parameters
        ----------
        duration_ms : float
            Duration to blank after each stim onset, in ms. Default: 2.0.
        pre_ms : float
            Duration to blank before each stim onset, in ms. Default: 0.0.
        source : str
            Which data array to blank: 'recording_data' (default) or
            'filtered_data'. Use 'filtered_data' if you want to blank
            the already-filtered signal.

        Raises
        ------
        ValueError
            If source='filtered_data' but filter() has not been called yet.
        RuntimeError
            If stim_indices is None.
        """
        if self.stim_indices is None:
            raise RuntimeError(
                "Cannot blank: stim_indices is None. "
                "Load a file with stimulation data or set stim_indices manually."
            )
        if source == 'filtered_data':
            if self.filtered_data is None:
                raise ValueError(
                    "source='filtered_data' but filtered_data is None. "
                    "Call rec.filter() first."
                )
            data = self.filtered_data
        else:
            data = self.recording_data

        pre_samp  = int(pre_ms  * 0.001 * self.sample_rate)
        post_samp = int(duration_ms * 0.001 * self.sample_rate)

        blanked = np.copy(data)
        for idx in self.stim_indices:
            start = int(max(0, idx - pre_samp))
            end   = int(min(blanked.shape[1], idx + post_samp))
            blanked[:, start:end] = 0.0

        self.blanked_data = blanked
        print(f"[blank] {len(self.stim_indices)} pulses blanked "
              f"({pre_ms} ms before, {duration_ms} ms after) → rec.blanked_data")

    # ------------------------------------------------------------------
    # Analysis methods (implementations in dataobj/analysis/)
    # ------------------------------------------------------------------

    def detect_direct_response(
        self,
        win_size_ms: float = 15.0,
        blank_ms: float = 3.5,
    ) -> None:
        """
        Detect direct (stimulus-evoked) responses and populate
        rec.direct_response.

        For RHS: ICA-based artifact removal followed by spike detection.
        For EDF: threshold-based detection on the blanked+filtered signal.

        Results are stored in rec.direct_response as a DataFrame with
        columns: channel, pulse_index, amplitude (µV), latency (ms),
        width (ms).

        Parameters
        ----------
        win_size_ms : float
            Analysis window size per pulse in ms. Default: 15.
        blank_ms : float
            Blanking duration after stim onset in ms (applied internally
            before detection). Default: 3.5.
        """
        from dataobj.analysis.direct import run_direct_response
        self.direct_response = run_direct_response(self, win_size_ms, blank_ms)
        print(f"[direct] {len(self.direct_response)} spikes detected "
              f"across {self.direct_response['channel'].nunique()} channels "
              f"→ rec.direct_response")

    def detect_indirect_response(
        self,
        blanking_ms: float = 15.0,
        threshold_std: float = 4.0,
    ) -> None:
        """
        Detect indirect (network-driven) responses and populate
        rec.indirect_response.

        Uses a noise-based threshold (N × std) on the blanked signal.

        Results are stored in rec.indirect_response as a DataFrame with
        columns: channel, spike_index, amplitude (µV), latency (ms),
        width (ms).

        Parameters
        ----------
        blanking_ms : float
            Window around each stim pulse to exclude from threshold
            estimation and detection. Default: 15.
        threshold_std : float
            Threshold multiplier on per-channel std. Default: 4.
        """
        from dataobj.analysis.indirect import run_indirect_response
        self.indirect_response = run_indirect_response(
            self, blanking_ms, threshold_std
        )
        print(f"[indirect] {len(self.indirect_response)} spikes detected "
              f"→ rec.indirect_response")

    def detect_spontaneous(
        self,
        min_duration_ms: float = 5.0,
        threshold_std: float = 4.0,
    ) -> None:
        """
        Detect spontaneous activity waves and populate rec.spontaneous.

        Results are stored in rec.spontaneous as a DataFrame with
        columns: channel, start_index, stop_index.

        Parameters
        ----------
        min_duration_ms : float
            Minimum wave duration in ms to be counted. Default: 5.
        threshold_std : float
            Threshold multiplier on per-channel std. Default: 4.
        """
        from dataobj.analysis.spontaneous import run_spontaneous
        self.spontaneous = run_spontaneous(self, min_duration_ms, threshold_std)
        print(f"[spontaneous] {len(self.spontaneous)} waves detected "
              f"→ rec.spontaneous")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        stim_info = (
            f"stim_indices={len(self.stim_indices)}"
            if self.stim_indices is not None else "stim_indices=None"
        )
        results = []
        if self.filtered_data   is not None: results.append("filtered")
        if self.blanked_data    is not None: results.append("blanked")
        if self.direct_response is not None: results.append("direct_response")
        if self.indirect_response is not None: results.append("indirect_response")
        if self.spontaneous     is not None: results.append("spontaneous")
        result_str = f", computed=[{', '.join(results)}]" if results else ""
        return (
            f"RetinalRecording("
            f"source='{self.source}', "
            f"file='{self.file_name}', "
            f"n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, "
            f"sample_rate={self.sample_rate} Hz, "
            f"duration={self.duration_sec:.1f} s, "
            f"{stim_info}"
            f"{result_str})"
        )