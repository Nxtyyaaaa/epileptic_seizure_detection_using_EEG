import numpy as np
from scipy import signal
from typing import Optional


class EEGFilter:
    """
    Standard zero-phase Butterworth bandpass and IIR notch filters for EEG signals.
    """

    def __init__(
        self,
        fs: float,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        order: int = 4,
        notch_freq: Optional[float] = 50.0,
        notch_q: float = 30.0,
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freq = notch_freq
        self.notch_q = notch_q

        # Ensure highcut is strictly below Nyquist
        nyq = 0.5 * fs
        if self.highcut >= nyq:
            self.highcut = nyq - 1.0

        # Design Butterworth bandpass filter
        self.b_band, self.a_band = signal.butter(
            self.order,
            [self.lowcut / nyq, self.highcut / nyq],
            btype="bandpass",
        )

        # Design Notch filter if notch_freq is below Nyquist
        self.b_notch, self.a_notch = None, None
        if self.notch_freq is not None and self.notch_freq < nyq:
            self.b_notch, self.a_notch = signal.iirnotch(
                w0=self.notch_freq / nyq,
                Q=self.notch_q,
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase filtering across the last axis of x.
        x can be 1D (samples) or 2D (channels, samples).
        """
        # If signal is shorter than filter padlen, return raw
        min_len = 3 * max(len(self.a_band), len(self.b_band))
        if x.shape[-1] <= min_len:
            return x

        filtered = signal.filtfilt(self.b_band, self.a_band, x, axis=-1)

        if self.b_notch is not None and self.a_notch is not None:
            filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=-1)

        return filtered.astype(np.float32)

