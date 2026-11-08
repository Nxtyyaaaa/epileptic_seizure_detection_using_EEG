import numpy as np
from scipy import signal
from typing import List, Dict, Optional


class DatasetHarmonizer:
    """
    Harmonizes EEG signals across different recording cohorts (e.g. CHB-MIT vs Siena):
    - Anti-aliasing filtering prior to downsampling
    - Resampling to target common frequency
    - Channel name normalization and channel selection
    """

    def __init__(self, target_fs: float = 256.0):
        self.target_fs = target_fs

    def resample_signal(self, x: np.ndarray, orig_fs: float) -> np.ndarray:
        """
        Resamples signal x from orig_fs to self.target_fs with zero-phase anti-aliasing filter.
        x: (..., samples)
        """
        if abs(orig_fs - self.target_fs) < 1e-2:
            return x.astype(np.float32)

        # Anti-aliasing filter if downsampling
        if orig_fs > self.target_fs:
            nyq_target = 0.5 * self.target_fs
            # Design low-pass anti-alias filter at 0.9 * new Nyquist
            cutoff = 0.9 * nyq_target / (0.5 * orig_fs)
            b, a = signal.butter(4, cutoff, btype="low")
            x = signal.filtfilt(b, a, x, axis=-1)

        target_samples = int(round(x.shape[-1] * (self.target_fs / orig_fs)))
        resampled = signal.resample(x, target_samples, axis=-1)
        return resampled.astype(np.float32)

    @staticmethod
    def standardize_channel_name(ch_name: str) -> str:
        """Normalizes electrode label strings (e.g., 'EEG FP1-F7' -> 'FP1-F7')."""
        name = ch_name.upper().replace("EEG", "").replace(" ", "").strip()
        name = name.replace("POL", "").strip("-")
        return name

