import numpy as np
from typing import List, Tuple, Optional


class WindowSegmenter:
    """
    Segments continuous multichannel or single-channel EEG signals into fixed-length windows.
    Adheres strictly to the leakage-safe evaluation requirement:
    - Overlapping windows never cross patient boundaries.
    - Labels are assigned based on a predefined overlap threshold with annotated seizure intervals.
    """

    def __init__(
        self,
        window_sec: float = 10.0,
        overlap_sec: float = 0.0,
        fs: float = 256.0,
        overlap_ratio_threshold: float = 0.5,
    ):
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.fs = fs
        self.overlap_ratio_threshold = overlap_ratio_threshold
        self.samples_per_window = int(round(window_sec * fs))
        self.step_samples = int(round((window_sec - overlap_sec) * fs))

        if self.step_samples <= 0:
            raise ValueError("Window step must be positive.")

    def segment_signal(
        self,
        signal: np.ndarray,
        seizure_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments a signal into windows.
        signal: shape (C, T) or (T,)
        Returns:
            windows: shape (N, C, samples_per_window) or (N, samples_per_window)
            labels: shape (N,) binary labels (1=ictal, 0=non-ictal)
            time_bounds: shape (N, 2) timestamps [start_sec, end_sec] for each window
        """
        is_1d = signal.ndim == 1
        n_samples = signal.shape[-1]
        intervals = seizure_intervals or []

        windows_list = []
        labels_list = []
        bounds_list = []

        for start_idx in range(0, n_samples - self.samples_per_window + 1, self.step_samples):
            end_idx = start_idx + self.samples_per_window
            t_start = start_idx / self.fs
            t_end = end_idx / self.fs

            # Label assignment based on overlap threshold
            is_ictal = 0
            for s_start, s_end in intervals:
                overlap_start = max(t_start, s_start)
                overlap_end = min(t_end, s_end)
                overlap_duration = max(0.0, overlap_end - overlap_start)
                if overlap_duration / self.window_sec >= self.overlap_ratio_threshold:
                    is_ictal = 1
                    break

            if is_1d:
                windows_list.append(signal[start_idx:end_idx])
            else:
                windows_list.append(signal[:, start_idx:end_idx])

            labels_list.append(is_ictal)
            bounds_list.append((t_start, t_end))

        if not windows_list:
            empty_shape = (0, self.samples_per_window) if is_1d else (0, signal.shape[0], self.samples_per_window)
            return np.empty(empty_shape, dtype=np.float32), np.empty(0, dtype=np.int64), np.empty((0, 2))

        return (
            np.array(windows_list, dtype=np.float32),
            np.array(labels_list, dtype=np.int64),
            np.array(bounds_list, dtype=np.float32),
        )

