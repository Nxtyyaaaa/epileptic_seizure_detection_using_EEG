import numpy as np
from typing import List, Dict, Tuple, Optional
from src.features.time_domain import extract_time_domain_features
from src.features.frequency_domain import extract_frequency_domain_features
from src.features.wavelet_features import extract_wavelet_features
from src.features.nonlinear import extract_nonlinear_features


class EEGFeatureExtractor:
    """
    Unified feature extraction pipeline covering all four families from the report:
    1. Time domain (statistical moments, RMS, zero crossings)
    2. Frequency domain (Welch PSD, band powers, relative powers, spectral entropy)
    3. Wavelet domain (DWT subband energies)
    4. Nonlinear & complexity (Hjorth activity/mobility/complexity, entropies)
    """

    def __init__(
        self,
        fs: float,
        include_time: bool = True,
        include_freq: bool = True,
        include_wavelet: bool = True,
        include_nonlinear: bool = True,
        multichannel_strategy: str = "mean",  # "mean" (average across channels) or "flatten"
    ):
        self.fs = fs
        self.include_time = include_time
        self.include_freq = include_freq
        self.include_wavelet = include_wavelet
        self.include_nonlinear = include_nonlinear
        self.multichannel_strategy = multichannel_strategy
        self.feature_names_: List[str] = []

    def extract_single_channel(self, x: np.ndarray) -> Dict[str, float]:
        """Extracts all configured feature families for a 1D EEG window."""
        feats: Dict[str, float] = {}

        if self.include_time:
            feats.update(extract_time_domain_features(x))

        if self.include_freq:
            feats.update(extract_frequency_domain_features(x, self.fs))

        if self.include_wavelet:
            feats.update(extract_wavelet_features(x, level=4))

        if self.include_nonlinear:
            feats.update(extract_nonlinear_features(x))

        return feats

    def extract_window(self, win: np.ndarray) -> Dict[str, float]:
        """
        Extracts features from a window of shape (T,) or (C, T).
        """
        if win.ndim == 1:
            return self.extract_single_channel(win)

        num_channels = win.shape[0]
        per_channel_feats = [self.extract_single_channel(win[ch]) for ch in range(num_channels)]

        if not per_channel_feats:
            return {}

        keys = list(per_channel_feats[0].keys())

        if self.multichannel_strategy == "mean":
            aggregated = {}
            for k in keys:
                vals = [ch_f[k] for ch_f in per_channel_feats]
                aggregated[k] = float(np.mean(vals))
            return aggregated
        else:
            # Flatten strategy: ch0_feat, ch1_feat, ...
            flattened = {}
            for ch_idx, ch_f in enumerate(per_channel_feats):
                for k, v in ch_f.items():
                    flattened[f"ch{ch_idx}_{k}"] = v
            return flattened

    def transform(self, windows: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Transforms an array of windows into a 2D feature matrix (N, D).
        windows: (N, T) or (N, C, T)
        Returns:
            X_feats: np.ndarray of shape (N, D)
            feature_names: List of column names
        """
        if len(windows) == 0:
            return np.empty((0, 0), dtype=np.float32), []

        feature_dicts = []
        for i in range(len(windows)):
            fd = self.extract_window(windows[i])
            feature_dicts.append(fd)

        feature_names = list(feature_dicts[0].keys())
        self.feature_names_ = feature_names

        matrix = np.zeros((len(windows), len(feature_names)), dtype=np.float32)
        for i, fd in enumerate(feature_dicts):
            for j, k in enumerate(feature_names):
                val = fd.get(k, 0.0)
                # Guard against NaN / Inf
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                matrix[i, j] = val

        return matrix, feature_names

