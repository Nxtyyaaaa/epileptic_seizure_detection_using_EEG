import numpy as np
from typing import Tuple, List
from src.balancing.base import BaseResampler


class SignalLevelAugmenter:
    """
    E6: Clinically justified signal-level waveform augmentation.
    Transforms raw minority EEG waveforms while preserving epileptic spike-wave morphology:
    - Jitter: mild Gaussian noise injection (scale ~2%)
    - Amplitude scaling: random scaling within [0.90, 1.10]
    - Temporal roll: minor cyclic shift within 5% of window
    """

    def __init__(
        self,
        noise_std_ratio: float = 0.02,
        scale_range: Tuple[float, float] = (0.90, 1.10),
        max_shift_ratio: float = 0.05,
        random_state: int = 42,
    ):
        self.noise_std_ratio = noise_std_ratio
        self.scale_range = scale_range
        self.max_shift_ratio = max_shift_ratio
        self.random_state = random_state

    def augment_waveform(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Applies random augmentation to a single raw EEG window (1D or 2D)."""
        aug = x.copy()
        n_samples = aug.shape[-1]

        # 1. Amplitude scaling
        scale = rng.uniform(self.scale_range[0], self.scale_range[1])
        aug = aug * scale

        # 2. Add mild Gaussian noise
        std = np.std(aug)
        noise = rng.normal(0, self.noise_std_ratio * std, size=aug.shape)
        aug = aug + noise

        # 3. Small time shift
        max_shift = int(n_samples * self.max_shift_ratio)
        if max_shift > 0:
            shift = rng.integers(-max_shift, max_shift + 1)
            aug = np.roll(aug, shift, axis=-1)

        return aug.astype(np.float32)

    def balance_raw_windows(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        target_ratio: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes augmented raw waveforms for the minority class.
        Strictly applied to training windows only.
        """
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(labels, return_counts=True)

        if len(classes) < 2:
            return windows.copy(), labels.copy()

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]

        maj_idx = np.where(labels == majority_class)[0]
        min_idx = np.where(labels == minority_class)[0]

        target_min_count = int(len(maj_idx) * target_ratio)
        n_needed = target_min_count - len(min_idx)

        if n_needed <= 0:
            return windows.copy(), labels.copy()

        augmented_windows = []
        for _ in range(n_needed):
            source_idx = rng.choice(min_idx)
            aug_win = self.augment_waveform(windows[source_idx], rng)
            augmented_windows.append(aug_win)

        X_aug = np.array(augmented_windows, dtype=np.float32)
        y_aug = np.full(n_needed, minority_class, dtype=labels.dtype)

        resampled_X = np.concatenate([windows, X_aug], axis=0)
        resampled_y = np.concatenate([labels, y_aug], axis=0)

        perm = rng.permutation(len(resampled_y))
        return resampled_X[perm], resampled_y[perm]

