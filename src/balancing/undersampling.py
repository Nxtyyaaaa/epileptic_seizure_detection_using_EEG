import numpy as np
from typing import Tuple
from src.balancing.base import BaseResampler


class RandomUnderSampler(BaseResampler):
    """
    E2: Random Under-Sampling (RUS).
    Randomly removes majority class samples until balanced ratio is reached.
    """

    def __init__(self, sampling_ratio: float = 1.0, random_state: int = 42):
        super().__init__(random_state)
        self.sampling_ratio = sampling_ratio

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)

        if len(classes) < 2:
            return X.copy(), y.copy()

        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]

        maj_idx = np.where(y == majority_class)[0]
        min_idx = np.where(y == minority_class)[0]

        target_maj_count = int(len(min_idx) / self.sampling_ratio)
        target_maj_count = min(len(maj_idx), max(1, target_maj_count))

        sampled_maj_idx = rng.choice(maj_idx, size=target_maj_count, replace=False)

        all_idx = np.concatenate([sampled_maj_idx, min_idx])
        rng.shuffle(all_idx)

        return X[all_idx], y[all_idx]

