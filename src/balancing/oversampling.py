import numpy as np
from typing import Tuple
from src.balancing.base import BaseResampler


class RandomOverSampler(BaseResampler):
    """
    E1: Random Over-Sampling (ROS).
    Duplicates minority class samples with replacement until class balance is reached.
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

        target_min_count = int(len(maj_idx) * self.sampling_ratio)
        if target_min_count <= len(min_idx):
            return X.copy(), y.copy()

        # Oversample minority indices with replacement
        additional_needed = target_min_count - len(min_idx)
        oversampled_min_idx = rng.choice(min_idx, size=additional_needed, replace=True)

        all_idx = np.concatenate([maj_idx, min_idx, oversampled_min_idx])
        rng.shuffle(all_idx)

        return X[all_idx], y[all_idx]

