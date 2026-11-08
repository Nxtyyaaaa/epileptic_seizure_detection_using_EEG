import numpy as np
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from src.balancing.base import BaseResampler


class SMOTEResampler(BaseResampler):
    """
    E3: Synthetic Minority Over-sampling Technique (SMOTE).
    Synthesizes new feature vectors along the line segment joining k-nearest minority neighbors.
    """

    def __init__(self, k_neighbors: int = 5, sampling_ratio: float = 1.0, random_state: int = 42):
        super().__init__(random_state)
        self.k_neighbors = k_neighbors
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

        X_min = X[min_idx]
        target_min_count = int(len(maj_idx) * self.sampling_ratio)
        n_synthetic_needed = target_min_count - len(min_idx)

        if n_synthetic_needed <= 0:
            return X.copy(), y.copy()

        k = min(self.k_neighbors, len(min_idx) - 1)
        if k < 1:
            # Fallback to simple random oversampling if only 1 minority sample
            extra_idx = rng.choice(min_idx, size=n_synthetic_needed, replace=True)
            return np.vstack([X, X[extra_idx]]), np.concatenate([y, y[extra_idx]])

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X_min)
        distances, indices = nn.kneighbors(X_min)

        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            # Pick a random minority sample
            sample_idx = rng.integers(0, len(min_idx))
            # Pick one of its k nearest neighbors (excluding self at index 0)
            neighbor_choice = rng.integers(1, k + 1)
            neighbor_idx = indices[sample_idx, neighbor_choice]

            diff = X_min[neighbor_idx] - X_min[sample_idx]
            lamb = rng.uniform(0.0, 1.0)
            synthetic_point = X_min[sample_idx] + lamb * diff
            synthetic_samples.append(synthetic_point)

        X_synth = np.array(synthetic_samples, dtype=np.float32)
        y_synth = np.full(n_synthetic_needed, minority_class, dtype=y.dtype)

        X_resampled = np.vstack([X, X_synth])
        y_resampled = np.concatenate([y, y_synth])

        perm = rng.permutation(len(y_resampled))
        return X_resampled[perm], y_resampled[perm]


class BorderlineSMOTEResampler(BaseResampler):
    """
    E4: Borderline-SMOTE (Han et al., 2005).
    Identifies minority samples situated in the 'DANGER' boundary zone
    (where majority samples comprise more than half of the neighborhood)
    and synthesizes samples exclusively from these critical boundary points.
    """

    def __init__(self, m_neighbors: int = 5, k_neighbors: int = 5, sampling_ratio: float = 1.0, random_state: int = 42):
        super().__init__(random_state)
        self.m_neighbors = m_neighbors
        self.k_neighbors = k_neighbors
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

        X_min = X[min_idx]
        target_min_count = int(len(maj_idx) * self.sampling_ratio)
        n_synthetic_needed = target_min_count - len(min_idx)

        if n_synthetic_needed <= 0:
            return X.copy(), y.copy()

        # Step 1: Find m nearest neighbors in the entire dataset X
        m = min(self.m_neighbors, len(X) - 1)
        if m < 1:
            return SMOTEResampler(k_neighbors=self.k_neighbors, random_state=self.random_state).fit_resample(X, y)

        nn_all = NearestNeighbors(n_neighbors=m + 1).fit(X)
        _, indices_all = nn_all.kneighbors(X_min)

        # Step 2: Determine 'DANGER' minority samples
        danger_indices = []
        for i, neighbor_indices in enumerate(indices_all):
            # Exclude self (index 0)
            n_maj = sum(1 for idx in neighbor_indices[1:] if y[idx] == majority_class)
            # In DANGER zone if m/2 <= n_maj < m
            if (m / 2.0) <= n_maj < m:
                danger_indices.append(i)

        # If danger set is empty, fall back to all minority points
        if not danger_indices:
            danger_indices = list(range(len(min_idx)))

        X_danger = X_min[danger_indices]

        # Step 3: For each danger point, find k nearest neighbors within minority class
        k = min(self.k_neighbors, len(X_min) - 1)
        if k < 1:
            # Fallback
            extra_idx = rng.choice(min_idx, size=n_synthetic_needed, replace=True)
            return np.vstack([X, X[extra_idx]]), np.concatenate([y, y[extra_idx]])

        nn_min = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
        _, indices_min = nn_min.kneighbors(X_danger)

        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            d_idx = rng.integers(0, len(danger_indices))
            neighbor_choice = rng.integers(1, k + 1)
            neighbor_sample_idx = indices_min[d_idx, neighbor_choice]

            diff = X_min[neighbor_sample_idx] - X_danger[d_idx]
            lamb = rng.uniform(0.0, 1.0)
            synthetic_point = X_danger[d_idx] + lamb * diff
            synthetic_samples.append(synthetic_point)

        X_synth = np.array(synthetic_samples, dtype=np.float32)
        y_synth = np.full(n_synthetic_needed, minority_class, dtype=y.dtype)

        X_resampled = np.vstack([X, X_synth])
        y_resampled = np.concatenate([y, y_synth])

        perm = rng.permutation(len(y_resampled))
        return X_resampled[perm], y_resampled[perm]

