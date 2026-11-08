import numpy as np
from typing import Dict


def compute_hjorth_parameters(x: np.ndarray) -> Dict[str, float]:
    """
    Computes Hjorth parameters:
    - Activity: variance of signal amplitude.
    - Mobility: square root of variance of first derivative divided by variance of signal.
    - Complexity: ratio of mobility of first derivative to mobility of signal.
    """
    if len(x) < 4:
        return {"hjorth_activity": 0.0, "hjorth_mobility": 0.0, "hjorth_complexity": 0.0}

    var_0 = np.var(x)
    if var_0 < 1e-12:
        return {"hjorth_activity": float(var_0), "hjorth_mobility": 0.0, "hjorth_complexity": 0.0}

    # First derivative
    dx = np.diff(x)
    var_d1 = np.var(dx)

    # Mobility
    mobility = np.sqrt(var_d1 / var_0)

    # Second derivative
    ddx = np.diff(dx)
    var_d2 = np.var(ddx)

    # Complexity
    if var_d1 < 1e-12:
        complexity = 0.0
    else:
        mobility_d1 = np.sqrt(var_d2 / var_d1)
        complexity = mobility_d1 / mobility if mobility > 1e-12 else 0.0

    return {
        "hjorth_activity": float(var_0),
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }


def compute_shannon_entropy(x: np.ndarray, num_bins: int = 16) -> float:
    """Computes Shannon entropy of the amplitude histogram."""
    if len(x) == 0:
        return 0.0
    counts, _ = np.histogram(x, bins=num_bins)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Computes Sample Entropy (SampEn) for time series complexity.
    Uses Chebyshev distance and downsamples if signal is large for speed.
    """
    n = len(x)
    if n < 20:
        return 0.0

    # Subsample if length > 256 for rapid computation
    step = max(1, n // 256)
    sig = x[::step]
    n_sub = len(sig)
    r_val = r * np.std(sig)
    if r_val < 1e-8:
        return 0.0

    def _count_matches(dim: int) -> int:
        vectors = np.array([sig[i : i + dim] for i in range(n_sub - dim + 1)])
        # Pairwise Chebyshev (max) distances
        diffs = np.abs(vectors[:, None, :] - vectors[None, :, :])
        max_diffs = np.max(diffs, axis=-1)
        # Exclude self matches
        np.fill_diagonal(max_diffs, np.inf)
        return int(np.sum(max_diffs < r_val))

    count_m = _count_matches(m)
    count_m1 = _count_matches(m + 1)

    if count_m == 0 or count_m1 == 0:
        return 0.0
    return float(-np.log(count_m1 / count_m))


def extract_nonlinear_features(x: np.ndarray) -> Dict[str, float]:
    """Extracts all nonlinear and complexity descriptors for an EEG window."""
    feats = compute_hjorth_parameters(x)
    feats["shannon_entropy"] = compute_shannon_entropy(x)
    feats["sample_entropy"] = compute_sample_entropy(x)
    return feats

