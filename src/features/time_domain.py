import numpy as np
from scipy import stats
from typing import Dict


def extract_time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """
    Computes statistical and amplitude distribution descriptors for a 1D EEG window.
    """
    if len(x) == 0:
        return {
            "mean": 0.0, "var": 0.0, "std": 0.0, "rms": 0.0,
            "skewness": 0.0, "kurtosis": 0.0, "peak_to_peak": 0.0,
            "zero_crossings": 0.0,
        }

    mean_val = float(np.mean(x))
    var_val = float(np.var(x))
    std_val = float(np.std(x))
    rms_val = float(np.sqrt(np.mean(x ** 2)))
    peak_to_peak = float(np.ptp(x))

    # Skewness and kurtosis
    skew_val = float(stats.skew(x, bias=False)) if var_val > 1e-8 else 0.0
    kurt_val = float(stats.kurtosis(x, bias=False)) if var_val > 1e-8 else 0.0

    # Zero-crossing count
    zero_crossings = float(np.sum(np.diff(np.signbit(x)) != 0))

    return {
        "mean": mean_val,
        "var": var_val,
        "std": std_val,
        "rms": rms_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "peak_to_peak": peak_to_peak,
        "zero_crossings": zero_crossings,
    }

