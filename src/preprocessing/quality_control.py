import numpy as np
from typing import Dict, Any, Tuple


class SignalQualityChecker:
    """
    Performs signal quality audit on EEG windows or channels:
    - Detects flatlines / zero-variance segments
    - Detects clipping and amplifier saturation
    - Flags extreme amplitude excursions (> physiological threshold e.g. 500 uV)
    """

    def __init__(
        self,
        flatline_var_threshold: float = 1e-4,
        clipping_threshold_percentile: float = 99.9,
        extreme_amplitude_uv: float = 800.0,
    ):
        self.flatline_var_threshold = flatline_var_threshold
        self.clipping_threshold_percentile = clipping_threshold_percentile
        self.extreme_amplitude_uv = extreme_amplitude_uv

    def check_window(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Audits a single EEG window x (1D or 2D).
        Returns a dictionary with anomaly flags and statistics.
        """
        variance = float(np.var(x))
        max_abs_val = float(np.max(np.abs(x)))

        is_flatline = variance < self.flatline_var_threshold
        is_extreme_excursion = max_abs_val > self.extreme_amplitude_uv

        # Check clipping: count consecutive identical values
        flat_runs = 0
        if x.ndim == 1:
            diffs = np.diff(x)
            zero_diffs = np.sum(diffs == 0)
            flat_runs = int(zero_diffs)
        is_clipped = flat_runs > (0.1 * x.shape[-1])

        is_usable = (not is_flatline) and (not is_clipped)

        return {
            "is_usable": is_usable,
            "is_flatline": is_flatline,
            "is_clipped": is_clipped,
            "is_extreme_excursion": is_extreme_excursion,
            "variance": variance,
            "max_abs_amplitude": max_abs_val,
        }

