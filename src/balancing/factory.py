from typing import Optional, Tuple, Dict, Any
import numpy as np
from src.balancing.base import BaseResampler
from src.balancing.oversampling import RandomOverSampler
from src.balancing.undersampling import RandomUnderSampler
from src.balancing.smote import SMOTEResampler, BorderlineSMOTEResampler
from src.balancing.signal_augmentation import SignalLevelAugmenter


class NoBalancing(BaseResampler):
    """E0: Baseline condition with no sampling modification."""

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return X.copy(), y.copy()


def get_balancing_strategy(
    strategy_id: str,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[Optional[BaseResampler], Dict[str, Any]]:
    """
    Returns the appropriate resampler and classifier kwargs for experiment condition:
    E0: Baseline (no balancing)
    E1: Random Over-Sampling
    E2: Random Under-Sampling
    E3: SMOTE
    E4: Borderline-SMOTE
    E5: Class-Weighted Learning (sampler=None, class_weight='balanced')
    E6: Signal-Level Augmentation
    """
    strat = strategy_id.upper()
    classifier_kwargs = {}

    if strat == "E0":
        return NoBalancing(random_state=random_state), {}
    elif strat == "E1":
        return RandomOverSampler(random_state=random_state, **kwargs), {}
    elif strat == "E2":
        return RandomUnderSampler(random_state=random_state, **kwargs), {}
    elif strat == "E3":
        return SMOTEResampler(random_state=random_state, **kwargs), {}
    elif strat == "E4":
        return BorderlineSMOTEResampler(random_state=random_state, **kwargs), {}
    elif strat == "E5":
        return None, {"class_weight": "balanced"}
    elif strat == "E6":
        return SignalLevelAugmenter(random_state=random_state, **kwargs), {}
    else:
        raise ValueError(f"Unknown balancing strategy ID: {strategy_id}. Expected E0, E1, E2, E3, E4, E5, or E6.")

