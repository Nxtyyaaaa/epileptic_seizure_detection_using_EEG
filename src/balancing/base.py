from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseResampler(ABC):
    """Abstract base class for all class imbalance mitigation strategies."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @abstractmethod
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resamples X and y to mitigate class imbalance.
        Must ONLY be applied to training partitions to prevent data leakage.
        """
        pass

