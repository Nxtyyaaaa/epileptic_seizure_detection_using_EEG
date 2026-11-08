import numpy as np
from src.balancing.factory import get_balancing_strategy
from src.balancing.signal_augmentation import SignalLevelAugmenter


def test_balancing_strategies():
    # Create synthetic imbalanced dataset: 90 non-seizure (0), 10 seizure (1)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 10)).astype(np.float32)
    y = np.array([0] * 90 + [1] * 10, dtype=np.int64)

    for strat in ["E0", "E1", "E2", "E3", "E4", "E5"]:
        sampler, clf_kwargs = get_balancing_strategy(strat, random_state=42)
        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X, y)
            assert len(y_res) > 0
            if strat == "E1" or strat == "E3" or strat == "E4":
                # Oversampled
                assert np.sum(y_res == 1) > 10
            elif strat == "E2":
                # Undersampled
                assert np.sum(y_res == 0) <= 90
            elif strat == "E0":
                assert len(y_res) == 100
        else:
            assert clf_kwargs.get("class_weight") == "balanced"


def test_signal_augmentation_e6():
    raw_windows = np.random.randn(20, 256).astype(np.float32)
    labels = np.array([0] * 18 + [1] * 2, dtype=np.int64)
    augmenter = SignalLevelAugmenter(random_state=42)
    X_aug, y_aug = augmenter.balance_raw_windows(raw_windows, labels, target_ratio=1.0)
    assert np.sum(y_aug == 1) > 2
    assert len(y_aug) > len(labels)

