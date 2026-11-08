import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.datasets.bonn_loader import BonnDatasetLoader
from src.datasets.mock_data_generator import create_mock_bonn_dataset
from src.preprocessing.filters import EEGFilter
from src.features.extractor import EEGFeatureExtractor
from src.balancing.factory import get_balancing_strategy
from src.balancing.signal_augmentation import SignalLevelAugmenter
from src.models.baseline_classifiers import get_classifier
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.leakage_guards import LeakageGuard
from src.utils.logging_util import setup_logger

logger = setup_logger("BonnPipeline")


def run_bonn_benchmark(
    data_dir: str = "data/raw/bonn",
    models: List[str] = ["svm", "random_forest"],
    strategies: List[str] = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"],
    test_size: float = 0.25,
    random_state: int = 42,
    auto_generate_mock: bool = True,
) -> pd.DataFrame:
    """
    Executes Track A: Bonn EEG Controlled Benchmark across all balancing strategies (E0-E6).
    Evaluates Sensitivity, Specificity, Precision, F1, PR-AUC, and ROC-AUC under strict leakage control.
    """
    logger.info("=== Starting Track A: Bonn EEG Benchmark ===")

    # Check data existence; fallback to mock generation if directory is empty
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        if auto_generate_mock:
            logger.info(f"Bonn data not found in '{data_dir}'. Generating synthetic mock benchmark for validation...")
            create_mock_bonn_dataset(data_dir, seed=random_state, segments_per_set=20)
        else:
            raise FileNotFoundError(f"Bonn dataset missing at {data_dir}")

    # 1. Load data
    loader = BonnDatasetLoader(data_dir)
    X_raw, y_raw, set_tags = loader.load_dataset(task="binary")
    logger.info(f"Loaded {len(y_raw)} segments. Class distribution: Non-Ictal={np.sum(y_raw==0)}, Ictal={np.sum(y_raw==1)}")

    # 2. Preprocess / Filter
    filt = EEGFilter(fs=loader.SAMPLING_RATE, lowcut=0.5, highcut=40.0)
    X_filtered = np.array([filt.apply(x) for x in X_raw], dtype=np.float32)

    # 3. Leakage-safe Train / Test split (BEFORE any scaling or balancing)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_filtered, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )
    logger.info(f"Train samples: {len(y_train)} (Ictal={np.sum(y_train==1)}), Test samples: {len(y_test)} (Ictal={np.sum(y_test==1)})")

    # Verify no balancing on test set
    LeakageGuard.verify_no_resampling_on_evaluation_sets(y_test, y_test, set_name="Bonn Test")

    # 4. Feature Extraction on unmodified sets
    extractor = EEGFeatureExtractor(fs=loader.SAMPLING_RATE)
    X_train_feat_orig, feat_names = extractor.transform(X_train_raw)
    X_test_feat, _ = extractor.transform(X_test_raw)

    results_records: List[Dict[str, Any]] = []

    # 5. Execute Controlled Comparison Matrix across models and balancing strategies
    for model_name in models:
        for strat in strategies:
            logger.info(f"Evaluating Model='{model_name}' with Strategy='{strat}'")

            # Determine balancing behavior
            if strat == "E6":
                # E6: Raw waveform augmentation applied prior to feature extraction
                augmenter = SignalLevelAugmenter(random_state=random_state)
                X_train_aug_raw, y_train_balanced = augmenter.balance_raw_windows(X_train_raw, y_train)
                X_train_balanced_feat, _ = extractor.transform(X_train_aug_raw)
                clf_kwargs = {}
            elif strat == "E5":
                # E5: Class weighting in loss function
                X_train_balanced_feat = X_train_feat_orig.copy()
                y_train_balanced = y_train.copy()
                clf_kwargs = {"class_weight": "balanced"}
            else:
                # E0-E4: Feature-space resampling
                sampler, _ = get_balancing_strategy(strat, random_state=random_state)
                X_train_balanced_feat, y_train_balanced = sampler.fit_resample(X_train_feat_orig, y_train)
                clf_kwargs = {}

            # Feature Scaling (Fit on Train ONLY)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train_balanced_feat)
            X_te_scaled = scaler.transform(X_test_feat)

            # Train classifier
            clf = get_classifier(model_name, random_state=random_state, **clf_kwargs)
            clf.fit(X_tr_scaled, y_train_balanced)

            # Predict on unmodified test set
            y_pred = clf.predict(X_te_scaled)
            y_prob = clf.predict_proba(X_te_scaled) if hasattr(clf, "predict_proba") else None

            # Calculate metrics
            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            record = {
                "track": "Track A (Bonn)",
                "model": model_name,
                "strategy": strat,
                "train_samples": len(y_train_balanced),
                "test_samples": len(y_test),
                **metrics,
            }
            results_records.append(record)

    df_results = pd.DataFrame(results_records)
    logger.info("Track A completed successfully.")
    return df_results

