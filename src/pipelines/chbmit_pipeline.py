import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

from src.datasets.chbmit_loader import CHBMITDatasetLoader, COMMON_CHB_CHANNELS
from src.datasets.mock_data_generator import create_mock_chbmit_dataset
from src.preprocessing.filters import EEGFilter
from src.features.extractor import EEGFeatureExtractor
from src.balancing.factory import get_balancing_strategy
from src.balancing.signal_augmentation import SignalLevelAugmenter
from src.models.baseline_classifiers import get_classifier
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.timeline_eval import compute_timeline_event_metrics
from src.evaluation.leakage_guards import LeakageGuard
from src.utils.logging_util import setup_logger

logger = setup_logger("CHBMITPipeline")


def run_chbmit_patient_experiment(
    data_dir: str = "data/raw/chbmit",
    train_patients: Optional[List[str]] = None,
    test_patients: Optional[List[str]] = None,
    models: List[str] = ["svm", "random_forest"],
    strategies: List[str] = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"],
    window_sec: float = 10.0,
    random_state: int = 42,
    auto_generate_mock: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes Track B: CHB-MIT Clinical Patient-Grouped Seizure Detection.
    Strictly splits at patient level to avoid patient identity leakage.
    Evaluates both window-level metrics and timeline event false alarms per hour (FA/h).
    """
    logger.info("=== Starting Track B: CHB-MIT Patient-Independent Evaluation ===")

    # Setup directories and mock data fallback
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        if auto_generate_mock:
            logger.info(f"CHB-MIT data not found in '{data_dir}'. Generating mock clinical pilot dataset...")
            create_mock_chbmit_dataset(data_dir, seed=random_state)
        else:
            raise FileNotFoundError(f"CHB-MIT dataset missing at {data_dir}")

    loader = CHBMITDatasetLoader(data_dir)
    available_patients = loader.list_patients()
    if not available_patients:
        raise ValueError(f"No patient folders found in {data_dir}")

    # Set default patient split if not provided
    if train_patients is None or test_patients is None:
        if len(available_patients) >= 4:
            train_patients = available_patients[:-2]
            test_patients = available_patients[-2:]
        else:
            train_patients = [available_patients[0]]
            test_patients = [available_patients[-1]]

    # 1. Verify Leakage Safety: Disjoint patient sets
    LeakageGuard.verify_patient_disjoint(train_patients, [], test_patients)
    logger.info(f"Patient-Grouped Split: TRAIN={train_patients}, TEST={test_patients}")

    # 2. Load Patient Windows
    def load_partition(p_list: List[str]):
        wins, labs, timelines = [], [], []
        for pid in p_list:
            w, l, meta = loader.load_patient_records(pid, window_sec=window_sec)
            if len(w) > 0:
                wins.append(w)
                labs.append(l)
                timelines.append(meta)
        if not wins:
            return np.empty(0), np.empty(0), []
        return np.concatenate(wins, axis=0), np.concatenate(labs, axis=0), timelines

    X_train_raw, y_train, train_meta = load_partition(train_patients)
    X_test_raw, y_test, test_meta = load_partition(test_patients)

    logger.info(
        f"Train Partition: {len(y_train)} windows (Ictal={np.sum(y_train==1)}, Non-Ictal={np.sum(y_train==0)})"
    )
    logger.info(
        f"Test Partition: {len(y_test)} windows (Ictal={np.sum(y_test==1)}, Non-Ictal={np.sum(y_test==0)})"
    )

    # 3. Preprocessing (Filtering)
    filt = EEGFilter(fs=loader.SAMPLING_RATE, lowcut=0.5, highcut=40.0)
    X_train_filt = np.array([filt.apply(w) for w in X_train_raw], dtype=np.float32)
    X_test_filt = np.array([filt.apply(w) for w in X_test_raw], dtype=np.float32)

    # 4. Feature Extraction on train & test
    extractor = EEGFeatureExtractor(fs=loader.SAMPLING_RATE, multichannel_strategy="mean")
    X_train_feat_orig, _ = extractor.transform(X_train_filt)
    X_test_feat, _ = extractor.transform(X_test_filt)

    results_records: List[Dict[str, Any]] = []
    timeline_records: Dict[str, Any] = {}

    # Total test duration for FA/h
    test_total_sec = sum(
        rec.get("duration_sec", 0.0)
        for t in test_meta
        for rec in t.get("records", [])
    ) or (len(y_test) * window_sec)

    # Extract all ground truth seizure intervals for test patients
    test_gt_seizures: List[Tuple[float, float]] = []
    current_time_offset = 0.0
    for t in test_meta:
        for rec in t.get("records", []):
            rec_dur = rec.get("duration_sec", 0.0)
            for s_start, s_end in rec.get("seizures", []):
                test_gt_seizures.append((current_time_offset + s_start, current_time_offset + s_end))
            current_time_offset += rec_dur

    test_window_bounds = np.array([
        (i * window_sec, (i + 1) * window_sec) for i in range(len(y_test))
    ], dtype=np.float32)

    # 5. Run Controlled Comparison Matrix
    for model_name in models:
        for strat in strategies:
            logger.info(f"Running CHB-MIT: Model={model_name}, Strategy={strat}")

            if strat == "E6":
                augmenter = SignalLevelAugmenter(random_state=random_state)
                X_tr_aug, y_tr_bal = augmenter.balance_raw_windows(X_train_filt, y_train)
                X_tr_feat, _ = extractor.transform(X_tr_aug)
                clf_kwargs = {}
            elif strat == "E5":
                X_tr_feat = X_train_feat_orig.copy()
                y_tr_bal = y_train.copy()
                clf_kwargs = {"class_weight": "balanced"}
            else:
                sampler, _ = get_balancing_strategy(strat, random_state=random_state)
                X_tr_feat, y_tr_bal = sampler.fit_resample(X_train_feat_orig, y_train)
                clf_kwargs = {}

            # Scaler fitted strictly on training data
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_feat)
            X_te_scaled = scaler.transform(X_test_feat)

            clf = get_classifier(model_name, random_state=random_state, **clf_kwargs)
            clf.fit(X_tr_scaled, y_tr_bal)

            y_pred = clf.predict(X_te_scaled)
            y_prob = clf.predict_proba(X_te_scaled) if hasattr(clf, "predict_proba") else None

            # Window-level metrics
            win_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

            # Timeline event-level metrics (False Alarms per hour)
            tl_metrics = compute_timeline_event_metrics(
                window_predictions=y_pred,
                window_bounds=test_window_bounds,
                ground_truth_seizures=test_gt_seizures,
                total_duration_sec=test_total_sec,
            )

            record = {
                "track": "Track B (CHB-MIT)",
                "model": model_name,
                "strategy": strat,
                "train_windows": len(y_tr_bal),
                "test_windows": len(y_test),
                **win_metrics,
                "event_sensitivity": tl_metrics["event_sensitivity"],
                "false_alarms_per_hour": tl_metrics["false_alarms_per_hour"],
                "total_events_detected": tl_metrics["total_detected_events"],
            }
            results_records.append(record)
            timeline_records[f"{model_name}_{strat}"] = tl_metrics

    df_results = pd.DataFrame(results_records)
    logger.info("Track B completed successfully.")
    return df_results, timeline_records
