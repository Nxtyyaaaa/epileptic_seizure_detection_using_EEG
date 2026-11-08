import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler

from src.datasets.chbmit_loader import CHBMITDatasetLoader
from src.datasets.siena_loader import SienaDatasetLoader
from src.datasets.mock_data_generator import create_mock_chbmit_dataset, generate_synthetic_eeg_window, write_minimal_mock_edf
from src.preprocessing.filters import EEGFilter
from src.preprocessing.harmonization import DatasetHarmonizer
from src.features.extractor import EEGFeatureExtractor
from src.balancing.factory import get_balancing_strategy
from src.models.baseline_classifiers import get_classifier
from src.evaluation.metrics import compute_classification_metrics
from src.utils.logging_util import setup_logger

logger = setup_logger("CrossDatasetPipeline")


def create_mock_siena_dataset(output_dir: str, seed: int = 200):
    """Creates a mock Siena external test dataset (512 Hz native)."""
    rng = np.random.default_rng(seed)
    fs = 512.0
    channels = ["EEG FP1", "EEG F7", "EEG T7", "EEG P7", "EEG FP2", "EEG F8"]
    os.makedirs(output_dir, exist_ok=True)

    for pid in ["pn01", "pn02"]:
        p_dir = os.path.join(output_dir, pid)
        os.makedirs(p_dir, exist_ok=True)
        edf_path = os.path.join(p_dir, f"{pid}_01.edf")
        txt_path = os.path.join(p_dir, f"{pid}_01.txt")

        # 90 seconds recording with seizure at 30-50s
        total_sec = 90.0
        n_samples = int(total_sec * fs)
        t = np.linspace(0, total_sec, n_samples, endpoint=False)

        sig_list = []
        for _ in channels:
            base = generate_synthetic_eeg_window(total_sec, fs, is_seizure=False, rng=rng)
            s_mask = (t >= 30.0) & (t <= 50.0)
            ictal = generate_synthetic_eeg_window(total_sec, fs, is_seizure=True, rng=rng)
            base[s_mask] = ictal[s_mask]
            sig_list.append(base)

        write_minimal_mock_edf(edf_path, channels, sig_list, fs=fs)
        with open(txt_path, "w") as f:
            f.write("30.0 50.0\n")


def run_cross_dataset_evaluation(
    chbmit_dir: str = "data/raw/chbmit",
    siena_dir: str = "data/raw/siena",
    model_name: str = "random_forest",
    balancing_strategy: str = "E3",  # Best performing or SMOTE
    harmonized_fs: float = 256.0,
    random_state: int = 42,
    auto_generate_mock: bool = True,
) -> pd.DataFrame:
    """
    Executes Track C: Train/Develop on CHB-MIT, Zero-shot Test on Siena Scalp EEG.
    Harmonizes sampling rate and channels. Measures generalization drop and domain shift.
    """
    logger.info("=== Starting Track C: Cross-Dataset Generalization (CHB-MIT -> Siena) ===")

    # Ensure datasets exist
    if not os.path.exists(chbmit_dir) or not os.listdir(chbmit_dir):
        if auto_generate_mock:
            create_mock_chbmit_dataset(chbmit_dir, seed=random_state)
        else:
            raise FileNotFoundError(f"CHB-MIT data missing at {chbmit_dir}")

    if not os.path.exists(siena_dir) or not os.listdir(siena_dir):
        if auto_generate_mock:
            create_mock_siena_dataset(siena_dir, seed=random_state)
        else:
            raise FileNotFoundError(f"Siena data missing at {siena_dir}")

    # 1. Load CHB-MIT (Source Domain)
    chb_loader = CHBMITDatasetLoader(chbmit_dir)
    chb_patients = chb_loader.list_patients()
    train_patients = chb_patients[:4] if len(chb_patients) >= 4 else chb_patients

    all_chb_wins, all_chb_labs = [], []
    for pid in train_patients:
        w, l, _ = chb_loader.load_patient_records(pid, window_sec=10.0)
        if len(w) > 0:
            all_chb_wins.append(w)
            all_chb_labs.append(l)

    X_chb_raw = np.concatenate(all_chb_wins, axis=0)
    y_chb = np.concatenate(all_chb_labs, axis=0)

    # 2. Load Siena (Target Domain)
    siena_loader = SienaDatasetLoader(siena_dir, target_fs=harmonized_fs)
    siena_patients = siena_loader.list_patients()

    all_siena_wins, all_siena_labs = [], []
    for pid in siena_patients:
        w, l, _ = siena_loader.load_patient_records(pid, window_sec=10.0)
        if len(w) > 0:
            all_siena_wins.append(w)
            all_siena_labs.append(l)

    X_siena_raw = np.concatenate(all_siena_wins, axis=0)
    y_siena = np.concatenate(all_siena_labs, axis=0)

    logger.info(f"Source (CHB-MIT) windows: {len(y_chb)}, Target (Siena) windows: {len(y_siena)}")

    # 3. Filtering & Preprocessing
    filt = EEGFilter(fs=harmonized_fs, lowcut=0.5, highcut=40.0)
    X_chb_filt = np.array([filt.apply(w) for w in X_chb_raw], dtype=np.float32)
    X_siena_filt = np.array([filt.apply(w) for w in X_siena_raw], dtype=np.float32)

    # 4. Feature Extraction
    extractor = EEGFeatureExtractor(fs=harmonized_fs, multichannel_strategy="mean")
    X_chb_feat, _ = extractor.transform(X_chb_filt)
    X_siena_feat, _ = extractor.transform(X_siena_filt)

    # 5. Imbalance Mitigation on CHB-MIT Training Set Only
    sampler, clf_kwargs = get_balancing_strategy(balancing_strategy, random_state=random_state)
    if sampler is not None:
        X_train_res, y_train_res = sampler.fit_resample(X_chb_feat, y_chb)
    else:
        X_train_res, y_train_res = X_chb_feat, y_chb

    # 6. Fit Scaler & Model on Source Domain ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_siena_scaled = scaler.transform(X_siena_feat)

    clf = get_classifier(model_name, random_state=random_state, **clf_kwargs)
    clf.fit(X_train_scaled, y_train_res)

    # Internal Validation performance on source domain
    y_pred_chb = clf.predict(scaler.transform(X_chb_feat))
    y_prob_chb = clf.predict_proba(scaler.transform(X_chb_feat)) if hasattr(clf, "predict_proba") else None
    internal_metrics = compute_classification_metrics(y_chb, y_pred_chb, y_prob_chb)

    # External Zero-shot Generalization on Siena
    y_pred_siena = clf.predict(X_siena_scaled)
    y_prob_siena = clf.predict_proba(X_siena_scaled) if hasattr(clf, "predict_proba") else None
    external_metrics = compute_classification_metrics(y_siena, y_pred_siena, y_prob_siena)

    records = [
        {"evaluation_type": "Internal Source (CHB-MIT)", "model": model_name, "strategy": balancing_strategy, **internal_metrics},
        {"evaluation_type": "External Target (Siena)", "model": model_name, "strategy": balancing_strategy, **external_metrics},
    ]

    df_cross = pd.DataFrame(records)
    logger.info("Track C cross-dataset evaluation completed.")
    return df_cross

