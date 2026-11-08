import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.datasets.edf_reader import SimpleEDFReader
from src.utils.logging_util import setup_logger

logger = setup_logger("SienaLoader")


class SienaDatasetLoader:
    """
    Loader for the Siena Scalp EEG Database (Detti et al. / PhysioNet).
    14 adult patients recorded at 512 Hz.
    Used for Track C (independent external cross-dataset generalization).
    """

    NATIVE_SAMPLING_RATE = 512.0
    TARGET_SAMPLING_RATE = 256.0  # Harmonized to CHB-MIT

    def __init__(self, data_dir: str, target_fs: float = 256.0):
        self.data_dir = data_dir
        self.target_fs = target_fs

    def list_patients(self) -> List[str]:
        if not os.path.exists(self.data_dir):
            return []
        entries = os.listdir(self.data_dir)
        patients = [d for d in entries if os.path.isdir(os.path.join(self.data_dir, d)) and d.lower().startswith("pn")]
        return sorted(patients)

    def parse_siena_annotations(self, annotation_file: str) -> List[Tuple[float, float]]:
        """Parses Siena text annotation file for seizure onset/offset in seconds."""
        seizures: List[Tuple[float, float]] = []
        if not os.path.exists(annotation_file):
            return seizures

        with open(annotation_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Common patterns: onset, offset in seconds
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) >= 2:
                try:
                    start_sec = float(parts[0])
                    end_sec = float(parts[1])
                    if end_sec > start_sec:
                        seizures.append((start_sec, end_sec))
                except ValueError:
                    continue
        return seizures

    def load_patient_records(
        self,
        patient_id: str,
        window_sec: float = 10.0,
        selected_channels: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        patient_dir = os.path.join(self.data_dir, patient_id)
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(f"Siena patient directory not found: {patient_dir}")

        edf_files = sorted(glob.glob(os.path.join(patient_dir, "*.edf")))
        all_windows: List[np.ndarray] = []
        all_labels: List[int] = []

        samples_per_window = int(window_sec * self.target_fs)

        for edf_path in edf_files:
            txt_path = edf_path.replace(".edf", ".txt")
            seizures = self.parse_siena_annotations(txt_path)

            try:
                reader = SimpleEDFReader(edf_path)
            except Exception as e:
                logger.warning(f"Error reading Siena EDF {edf_path}: {e}")
                continue

            # Downsample if needed
            downsample_factor = int(round(self.NATIVE_SAMPLING_RATE / self.target_fs))
            signals = []
            for ch_name, sig in reader.signals.items():
                if downsample_factor > 1:
                    # Simple decimation (anti-aliasing handled in preprocessing)
                    sig_resampled = sig[::downsample_factor]
                else:
                    sig_resampled = sig
                signals.append(sig_resampled)

            if not signals:
                continue

            min_len = min(len(s) for s in signals)
            stacked = np.stack([s[:min_len] for s in signals], axis=0)

            for start_idx in range(0, min_len - samples_per_window + 1, samples_per_window):
                end_idx = start_idx + samples_per_window
                t_start = start_idx / self.target_fs
                t_end = end_idx / self.target_fs

                is_ictal = False
                for s_start, s_end in seizures:
                    if max(t_start, s_start) < min(t_end, s_end):
                        is_ictal = True
                        break

                all_windows.append(stacked[:, start_idx:end_idx])
                all_labels.append(1 if is_ictal else 0)

        if not all_windows:
            return np.empty((0, samples_per_window)), np.empty(0), {}

        return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.int64), {"patient_id": patient_id}

