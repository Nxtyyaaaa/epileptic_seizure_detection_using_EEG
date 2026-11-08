import os
import re
import glob
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.datasets.edf_reader import SimpleEDFReader
from src.utils.logging_util import setup_logger

logger = setup_logger("CHBMITLoader")

# Standard 18 bipolar channels common across CHB-MIT recordings
COMMON_CHB_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]


class CHBMITSummaryParser:
    """Parses CHB-MIT subject summary text files for seizure intervals and metadata."""

    @staticmethod
    def parse_summary_file(summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parses a chbXX-summary.txt file.
        Returns a dict mapping edf_filename -> list of (start_sec, end_sec) tuples.
        """
        if not os.path.exists(summary_path):
            logger.warning(f"Summary file not found: {summary_path}")
            return {}

        seizure_map: Dict[str, List[Tuple[float, float]]] = {}
        current_file: Optional[str] = None

        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Match file name
            file_match = re.match(r"^File Name:\s*([A-Za-z0-9_\.]+\.edf)", line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1).strip()
                if current_file not in seizure_map:
                    seizure_map[current_file] = []
                continue

            # Match seizure start / end
            start_match = re.search(r"Seizure\s*(\d*)\s*Start Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)
            end_match = re.search(r"Seizure\s*(\d*)\s*End Time:\s*(\d+)\s*seconds", line, re.IGNORECASE)

            if start_match and current_file:
                start_sec = float(start_match.group(2))
                # Temporary store
                seizure_map[current_file].append((start_sec, -1.0))
            elif end_match and current_file:
                end_sec = float(end_match.group(2))
                if seizure_map[current_file] and seizure_map[current_file][-1][1] == -1.0:
                    start_sec = seizure_map[current_file][-1][0]
                    seizure_map[current_file][-1] = (start_sec, end_sec)

        # Filter out incomplete entries
        for fn in list(seizure_map.keys()):
            seizure_map[fn] = [pair for pair in seizure_map[fn] if pair[1] > pair[0]]

        return seizure_map


class CHBMITDatasetLoader:
    """
    Loader for the CHB-MIT Scalp EEG database (Goldberger et al. / PhysioNet).
    Handles patient-level splits, annotation extraction, windowing, and channel mapping.
    """

    SAMPLING_RATE = 256.0

    def __init__(self, data_dir: str, selected_channels: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.selected_channels = selected_channels or COMMON_CHB_CHANNELS

    def list_patients(self) -> List[str]:
        """Returns sorted list of available patient folders (e.g. ['chb01', 'chb02', ...])."""
        if not os.path.exists(self.data_dir):
            return []
        entries = os.listdir(self.data_dir)
        patients = [d for d in entries if os.path.isdir(os.path.join(self.data_dir, d)) and d.lower().startswith("chb")]
        return sorted(patients)

    def load_patient_records(
        self,
        patient_id: str,
        window_sec: float = 10.0,
        overlap_sec: float = 0.0,
        seizure_overlap_threshold: float = 0.5,
        max_non_seizure_hours: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Loads all EDF records for a single patient, extracting windows and binary labels.
        Returns:
            X: (N_windows, num_channels, samples_per_window) or (N, samples)
            y: (N_windows,) where 1=ictal, 0=non-ictal
            metadata: dictionary of recording info and event timelines
        """
        patient_path = os.path.join(self.data_dir, patient_id)
        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"Patient directory not found: {patient_path}")

        # Find summary file
        summary_files = glob.glob(os.path.join(patient_path, "*summary.txt"))
        seizure_annotations: Dict[str, List[Tuple[float, float]]] = {}
        if summary_files:
            seizure_annotations = CHBMITSummaryParser.parse_summary_file(summary_files[0])

        edf_files = sorted(glob.glob(os.path.join(patient_path, "*.edf")))
        all_windows: List[np.ndarray] = []
        all_labels: List[int] = []
        timeline_info: List[Dict[str, Any]] = []

        samples_per_window = int(window_sec * self.SAMPLING_RATE)
        step_samples = int((window_sec - overlap_sec) * self.SAMPLING_RATE)

        for edf_path in edf_files:
            file_name = os.path.basename(edf_path)
            seizures = seizure_annotations.get(file_name, [])

            try:
                reader = SimpleEDFReader(edf_path)
            except Exception as e:
                logger.warning(f"Could not read EDF {file_name}: {e}")
                continue

            # Extract selected channels
            channel_signals = []
            for ch in self.selected_channels:
                sig = reader.get_signal(ch)
                if sig is not None:
                    channel_signals.append(sig)

            if not channel_signals:
                # If named channels not matched, fallback to first available channels
                all_ch_keys = list(reader.signals.keys())[:len(self.selected_channels)]
                channel_signals = [reader.signals[k] for k in all_ch_keys]

            if not channel_signals:
                continue

            min_len = min(len(s) for s in channel_signals)
            stacked = np.stack([s[:min_len] for s in channel_signals], axis=0)  # (C, T)

            # Generate windows
            rec_windows = 0
            rec_seizures = 0
            for start_idx in range(0, min_len - samples_per_window + 1, step_samples):
                end_idx = start_idx + samples_per_window
                t_start = start_idx / self.SAMPLING_RATE
                t_end = end_idx / self.SAMPLING_RATE

                # Check seizure overlap
                is_ictal = False
                for s_start, s_end in seizures:
                    overlap_start = max(t_start, s_start)
                    overlap_end = min(t_end, s_end)
                    overlap_duration = max(0.0, overlap_end - overlap_start)
                    if overlap_duration / window_sec >= seizure_overlap_threshold:
                        is_ictal = True
                        break

                all_windows.append(stacked[:, start_idx:end_idx])
                all_labels.append(1 if is_ictal else 0)
                rec_windows += 1
                if is_ictal:
                    rec_seizures += 1

            timeline_info.append({
                "file_name": file_name,
                "duration_sec": min_len / self.SAMPLING_RATE,
                "total_windows": rec_windows,
                "ictal_windows": rec_seizures,
                "seizures": seizures
            })

        if not all_windows:
            return np.empty((0, len(self.selected_channels), samples_per_window)), np.empty(0), {"records": []}

        X = np.array(all_windows, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int64)

        return X, y, {"patient_id": patient_id, "records": timeline_info}

