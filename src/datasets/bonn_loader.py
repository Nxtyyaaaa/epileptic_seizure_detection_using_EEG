import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional


class BonnDatasetLoader:
    """
    Loader for the University of Bonn EEG benchmark dataset (Andrzejak et al., 2001).
    Consists of 5 sets (A-E / Z, O, N, F, S), each with 100 single-channel segments
    of 4,096 samples sampled at 173.61 Hz.
    """

    SAMPLING_RATE = 173.61
    SAMPLES_PER_SEGMENT = 4096
    SET_MAPPING = {
        "A": "Z",  # Surface EEG, healthy, eyes open
        "B": "O",  # Surface EEG, healthy, eyes closed
        "C": "N",  # Intracranial, seizure-free, hippocampal formation
        "D": "F",  # Intracranial, seizure-free, epileptogenic zone
        "E": "S",  # Intracranial, seizure activity (Ictal)
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_dataset(
        self,
        task: str = "binary",  # "binary" (E vs A-D) or "multiclass" (A, B, C, D, E)
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Loads all available Bonn segments from the directory structure.
        Returns:
            X: np.ndarray of shape (N, 4096)
            y: np.ndarray of shape (N,)
            set_labels: list of original set tags ('A', 'B', 'C', 'D', 'E')
        """
        segments: List[np.ndarray] = []
        labels: List[int] = []
        set_tags: List[str] = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Bonn dataset directory not found: {self.data_dir}")

        # Search patterns: both directory-per-set or flat directory with file prefixes
        for set_letter, prefix in self.SET_MAPPING.items():
            pattern1 = os.path.join(self.data_dir, set_letter, "*.txt")
            pattern2 = os.path.join(self.data_dir, prefix, "*.txt")
            pattern3 = os.path.join(self.data_dir, f"{prefix}*.txt")
            pattern4 = os.path.join(self.data_dir, f"{set_letter}*.txt")

            file_list = sorted(list(set(
                glob.glob(pattern1) + glob.glob(pattern2) + glob.glob(pattern3) + glob.glob(pattern4)
            )))

            for file_path in file_list:
                try:
                    data = np.loadtxt(file_path, dtype=np.float32)
                    if len(data) != self.SAMPLES_PER_SEGMENT:
                        # Truncate or pad to exactly 4096 if slight difference
                        if len(data) > self.SAMPLES_PER_SEGMENT:
                            data = data[: self.SAMPLES_PER_SEGMENT]
                        else:
                            continue
                    segments.append(data)
                    set_tags.append(set_letter)

                    if task == "binary":
                        # Set E (ictal) = 1, Sets A-D (non-ictal) = 0
                        labels.append(1 if set_letter == "E" else 0)
                    else:
                        class_idx = ["A", "B", "C", "D", "E"].index(set_letter)
                        labels.append(class_idx)
                except Exception as e:
                    continue

        if not segments:
            raise ValueError(f"No valid Bonn EEG segments found in {self.data_dir}")

        X = np.array(segments, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        return X, y, set_tags

