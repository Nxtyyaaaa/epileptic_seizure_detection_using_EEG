import struct
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class SimpleEDFReader:
    """
    Pure Python/NumPy reader for European Data Format (.edf) files.
    Works standalone without requiring mne or pyedflib C-extensions.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header: Dict[str, Any] = {}
        self.signals: Dict[str, np.ndarray] = {}
        self.sample_rates: Dict[str, float] = {}
        self._read_header_and_data()

    def _read_header_and_data(self):
        with open(self.file_path, "rb") as f:
            # First 256 bytes
            version = f.read(8).decode("ascii", errors="ignore").strip()
            patient_id = f.read(80).decode("ascii", errors="ignore").strip()
            recording_id = f.read(80).decode("ascii", errors="ignore").strip()
            start_date = f.read(8).decode("ascii", errors="ignore").strip()
            start_time = f.read(8).decode("ascii", errors="ignore").strip()
            header_bytes = int(f.read(8).decode("ascii", errors="ignore").strip() or 0)
            reserved = f.read(44)
            num_records = int(f.read(8).decode("ascii", errors="ignore").strip() or 0)
            record_duration = float(f.read(8).decode("ascii", errors="ignore").strip() or 1.0)
            num_signals = int(f.read(4).decode("ascii", errors="ignore").strip() or 0)

            if num_signals <= 0:
                raise ValueError(f"Invalid signal count {num_signals} in EDF: {self.file_path}")

            # Per-signal headers (each field is num_signals * length)
            labels = [f.read(16).decode("ascii", errors="ignore").strip() for _ in range(num_signals)]
            transducers = [f.read(80).decode("ascii", errors="ignore").strip() for _ in range(num_signals)]
            dimensions = [f.read(8).decode("ascii", errors="ignore").strip() for _ in range(num_signals)]
            phys_mins = [float(f.read(8).decode("ascii", errors="ignore").strip() or 0.0) for _ in range(num_signals)]
            phys_maxs = [float(f.read(8).decode("ascii", errors="ignore").strip() or 1.0) for _ in range(num_signals)]
            dig_mins = [float(f.read(8).decode("ascii", errors="ignore").strip() or -32768) for _ in range(num_signals)]
            dig_maxs = [float(f.read(8).decode("ascii", errors="ignore").strip() or 32767) for _ in range(num_signals)]
            prefilterings = [f.read(80).decode("ascii", errors="ignore").strip() for _ in range(num_signals)]
            nr_samples_per_rec = [int(f.read(8).decode("ascii", errors="ignore").strip() or 0) for _ in range(num_signals)]
            f.read(32 * num_signals)  # reserved per-signal

            self.header = {
                "version": version,
                "patient_id": patient_id,
                "recording_id": recording_id,
                "start_date": start_date,
                "start_time": start_time,
                "num_records": num_records,
                "record_duration": record_duration,
                "num_signals": num_signals,
                "channel_names": labels,
                "units": dimensions,
                "samples_per_record": nr_samples_per_rec,
            }

            # Calculate scaling factors
            scales = []
            offsets = []
            for i in range(num_signals):
                d_range = dig_maxs[i] - dig_mins[i]
                p_range = phys_maxs[i] - phys_mins[i]
                scale = p_range / d_range if d_range != 0 else 1.0
                offset = phys_mins[i] - dig_mins[i] * scale
                scales.append(scale)
                offsets.append(offset)
                fs = nr_samples_per_rec[i] / record_duration if record_duration > 0 else 256.0
                self.sample_rates[labels[i]] = fs

            # Read record data
            total_samples_per_rec = sum(nr_samples_per_rec)
            raw_data = np.frombuffer(f.read(), dtype=np.int16)

            actual_records = len(raw_data) // total_samples_per_rec if total_samples_per_rec > 0 else 0
            if num_records > 0 and actual_records > 0:
                effective_records = min(num_records, actual_records)
            else:
                effective_records = actual_records

            # Initialize arrays
            channel_arrays = {
                labels[i]: np.zeros(effective_records * nr_samples_per_rec[i], dtype=np.float32)
                for i in range(num_signals)
            }

            # Reshape or unpack records
            offset_idx = 0
            for rec in range(effective_records):
                for i in range(num_signals):
                    ns = nr_samples_per_rec[i]
                    chunk = raw_data[offset_idx : offset_idx + ns]
                    offset_idx += ns
                    start_pos = rec * ns
                    channel_arrays[labels[i]][start_pos : start_pos + ns] = chunk * scales[i] + offsets[i]

            self.signals = channel_arrays

    def get_signal(self, channel_name: str) -> Optional[np.ndarray]:
        """Returns signal for a specific channel name (case-insensitive search)."""
        if channel_name in self.signals:
            return self.signals[channel_name]
        for k, v in self.signals.items():
            if k.lower() == channel_name.lower():
                return v
        return None

    def get_common_sample_rate(self) -> float:
        """Returns dominant sampling frequency."""
        if not self.sample_rates:
            return 256.0
        return list(self.sample_rates.values())[0]

