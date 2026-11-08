import os
import struct
import numpy as np
from typing import List, Tuple


def generate_synthetic_eeg_window(
    duration_sec: float,
    fs: float,
    is_seizure: bool,
    rng: np.random.Generator,
    num_channels: int = 1,
) -> np.ndarray:
    """
    Generates realistic synthetic EEG signals.
    - Non-seizure: Mixture of 1/f pink noise, physiological alpha (8-12 Hz) & beta (13-30 Hz) rhythms.
    - Seizure (Ictal): High-amplitude 3-Hz rhythmic spike-and-slow-wave complexes and fast poly-spikes.
    """
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    channels_data = []

    for _ in range(num_channels):
        # Background pink-like noise
        noise = rng.standard_normal(n_samples)
        # Low frequency drift
        drift = 10.0 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
        # Normal physiological alpha rhythm (8 - 12 Hz)
        alpha = 25.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        # Normal beta rhythm (15 - 25 Hz)
        beta = 12.0 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))

        sig = drift + alpha + beta + 15.0 * noise

        if is_seizure:
            # Ictal spike-and-wave pattern: 3 Hz harmonic components with sharp asymmetry
            spike_wave = 120.0 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi))
            harmonics = 60.0 * np.sin(2 * np.pi * 6.0 * t) + 40.0 * np.sin(2 * np.pi * 9.0 * t)
            # High-frequency fast activity burst
            fast_activity = 35.0 * np.sin(2 * np.pi * 18.0 * t)
            sig = sig * 0.5 + spike_wave + harmonics + fast_activity

        channels_data.append(sig.astype(np.float32))

    if num_channels == 1:
        return channels_data[0]
    return np.stack(channels_data, axis=0)


def create_mock_bonn_dataset(output_dir: str, seed: int = 42, segments_per_set: int = 20):
    """
    Creates a sample Bonn EEG benchmark directory structure with text files.
    Sets A, B, C, D (non-seizure) and Set E (seizure).
    """
    rng = np.random.default_rng(seed)
    fs = 173.61
    duration = 4096 / fs  # ~23.59 seconds

    os.makedirs(output_dir, exist_ok=True)
    sets_info = [
        ("A", False),  # Normal eyes open
        ("B", False),  # Normal eyes closed
        ("C", False),  # Interictal hippocampal
        ("D", False),  # Interictal epileptogenic zone
        ("E", True),   # Ictal (seizure)
    ]

    for set_name, is_seizure in sets_info:
        set_dir = os.path.join(output_dir, set_name)
        os.makedirs(set_dir, exist_ok=True)
        for i in range(1, segments_per_set + 1):
            file_name = f"{set_name}{i:03d}.txt"
            file_path = os.path.join(set_dir, file_name)
            sig = generate_synthetic_eeg_window(duration, fs, is_seizure, rng, num_channels=1)
            # Ensure exactly 4096 samples
            if len(sig) > 4096:
                sig = sig[:4096]
            elif len(sig) < 4096:
                sig = np.pad(sig, (0, 4096 - len(sig)))
            np.savetxt(file_path, sig, fmt="%.2f")


def write_minimal_mock_edf(
    file_path: str,
    channel_names: List[str],
    signals: List[np.ndarray],
    fs: float = 256.0,
):
    """Writes a standard compliant EDF file from given numpy signal arrays."""
    num_signals = len(channel_names)
    n_samples = len(signals[0])
    record_duration = 1.0  # 1 second records
    samples_per_record = int(fs * record_duration)
    num_records = n_samples // samples_per_record

    with open(file_path, "wb") as f:
        # 256 bytes main header
        f.write(b"0       ")  # version (8)
        f.write(f"{'PATIENT_MOCK':<80}".encode("ascii"))  # patient id (80)
        f.write(f"{'RECORDING_MOCK':<80}".encode("ascii"))  # recording id (80)
        f.write(b"01.01.26")  # start date (8)
        f.write(b"12.00.00")  # start time (8)
        header_bytes = 256 + 256 * num_signals
        f.write(f"{header_bytes:<8}".encode("ascii"))  # header bytes (8)
        f.write(b" " * 44)  # reserved (44)
        f.write(f"{num_records:<8}".encode("ascii"))  # num records (8)
        f.write(f"{record_duration:<8.1f}".encode("ascii"))  # duration (8)
        f.write(f"{num_signals:<4}".encode("ascii"))  # num signals (4)

        # Signal headers
        for name in channel_names:
            f.write(f"{name:<16}".encode("ascii"))
        for _ in range(num_signals):
            f.write(f"{'AgAgCl electrode':<80}".encode("ascii"))
        for _ in range(num_signals):
            f.write(f"{'uV':<8}".encode("ascii"))
        for _ in range(num_signals):
            f.write(f"{-1000.0:<8.1f}".encode("ascii"))  # phys min
        for _ in range(num_signals):
            f.write(f"{1000.0:<8.1f}".encode("ascii"))   # phys max
        for _ in range(num_signals):
            f.write(f"{-32768:<8}".encode("ascii"))     # dig min
        for _ in range(num_signals):
            f.write(f"{32767:<8}".encode("ascii"))      # dig max
        for _ in range(num_signals):
            f.write(f"{'None':<80}".encode("ascii"))
        for _ in range(num_signals):
            f.write(f"{samples_per_record:<8}".encode("ascii"))
        for _ in range(num_signals):
            f.write(b" " * 32)

        # Write data record by record
        scale = 65535.0 / 2000.0
        for rec in range(num_records):
            for ch in range(num_signals):
                chunk = signals[ch][rec * samples_per_record : (rec + 1) * samples_per_record]
                dig_chunk = np.clip(chunk * scale, -32768, 32767).astype(np.int16)
                f.write(dig_chunk.tobytes())


def create_mock_chbmit_dataset(
    output_dir: str,
    patient_ids: List[str] = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06"],
    seed: int = 100,
):
    """
    Creates mock CHB-MIT clinical database with realistic seizure intervals and summaries.
    """
    rng = np.random.default_rng(seed)
    fs = 256.0
    channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1"]
    total_sec = 120.0  # 2 minutes per mock recording

    os.makedirs(output_dir, exist_ok=True)
    for p_idx, pid in enumerate(patient_ids):
        p_dir = os.path.join(output_dir, pid)
        os.makedirs(p_dir, exist_ok=True)

        summary_path = os.path.join(p_dir, f"{pid}-summary.txt")
        edf_name = f"{pid}_01.edf"
        edf_path = os.path.join(p_dir, edf_name)

        # Create one seizure interval between 40s and 70s
        seizure_start = 40.0
        seizure_end = 70.0

        # Generate signals for each channel
        n_samples = int(total_sec * fs)
        t = np.linspace(0, total_sec, n_samples, endpoint=False)
        sig_list = []
        for ch in channels:
            # Baseline signal
            base = generate_synthetic_eeg_window(total_sec, fs, is_seizure=False, rng=rng)
            # Inject seizure
            s_mask = (t >= seizure_start) & (t <= seizure_end)
            ictal = generate_synthetic_eeg_window(total_sec, fs, is_seizure=True, rng=rng)
            base[s_mask] = ictal[s_mask]
            sig_list.append(base)

        write_minimal_mock_edf(edf_path, channels, sig_list, fs=fs)

        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(f"Data File Name: {edf_name}\n")
            sf.write(f"File Name: {edf_name}\n")
            sf.write("Number of Seizures in File: 1\n")
            sf.write(f"Seizure 1 Start Time: {int(seizure_start)} seconds\n")
            sf.write(f"Seizure 1 End Time: {int(seizure_end)} seconds\n")

