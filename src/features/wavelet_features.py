import numpy as np
from typing import Dict

# Daubechies 4 (db4) low-pass (scaling) and high-pass (wavelet) filter coefficients
DB4_H = np.array([
    (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
    (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
    (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
    (1 - np.sqrt(3)) / (4 * np.sqrt(2)),
], dtype=np.float32)
DB4_G = np.array([DB4_H[3], -DB4_H[2], DB4_H[1], -DB4_H[0]], dtype=np.float32)


def _dwt_single_level_numpy(x: np.ndarray) -> (np.ndarray, np.ndarray):
    """Computes single-level DWT using db4 filters and periodic padding."""
    cA = np.convolve(x, DB4_H, mode="same")[::2]
    cD = np.convolve(x, DB4_G, mode="same")[::2]
    return cA, cD


def extract_wavelet_features(x: np.ndarray, level: int = 4) -> Dict[str, float]:
    """
    Computes Discrete Wavelet Transform (DWT) subband energies and relative energies.
    Uses pywt if installed; otherwise falls back to exact NumPy db4 filter cascade.
    Subbands: D1, D2, D3, D4, and A4.
    """
    feats: Dict[str, float] = {}

    try:
        import pywt
        coeffs = pywt.wavedec(x, "db4", level=level)
        # coeffs = [cA_level, cD_level, ..., cD_1]
        a_coeff = coeffs[0]
        d_coeffs = coeffs[1:]  # D_level down to D_1
    except ImportError:
        # Pure NumPy DWT implementation
        curr_sig = x
        d_coeffs_rev = []
        for _ in range(level):
            if len(curr_sig) < 8:
                break
            curr_sig, cD = _dwt_single_level_numpy(curr_sig)
            d_coeffs_rev.append(cD)
        a_coeff = curr_sig
        d_coeffs = d_coeffs_rev  # D_level down to D_1

    # Compute subband energies
    energies = []
    a_energy = float(np.sum(a_coeff ** 2))
    feats[f"wavelet_energy_A{level}"] = a_energy
    energies.append(a_energy)

    for i, cD in enumerate(d_coeffs):
        d_idx = level - i
        e = float(np.sum(cD ** 2))
        feats[f"wavelet_energy_D{d_idx}"] = e
        energies.append(e)

    total_energy = sum(energies)
    feats["wavelet_total_energy"] = total_energy

    # Relative energies
    feats[f"wavelet_rel_energy_A{level}"] = a_energy / total_energy if total_energy > 1e-12 else 0.0
    for i, cD in enumerate(d_coeffs):
        d_idx = level - i
        e = feats[f"wavelet_energy_D{d_idx}"]
        feats[f"wavelet_rel_energy_D{d_idx}"] = e / total_energy if total_energy > 1e-12 else 0.0

    return feats

