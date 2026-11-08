import numpy as np
from scipy import signal
from typing import Dict, Tuple

# Standard physiological EEG sub-bands
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


def extract_frequency_domain_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Computes Welch Power Spectral Density (PSD), band powers, relative powers,
    spectral entropy, and band power ratios for an EEG window.
    """
    feats: Dict[str, float] = {}

    # Welch PSD computation
    nperseg = min(len(x), int(2.0 * fs)) if len(x) >= int(fs) else len(x)
    if nperseg < 4:
        # Default zeros
        for band in EEG_BANDS:
            feats[f"bp_{band}"] = 0.0
            feats[f"rel_bp_{band}"] = 0.0
        feats["total_power"] = 0.0
        feats["spectral_entropy"] = 0.0
        feats["ratio_theta_alpha_to_beta_gamma"] = 0.0
        feats["ratio_delta_to_theta"] = 0.0
        return feats

    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    total_power = float(np.sum(psd) * df)
    feats["total_power"] = total_power

    # Compute subband powers
    for band_name, (low, high) in EEG_BANDS.items():
        band_mask = (freqs >= low) & (freqs < high)
        if np.any(band_mask):
            bp = float(np.sum(psd[band_mask]) * df)
        else:
            bp = 0.0
        feats[f"bp_{band_name}"] = bp
        feats[f"rel_bp_{band_name}"] = bp / total_power if total_power > 1e-12 else 0.0

    # Spectral Entropy
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = -float(np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0.0
    feats["spectral_entropy"] = spectral_entropy

    # Clinical ratios frequently elevated during seizure slow-wave discharge
    theta_alpha = feats["bp_theta"] + feats["bp_alpha"]
    beta_gamma = feats["bp_beta"] + feats["bp_gamma"]
    feats["ratio_theta_alpha_to_beta_gamma"] = theta_alpha / beta_gamma if beta_gamma > 1e-8 else 0.0
    feats["ratio_delta_to_theta"] = feats["bp_delta"] / feats["bp_theta"] if feats["bp_theta"] > 1e-8 else 0.0

    return feats

