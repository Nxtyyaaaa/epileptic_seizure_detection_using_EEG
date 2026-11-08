import numpy as np
from src.features.extractor import EEGFeatureExtractor
from src.features.time_domain import extract_time_domain_features
from src.features.frequency_domain import extract_frequency_domain_features
from src.features.wavelet_features import extract_wavelet_features
from src.features.nonlinear import compute_hjorth_parameters


def test_time_domain_features():
    x = np.sin(np.linspace(0, 10, 100))
    feats = extract_time_domain_features(x)
    assert "mean" in feats
    assert "rms" in feats
    assert "peak_to_peak" in feats
    assert not np.isnan(feats["rms"])


def test_frequency_domain_features():
    fs = 256.0
    t = np.linspace(0, 2.0, int(2 * fs), endpoint=False)
    x = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz alpha wave
    feats = extract_frequency_domain_features(x, fs=fs)
    assert "bp_alpha" in feats
    assert feats["bp_alpha"] > feats["bp_delta"]


def test_wavelet_features():
    x = np.random.randn(512).astype(np.float32)
    feats = extract_wavelet_features(x, level=4)
    assert "wavelet_energy_A4" in feats
    assert "wavelet_energy_D1" in feats
    assert feats["wavelet_total_energy"] > 0


def test_hjorth_parameters():
    x = np.random.randn(256)
    hjorth = compute_hjorth_parameters(x)
    assert "hjorth_activity" in hjorth
    assert "hjorth_mobility" in hjorth
    assert "hjorth_complexity" in hjorth


def test_unified_feature_extractor():
    fs = 173.61
    windows = np.random.randn(5, int(fs * 5)).astype(np.float32)
    extractor = EEGFeatureExtractor(fs=fs)
    X_feat, names = extractor.transform(windows)
    assert X_feat.shape[0] == 5
    assert X_feat.shape[1] == len(names)
    assert not np.isnan(X_feat).any()
