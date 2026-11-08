import sys
import unittest
import numpy as np

# Import tests
from tests.test_features import (
    test_time_domain_features,
    test_frequency_domain_features,
    test_wavelet_features,
    test_hjorth_parameters,
    test_unified_feature_extractor,
)
from tests.test_balancing import (
    test_balancing_strategies,
    test_signal_augmentation_e6,
)
from tests.test_leakage_guards import (
    test_leakage_guard_patient_overlap,
    test_leakage_guard_disjoint_success,
    test_leakage_guard_resampling_on_test,
)
from tests.test_pipelines import (
    test_bonn_pipeline_end_to_end,
    test_chbmit_pipeline_end_to_end,
)
import tempfile
import pathlib


def run_all_tests():
    print("Running Project Test Suite...\n")
    tests = [
        ("Time Domain Features", test_time_domain_features),
        ("Frequency Domain Features", test_frequency_domain_features),
        ("Wavelet Features", test_wavelet_features),
        ("Hjorth Parameters", test_hjorth_parameters),
        ("Unified Feature Extractor", test_unified_feature_extractor),
        ("Balancing Strategies (E0-E5)", test_balancing_strategies),
        ("Signal Augmentation (E6)", test_signal_augmentation_e6),
        ("Leakage Guard (Overlap)", test_leakage_guard_patient_overlap),
        ("Leakage Guard (Disjoint)", test_leakage_guard_disjoint_success),
        ("Leakage Guard (Resampling on Test)", test_leakage_guard_resampling_on_test),
    ]

    passed = 0
    for name, func in tests:
        try:
            func()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {name}: {e}")

    # Temporary directory for pipeline tests
    with tempfile.TemporaryDirectory() as tmp_d:
        p = pathlib.Path(tmp_d)
        try:
            test_bonn_pipeline_end_to_end(p)
            print(" [PASS] Bonn Pipeline End-to-End")
            passed += 1
        except Exception as e:
            print(f" [FAIL] Bonn Pipeline End-to-End: {e}")

        try:
            test_chbmit_pipeline_end_to_end(p)
            print(" [PASS] CHB-MIT Pipeline End-to-End")
            passed += 1
        except Exception as e:
            print(f" [FAIL] CHB-MIT Pipeline End-to-End: {e}")

    total = len(tests) + 2
    print(f"\nTest Summary: {passed}/{total} Passed.")
    if passed == total:
        print("ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

