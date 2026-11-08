import numpy as np
from src.evaluation.leakage_guards import LeakageGuard, LeakageGuardError


def test_leakage_guard_patient_overlap():
    train_pts = ["chb01", "chb02", "chb03"]
    val_pts = ["chb04"]
    test_pts = ["chb03", "chb05"]  # chb03 in both train and test!

    threw = False
    try:
        LeakageGuard.verify_patient_disjoint(train_pts, val_pts, test_pts)
    except LeakageGuardError:
        threw = True
    assert threw, "LeakageGuard did not raise on overlapping patients"


def test_leakage_guard_disjoint_success():
    train_pts = ["chb01", "chb02"]
    val_pts = ["chb03"]
    test_pts = ["chb04", "chb05"]
    # Should not raise
    LeakageGuard.verify_patient_disjoint(train_pts, val_pts, test_pts)


def test_leakage_guard_resampling_on_test():
    y_orig = np.array([0, 0, 0, 1])
    y_altered = np.array([0, 0, 1, 1])  # resampled!

    threw = False
    try:
        LeakageGuard.verify_no_resampling_on_evaluation_sets(y_orig, y_altered, set_name="Test")
    except LeakageGuardError:
        threw = True
    assert threw, "LeakageGuard did not raise on test resampling alteration"

