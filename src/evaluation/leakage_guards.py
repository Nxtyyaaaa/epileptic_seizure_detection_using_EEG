import numpy as np
from typing import List, Set, Any


class LeakageGuardError(Exception):
    """Raised when data leakage between training, validation, or testing partitions is detected."""
    pass


class LeakageGuard:
    """
    Enforces strict experimental design integrity as mandated in Section 3.7 of the project report.
    """

    @staticmethod
    def verify_patient_disjoint(
        train_patients: List[str],
        val_patients: List[str],
        test_patients: List[str],
    ):
        """Ensures that no patient ID appears in more than one partition."""
        s_train = set(train_patients)
        s_val = set(val_patients)
        s_test = set(test_patients)

        train_val = s_train.intersection(s_val)
        train_test = s_train.intersection(s_test)
        val_test = s_val.intersection(s_test)

        if train_val:
            raise LeakageGuardError(f"Data leakage detected! Patients present in both TRAIN and VAL: {train_val}")
        if train_test:
            raise LeakageGuardError(f"Data leakage detected! Patients present in both TRAIN and TEST: {train_test}")
        if val_test:
            raise LeakageGuardError(f"Data leakage detected! Patients present in both VAL and TEST: {val_test}")

    @staticmethod
    def verify_no_resampling_on_evaluation_sets(
        y_original: np.ndarray,
        y_evaluated: np.ndarray,
        set_name: str = "Test",
    ):
        """
        Confirms that test/validation sets have NOT been resampled or synthetically generated.
        """
        if len(y_original) != len(y_evaluated):
            raise LeakageGuardError(
                f"Data leakage violation! {set_name} set sample count changed from {len(y_original)} to {len(y_evaluated)}. "
                "Balancing must NEVER be applied to validation or test data."
            )
        if not np.array_equal(y_original, y_evaluated):
            raise LeakageGuardError(
                f"Data leakage violation! {set_name} target distribution was altered."
            )

