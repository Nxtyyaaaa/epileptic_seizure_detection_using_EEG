from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator


def get_classifier(
    model_name: str,
    random_state: int = 42,
    class_weight: Optional[Any] = None,
    **kwargs: Any,
) -> BaseEstimator:
    """
    Instantiates standard baseline classifiers for EEG seizure detection.
    Supports dynamic class weighting for Experiment condition E5.
    """
    name = model_name.lower().replace("-", "_").replace(" ", "_")

    if name in ["svm", "svm_rbf", "svc"]:
        return SVC(
            kernel=kwargs.get("kernel", "rbf"),
            C=kwargs.get("C", 1.0),
            gamma=kwargs.get("gamma", "scale"),
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )
    elif name in ["svm_linear", "linear_svm"]:
        return SVC(
            kernel="linear",
            C=kwargs.get("C", 1.0),
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )
    elif name in ["random_forest", "rf"]:
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 15),
            min_samples_split=kwargs.get("min_samples_split", 2),
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    elif name in ["logistic_regression", "lr"]:
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=class_weight,
            random_state=random_state,
        )
    elif name in ["gradient_boosting", "gbm"]:
        return GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            max_depth=kwargs.get("max_depth", 3),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier model: {model_name}")

