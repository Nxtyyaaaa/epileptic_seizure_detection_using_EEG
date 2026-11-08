import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes all standard clinical evaluation metrics from the report:
    - Sensitivity (Recall)
    - Specificity
    - Precision
    - F1-Score & Macro-F1
    - PR-AUC (Area Under Precision-Recall Curve)
    - ROC-AUC
    - Accuracy
    - Confusion matrix components (TP, TN, FP, FN)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    roc_auc = 0.5
    pr_auc = 0.0

    if y_prob is not None:
        try:
            # If 2D probabilities, extract positive class
            p = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, p))
                prec_arr, rec_arr, _ = precision_recall_curve(y_true, p)
                pr_auc = float(auc(rec_arr, prec_arr))
        except Exception:
            pass

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

