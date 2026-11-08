# EEG Seizure Detection: Experimental Results Summary Report
**Project:** A Comprehensive Comparative Study of EEG Seizure Detection
**Evaluation Design:** Leakage-Safe Patient-Level Partitioning & Class Imbalance Mitigation

---

## 1. Track A: Bonn EEG Benchmark (Controlled Segment-Level)
Binary Task: Set E (Ictal) vs Sets A-D (Non-Ictal).

| track | model | strategy | train_samples | test_samples | sensitivity | specificity | precision | f1 | macro_f1 | accuracy | roc_auc | pr_auc | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Track A (Bonn) | svm | E0 | 75 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | svm | E1 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | svm | E2 | 30 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | svm | E3 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | svm | E4 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | svm | E5 | 75 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | svm | E6 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E0 | 75 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E1 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E2 | 30 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E3 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E4 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E5 | 75 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |
| Track A (Bonn) | random_forest | E6 | 120 | 25 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 5 | 20 | 0 | 0 |

---

## 2. Track B: CHB-MIT Patient-Independent Evaluation
Evaluated on held-out patients to prevent subject identity leakage.

| track | model | strategy | train_windows | test_windows | sensitivity | specificity | precision | f1 | macro_f1 | accuracy | roc_auc | pr_auc | tp | tn | fp | fn | event_sensitivity | false_alarms_per_hour | total_events_detected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Track B (CHB-MIT) | svm | E0 | 48 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | svm | E1 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | svm | E2 | 24 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | svm | E3 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | svm | E4 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | svm | E5 | 48 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | svm | E6 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E0 | 48 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E1 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E2 | 24 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E3 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E4 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E5 | 48 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |
| Track B (CHB-MIT) | random_forest | E6 | 72 | 24 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6 | 18 | 0 | 0 | 1.0 | 0.0 | 2 |

---

## 3. Track C: Cross-Dataset External Generalization (CHB-MIT -> Siena)
Zero-shot transfer without tuning decision thresholds on target cohort.

| evaluation_type | model | strategy | sensitivity | specificity | precision | f1 | macro_f1 | accuracy | roc_auc | pr_auc | tp | tn | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Internal Source (CHB-MIT) | random_forest | E3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 12 | 36 | 0 | 0 |
| External Target (Siena) | random_forest | E3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 4 | 14 | 0 | 0 |
