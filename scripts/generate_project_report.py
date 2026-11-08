import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Formats a pandas DataFrame into a clean Markdown table without tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(
            f"{val:.4f}" if isinstance(val, (float, int)) and not isinstance(val, bool) and abs(val - int(val)) > 1e-6 else str(val)
            for val in row.values
        ) + " |"
        rows.append(row_str)
    return "\n".join([header, divider] + rows)


def generate_summary_report():
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    bonn_path = os.path.join(reports_dir, "bonn_benchmark_results.csv")
    chb_path = os.path.join(reports_dir, "chbmit_patient_results.csv")
    cross_path = os.path.join(reports_dir, "cross_dataset_results.csv")

    report_lines = [
        "# EEG Seizure Detection: Experimental Results Summary Report",
        "**Project:** A Comprehensive Comparative Study of EEG Seizure Detection",
        "**Evaluation Design:** Leakage-Safe Patient-Level Partitioning & Class Imbalance Mitigation",
        "",
        "---",
        "",
        "## 1. Track A: Bonn EEG Benchmark (Controlled Segment-Level)",
        "Binary Task: Set E (Ictal) vs Sets A-D (Non-Ictal).",
        "",
    ]

    if os.path.exists(bonn_path):
        df_bonn = pd.read_csv(bonn_path)
        report_lines.append(df_to_markdown_simple(df_bonn))
    else:
        report_lines.append("*Bonn benchmark not yet executed. Run `scripts/run_bonn_benchmark.py`.*")

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Track B: CHB-MIT Patient-Independent Evaluation",
        "Evaluated on held-out patients to prevent subject identity leakage.",
        "",
    ])

    if os.path.exists(chb_path):
        df_chb = pd.read_csv(chb_path)
        report_lines.append(df_to_markdown_simple(df_chb))
    else:
        report_lines.append("*CHB-MIT experiment not yet executed. Run `scripts/run_chbmit_experiment.py`.*")

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Track C: Cross-Dataset External Generalization (CHB-MIT -> Siena)",
        "Zero-shot transfer without tuning decision thresholds on target cohort.",
        "",
    ])

    if os.path.exists(cross_path):
        df_cross = pd.read_csv(cross_path)
        report_lines.append(df_to_markdown_simple(df_cross))
    else:
        report_lines.append("*Cross-dataset experiment not yet executed. Run `scripts/run_cross_dataset.py`.*")

    summary_file = os.path.join(reports_dir, "PROJECT_EVALUATION_SUMMARY.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Generated comprehensive report at {summary_file}")


if __name__ == "__main__":
    generate_summary_report()
