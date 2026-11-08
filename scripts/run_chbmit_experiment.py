import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.chbmit_pipeline import run_chbmit_patient_experiment
from src.evaluation.visualizer import ResultVisualizer
from src.utils.config import load_config
from src.utils.logging_util import setup_logger

logger = setup_logger("RunCHBMITExperiment")


def main():
    config_path = "configs/chbmit_config.yaml"
    cfg = load_config(config_path) if os.path.exists(config_path) else {}

    output_dir = "reports"
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    pilot_pts = cfg.get("dataset", {}).get("pilot_patients", {})
    train_pts = pilot_pts.get("train", ["chb01", "chb02", "chb03", "chb04"])
    test_pts = pilot_pts.get("test", ["chb05", "chb06"])

    df_results, timeline_info = run_chbmit_patient_experiment(
        data_dir=cfg.get("dataset", {}).get("data_dir", "data/raw/chbmit"),
        train_patients=train_pts,
        test_patients=test_pts,
        models=cfg.get("experiments", {}).get("models", ["svm", "random_forest"]),
        strategies=cfg.get("experiments", {}).get("strategies", ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]),
        window_sec=cfg.get("windowing", {}).get("window_sec", 10.0),
        random_state=cfg.get("experiments", {}).get("random_state", 42),
    )

    csv_path = os.path.join(output_dir, "chbmit_patient_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"CHB-MIT results saved to {csv_path}")

    vis = ResultVisualizer(output_dir=figures_dir)
    rf_subset = df_results[df_results["model"] == "random_forest"]
    if not rf_subset.empty:
        vis.plot_balancing_comparison_bar(
            rf_subset,
            metrics=["sensitivity", "precision", "f1", "false_alarms_per_hour"],
            title="CHB-MIT Patient-Independent: Balancing Strategies (Random Forest)",
            filename="chbmit_balancing_rf.png",
        )

    print("\n=== Track B: CHB-MIT Patient-Independent Summary Table ===")
    display_cols = ["model", "strategy", "sensitivity", "precision", "f1", "false_alarms_per_hour", "total_events_detected"]
    print(df_results[[c for c in display_cols if c in df_results.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

