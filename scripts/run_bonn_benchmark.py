import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.bonn_pipeline import run_bonn_benchmark
from src.evaluation.visualizer import ResultVisualizer
from src.utils.config import load_config
from src.utils.logging_util import setup_logger

logger = setup_logger("RunBonnBenchmark")


def main():
    config_path = "configs/bonn_config.yaml"
    cfg = load_config(config_path) if os.path.exists(config_path) else {}

    output_dir = "reports"
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df_results = run_bonn_benchmark(
        data_dir=cfg.get("dataset", {}).get("data_dir", "data/raw/bonn"),
        models=cfg.get("experiments", {}).get("models", ["svm", "random_forest"]),
        strategies=cfg.get("experiments", {}).get("strategies", ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]),
        test_size=cfg.get("experiments", {}).get("test_size", 0.25),
        random_state=cfg.get("experiments", {}).get("random_state", 42),
    )

    csv_path = os.path.join(output_dir, "bonn_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Generate visual summary
    vis = ResultVisualizer(output_dir=figures_dir)
    rf_subset = df_results[df_results["model"] == "random_forest"]
    if not rf_subset.empty:
        vis.plot_balancing_comparison_bar(
            rf_subset,
            title="Bonn EEG Benchmark: Balancing Strategies (Random Forest)",
            filename="bonn_balancing_rf.png",
        )

    print("\n=== Track A: Bonn Benchmark Summary Table ===")
    display_cols = ["model", "strategy", "sensitivity", "specificity", "precision", "f1", "pr_auc", "accuracy"]
    print(df_results[[c for c in display_cols if c in df_results.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

