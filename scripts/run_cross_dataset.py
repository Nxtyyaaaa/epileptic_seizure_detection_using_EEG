import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.cross_dataset_pipeline import run_cross_dataset_evaluation
from src.utils.config import load_config
from src.utils.logging_util import setup_logger

logger = setup_logger("RunCrossDataset")


def main():
    config_path = "configs/siena_config.yaml"
    cfg = load_config(config_path) if os.path.exists(config_path) else {}

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    df_cross = run_cross_dataset_evaluation(
        chbmit_dir="data/raw/chbmit",
        siena_dir="data/raw/siena",
        model_name=cfg.get("evaluation", {}).get("model", "random_forest"),
        balancing_strategy=cfg.get("evaluation", {}).get("best_strategy", "E3"),
        harmonized_fs=cfg.get("dataset", {}).get("target_fs", 256.0),
    )

    csv_path = os.path.join(output_dir, "cross_dataset_results.csv")
    df_cross.to_csv(csv_path, index=False)
    logger.info(f"Cross-dataset results saved to {csv_path}")

    print("\n=== Track C: Cross-Dataset Generalization Summary Table ===")
    display_cols = ["evaluation_type", "model", "strategy", "sensitivity", "specificity", "precision", "f1", "pr_auc", "accuracy"]
    print(df_cross[[c for c in display_cols if c in df_cross.columns]].to_string(index=False))


if __name__ == "__main__":
    main()

