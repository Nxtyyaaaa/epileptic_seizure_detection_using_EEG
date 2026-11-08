import os
import shutil
from src.pipelines.bonn_pipeline import run_bonn_benchmark
from src.pipelines.chbmit_pipeline import run_chbmit_patient_experiment
from src.pipelines.cross_dataset_pipeline import run_cross_dataset_evaluation


def test_bonn_pipeline_end_to_end(tmp_path):
    test_dir = str(tmp_path / "bonn_test")
    df = run_bonn_benchmark(
        data_dir=test_dir,
        models=["random_forest"],
        strategies=["E0", "E1", "E3"],
        test_size=0.3,
        auto_generate_mock=True,
    )
    assert not df.empty
    assert "sensitivity" in df.columns
    assert "pr_auc" in df.columns
    assert len(df) == 3


def test_chbmit_pipeline_end_to_end(tmp_path):
    test_dir = str(tmp_path / "chbmit_test")
    df, tl = run_chbmit_patient_experiment(
        data_dir=test_dir,
        train_patients=["chb01", "chb02"],
        test_patients=["chb03"],
        models=["random_forest"],
        strategies=["E0", "E3"],
        auto_generate_mock=True,
    )
    assert not df.empty
    assert "false_alarms_per_hour" in df.columns
    assert "event_sensitivity" in df.columns
