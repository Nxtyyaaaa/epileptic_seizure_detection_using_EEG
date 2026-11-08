import sys
import argparse
import os

from src.utils.logging_util import setup_logger

logger = setup_logger("EEGSeizureMain")


def main():
    parser = argparse.ArgumentParser(
        description="A Comprehensive Comparative Study of EEG Seizure Detection (Bonn, CHB-MIT, Siena)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["bonn", "chbmit", "cross", "all"],
        default="all",
        help=(
            "Experimental Track to run:\n"
            "  bonn   : Track A (Controlled benchmark on Bonn sets A-E, comparing E0-E6)\n"
            "  chbmit : Track B (Clinical patient-grouped evaluation on CHB-MIT)\n"
            "  cross  : Track C (Cross-dataset generalization CHB-MIT -> Siena)\n"
            "  all    : Execute all tracks and generate comprehensive report\n"
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test suite and verify environment integrity",
    )

    args = parser.parse_args()

    if args.test:
        from run_tests import run_all_tests
        sys.exit(run_all_tests())

    track = args.track.lower()

    if track in ["bonn", "all"]:
        logger.info("\n>>> EXECUTING TRACK A: BONN BENCHMARK <<<\n")
        from scripts.run_bonn_benchmark import main as run_bonn
        run_bonn()

    if track in ["chbmit", "all"]:
        logger.info("\n>>> EXECUTING TRACK B: CHB-MIT PATIENT EVALUATION <<<\n")
        from scripts.run_chbmit_experiment import main as run_chbmit
        run_chbmit()

    if track in ["cross", "all"]:
        logger.info("\n>>> EXECUTING TRACK C: CROSS-DATASET EVALUATION <<<\n")
        from scripts.run_cross_dataset import main as run_cross
        run_cross()

    if track == "all":
        logger.info("\n>>> COMPILING FINAL SUMMARY REPORT <<<\n")
        from scripts.generate_project_report import generate_summary_report
        generate_summary_report()
        logger.info("Project execution finished. Check 'reports/' for tables and figures.")


if __name__ == "__main__":
    main()

