#!/usr/bin/env python3
"""Validate a UR3e kinematics HDF5 dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from pathlib import Path
from ur3_mujoco.utils.config_loader import load_config
from ur3_mujoco.validation.kinematics_dataset_validator import KinematicsDatasetValidator
from ur3_mujoco.validation.report import write_report

BASE_DIR = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', default='configs/kinematics_debug.yaml')
    args = parser.parse_args()

    config = load_config(str(BASE_DIR / args.config))
    validator = KinematicsDatasetValidator(args.dataset, config)
    passed = validator.validate()

    report_path = str(BASE_DIR / "docs" / "VALIDATION_REPORT.md")
    write_report(args.dataset, validator.errors, validator.warnings, report_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
