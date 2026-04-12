from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from misum_repro.utils import load_yaml, save_json


EXPERIMENTS = [
    {"name": "arm_o1_full", "config": "configs/full_runs/arm_o1_full.yaml", "arch": "ARM", "opt": "O1"},
    {"name": "arm_o2_full", "config": "configs/full_runs/arm_o2_full.yaml", "arch": "ARM", "opt": "O2"},
    {"name": "arm_o3_full", "config": "configs/full_runs/arm_o3_full.yaml", "arch": "ARM", "opt": "O3"},
    {"name": "x86_o1_full", "config": "configs/full_runs/x86_o1_full.yaml", "arch": "X86", "opt": "O1"},
    {"name": "x86_o2_full", "config": "configs/full_runs/x86_o2_full.yaml", "arch": "X86", "opt": "O2"},
    {"name": "x86_o3_full", "config": "configs/full_runs/x86_o3_full.yaml", "arch": "X86", "opt": "O3"},
    {"name": "x64_o1_full", "config": "configs/full_runs/x64_o1_full.yaml", "arch": "X64", "opt": "O1"},
    {"name": "x64_o2_full", "config": "configs/full_runs/x64_o2_full.yaml", "arch": "X64", "opt": "O2"},
    {"name": "x64_o3_full", "config": "configs/full_runs/x64_o3_full.yaml", "arch": "X64", "opt": "O3"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MiSum experiment matrix.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[],
        help="Optional subset of experiment names to run. Defaults to all.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiments that would be executed without starting training.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one experiment fails.",
    )
    return parser.parse_args()


def select_experiments(names: list[str]) -> list[dict[str, str]]:
    if not names:
        return list(EXPERIMENTS)
    wanted = set(names)
    selected = [experiment for experiment in EXPERIMENTS if experiment["name"] in wanted]
    missing = wanted - {experiment["name"] for experiment in selected}
    if missing:
        raise SystemExit(f"Unknown experiment names: {', '.join(sorted(missing))}")
    return selected


def load_metrics(run_name: str) -> dict[str, Any] | None:
    metrics_path = Path("artifacts") / run_name / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def is_completed_run(run_name: str) -> bool:
    metrics = load_metrics(run_name)
    if metrics is None:
        return False
    return "best_valid" in metrics and "test" in metrics


def write_summary(selected: list[dict[str, str]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    for experiment in selected:
        metrics = load_metrics(experiment["name"])
        if metrics is None:
            summary_rows.append(
                {
                    "name": experiment["name"],
                    "arch": experiment["arch"],
                    "opt": experiment["opt"],
                    "status": "missing",
                }
            )
            continue

        if "best_valid" not in metrics or "test" not in metrics:
            summary_rows.append(
                {
                    "name": experiment["name"],
                    "arch": experiment["arch"],
                    "opt": experiment["opt"],
                    "status": "partial",
                }
            )
            continue

        best_valid = metrics.get("best_valid", {})
        test = metrics.get("test", {})
        summary_rows.append(
            {
                "name": experiment["name"],
                "arch": experiment["arch"],
                "opt": experiment["opt"],
                "status": "completed",
                "device": metrics.get("device"),
                "amp": metrics.get("amp"),
                "train_size": metrics.get("dataset_sizes", {}).get("train"),
                "valid_size": metrics.get("dataset_sizes", {}).get("valid"),
                "test_size": metrics.get("dataset_sizes", {}).get("test"),
                "valid_bleu4": best_valid.get("bleu4"),
                "valid_meteor": best_valid.get("meteor"),
                "valid_rouge_l": best_valid.get("rouge_l"),
                "test_bleu4": test.get("bleu4"),
                "test_meteor": test.get("meteor"),
                "test_rouge_l": test.get("rouge_l"),
            }
        )

    summary_dir = Path("artifacts") / "experiment_matrix"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary_dir / "summary.json", summary_rows)

    csv_path = summary_dir / "summary.csv"
    fieldnames = [
        "name",
        "arch",
        "opt",
        "status",
        "device",
        "amp",
        "train_size",
        "valid_size",
        "test_size",
        "valid_bleu4",
        "valid_meteor",
        "valid_rouge_l",
        "test_bleu4",
        "test_meteor",
        "test_rouge_l",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    selected = select_experiments(args.experiments)

    for experiment in selected:
        config_path = Path(experiment["config"])
        config = load_yaml(config_path)
        metrics_path = Path("artifacts") / config["run_name"] / "metrics.json"
        command = [sys.executable, "train.py", "--config", str(config_path)]

        print(f"[experiment] {experiment['name']} -> {' '.join(command)}")
        if args.skip_existing and is_completed_run(config["run_name"]):
            print(f"[skip] completed metrics found at {metrics_path}")
            continue
        if args.dry_run:
            continue

        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"[failed] {experiment['name']} exited with code {result.returncode}")
            if args.stop_on_error:
                break

    write_summary(selected)
    print("[summary] wrote artifacts/experiment_matrix/summary.json and summary.csv")


if __name__ == "__main__":
    main()
