from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TableCell:
    model_family: str
    dataset: str
    model_alias: str | None
    supported: bool
    limitation: str | None = None

    @property
    def cell_id(self) -> str:
        return f"{self.model_family}_{self.dataset}"


def build_matrix() -> list[TableCell]:
    aliases = {
        "wordcnn": {
            "mr": "cnn-mr",
            "imdb": "cnn-imdb",
            "yelp": "cnn-yelp",
            "ag": "cnn-ag-news",
        },
        "wordlstm": {
            "mr": "lstm-mr",
            "imdb": "lstm-imdb",
            "yelp": "lstm-yelp",
            "ag": "lstm-ag-news",
        },
        "bert": {
            "mr": "bert-base-uncased-mr",
            "imdb": "bert-base-uncased-imdb",
            "yelp": "bert-base-uncased-yelp",
            "ag": "bert-base-uncased-ag-news",
        },
    }
    cells: list[TableCell] = []
    for model_family in ("wordcnn", "wordlstm", "bert"):
        for dataset in ("mr", "imdb", "yelp", "ag", "fake"):
            model_alias = aliases[model_family].get(dataset)
            cells.append(
                TableCell(
                    model_family=model_family,
                    dataset=dataset,
                    model_alias=model_alias,
                    supported=model_alias is not None,
                    limitation=(
                        None
                        if model_alias is not None
                        else "No maintained TextAttack checkpoint/dataset alias for Fake News."
                    ),
                )
            )
    return cells


def build_attack_command(
    python_executable: Path,
    attack_file: Path,
    cell: TableCell,
    output_dir: Path,
    num_examples: int,
    random_seed: int,
) -> list[str]:
    if not cell.supported or cell.model_alias is None:
        raise ValueError(f"Unsupported table cell: {cell.cell_id}")
    command = [
        str(python_executable),
        "-m",
        "textattack",
        "attack",
        "--attack-from-file",
        f"{attack_file}^Attack",
        "--model",
        cell.model_alias,
        "--num-examples",
        str(num_examples),
        "--shuffle",
        "--random-seed",
        str(random_seed),
        "--log-to-txt",
        str(output_dir / "attack_results.txt"),
        "--log-to-csv",
        str(output_dir / "attack_results.csv"),
        "--log-summary-to-json",
        str(output_dir / "attack_summary.json"),
    ]
    if num_examples >= 50:
        command.extend(
            [
                "--checkpoint-interval",
                "50",
                "--checkpoint-dir",
                str(output_dir / "checkpoints"),
            ]
        )
    return command


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=8240)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "main_table_smoke",
    )
    parser.add_argument("--only", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.num_examples <= 1000:
        raise SystemExit("--num-examples must be between 1 and 1000")

    project_root = Path(__file__).resolve().parents[1]
    python_executable = project_root / ".venv" / "Scripts" / "python.exe"
    attack_file = project_root / "scripts" / "textfooler_paper_delta_0_7.py"
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    )
    output_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.only)
    matrix = build_matrix()
    known_ids = {cell.cell_id for cell in matrix}
    unknown_ids = selected - known_ids
    if unknown_ids:
        raise SystemExit(f"Unknown table cells: {', '.join(sorted(unknown_ids))}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_examples_per_cell": args.num_examples,
        "random_seed": args.random_seed,
        "paper_word_similarity_delta": 0.7,
        "cells": [],
    }
    for cell in matrix:
        record = asdict(cell) | {"cell_id": cell.cell_id, "status": "not_selected"}
        if not cell.supported:
            record["status"] = "blocked_missing_checkpoint"
        payload["cells"].append(record)
    status_path = output_root / "matrix_status.json"
    write_status(status_path, payload)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "TA_CACHE_DIR": str(project_root / "data" / "cache" / "textattack"),
            "HF_HOME": str(project_root / "data" / "cache" / "huggingface"),
            "TORCH_HOME": str(project_root / "data" / "cache" / "torch"),
            "NLTK_DATA": str(project_root / "data" / "cache" / "nltk"),
            "TFHUB_CACHE_DIR": str(
                project_root / "data" / "cache" / "tensorflow_hub"
            ),
            "TF_CPP_MIN_LOG_LEVEL": "3",
        }
    )

    failures = 0
    records_by_id = {record["cell_id"]: record for record in payload["cells"]}
    for cell in matrix:
        if not cell.supported or (selected and cell.cell_id not in selected):
            continue
        record = records_by_id[cell.cell_id]
        output_dir = output_root / cell.cell_id
        if output_dir.exists() and any(output_dir.iterdir()):
            record["status"] = "existing_output"
            write_status(status_path, payload)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_attack_command(
            python_executable=python_executable,
            attack_file=attack_file,
            cell=cell,
            output_dir=output_dir,
            num_examples=args.num_examples,
            random_seed=args.random_seed,
        )
        record["status"] = "running"
        record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["command"] = command
        write_status(status_path, payload)

        completed = subprocess.run(command, cwd=project_root, env=environment)
        record["return_code"] = completed.returncode
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        failures += completed.returncode != 0
        write_status(status_path, payload)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
