"""Post-process a TextAttack TextFooler run and compute USE similarity."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


def plain_text(text: str) -> str:
    """Remove TextAttack's CSV markers around changed words."""
    return text.replace("[[", "").replace("]]", "")


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, np.finfo(np.float32).eps)


def main() -> None:
    args = parse_args()
    with args.csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts = Counter(row["result_type"] for row in rows)
    successful = [row for row in rows if row["result_type"] == "Successful"]

    model_url = (
        "https://www.kaggle.com/models/google/universal-sentence-encoder/"
        "TensorFlow1/large/3"
    )
    model_path = args.model_path or (
        Path(__file__).resolve().parents[1] / "models" / "use-large-v3" / "model"
    )
    model = hub.load(str(model_path))
    embed = model.signatures["default"]
    similarities: list[float] = []
    for start in range(0, len(successful), args.batch_size):
        batch = successful[start : start + args.batch_size]
        originals = [plain_text(row["original_text"]) for row in batch]
        perturbed = [plain_text(row["perturbed_text"]) for row in batch]
        original_vectors = np.asarray(embed(text=tf.constant(originals))["default"])
        perturbed_vectors = np.asarray(embed(text=tf.constant(perturbed))["default"])
        similarities.extend(cosine_rows(original_vectors, perturbed_vectors).tolist())

    total = len(rows)
    succeeded = counts["Successful"]
    failed = counts["Failed"]
    skipped = counts["Skipped"]
    eligible = succeeded + failed
    similarity_array = np.asarray(similarities, dtype=np.float64)

    output = {
        "source_csv": str(args.csv_path.resolve()),
        "examples": total,
        "successful": succeeded,
        "failed": failed,
        "skipped": skipped,
        "derived_metrics": {
            "original_accuracy_percent": 100.0 * eligible / total,
            "accuracy_under_attack_percent": 100.0 * failed / total,
            "attack_success_rate_percent": 100.0 * succeeded / eligible,
        },
        "semantic_similarity": {
            "model": "Universal Sentence Encoder Large v3",
            "authors_url": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
            "resolved_mirror": model_url,
            "local_model_path": str(model_path.resolve()),
            "population": "successful attacks only",
            "n": int(similarity_array.size),
            "mean_cosine": float(np.mean(similarity_array)),
            "median_cosine": float(np.median(similarity_array)),
            "standard_deviation": float(np.std(similarity_array)),
            "minimum": float(np.min(similarity_array)),
            "maximum": float(np.max(similarity_array)),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
