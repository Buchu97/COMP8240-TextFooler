"""Verify completed attack logs and measure full-text USE-Large cosine similarity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_type1 import attack_metrics
from type1_data import sha256, write_json
from type1_models import configure_cache, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(args.run_dir / "outcomes.csv")
    if summary["status"] != "completed" or len(rows) != summary["examples_planned"]:
        raise SystemExit("Run is incomplete; full-run analysis refused")
    for row in rows:
        for name in ("original_prediction", "perturbed_prediction", "num_queries",
                     "original_word_count", "changed_word_count"):
            row[name] = int(row[name])
    metrics = attack_metrics(rows)
    if metrics != summary["metrics"]:
        raise ValueError("Recomputed metrics differ from run summary")
    for filename, digest in summary["artifacts_sha256"].items():
        if sha256(args.run_dir / filename) != digest:
            raise ValueError(f"Artifact hash mismatch: {filename}")
    output = args.run_dir / "semantic_similarity.json"
    if output.exists() or (args.run_dir / "semantic_similarity.csv").exists():
        raise SystemExit("Semantic output already exists")
    successful = [r for r in rows if r["result_type"] == "Successful"]
    similarities = []
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models/use-large-v3/model"
    if successful:
        configure_cache(root)
        import numpy as np
        import tensorflow as tf
        import tensorflow_hub as hub
        from analyze_core_replication import cosine_rows
        tf.config.set_visible_devices([], "GPU")
        model = hub.load(str(model_path))
        embed = model.signatures["default"]
        for start in range(0, len(successful), args.batch_size):
            batch = successful[start:start + args.batch_size]
            left = np.asarray(embed(text=tf.constant([r["original_text"] for r in batch]))["default"])
            right = np.asarray(embed(text=tf.constant([r["perturbed_text"] for r in batch]))["default"])
            similarities.extend(cosine_rows(left, right).tolist())
    with (args.run_dir / "semantic_similarity.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "source_id", "cosine_similarity"])
        writer.writeheader()
        writer.writerows({"row": r["row"], "source_id": r["source_id"], "cosine_similarity": score}
                         for r, score in zip(successful, similarities))
    result = {"dataset": summary["dataset"], "model": "Universal Sentence Encoder Large v3",
        "authors_url": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
        "saved_model_sha256": sha256(model_path / "saved_model.pb") if successful else None,
        "metric": "Full original/adversarial text cosine similarity; not the attack's windowed angular constraint",
        "population": "Successful attacks only", "n": len(similarities),
        "mean_cosine": sum(similarities) / len(similarities) if similarities else None,
        "minimum": min(similarities) if similarities else None,
        "maximum": max(similarities) if similarities else None,
        "semantic_or_task_label_preservation_verified": False,
        "recomputed_attack_metrics_match": True, "source_outcomes_sha256": sha256(args.run_dir / "outcomes.csv")}
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "saved_model_sha256"}), flush=True)


if __name__ == "__main__":
    main()
