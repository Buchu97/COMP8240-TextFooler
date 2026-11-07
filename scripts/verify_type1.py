"""Independently reconcile saved Type 1 examples, metrics, hashes and similarities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from type1_data import sha256, write_json
from type1_models import read_csv


def independent_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No result rows")
    counts = Counter()
    queries, perturbations = [], []
    for index, row in enumerate(rows):
        if int(row["row"]) != index:
            raise ValueError("Nonsequential outcome rows")
        label, original, perturbed = [int(row[key]) for key in
                                      ("label", "original_prediction", "perturbed_prediction")]
        if original != label:
            expected = "Skipped"
            if perturbed != original:
                raise ValueError("Skipped outcome changed prediction")
        else:
            expected = "Failed" if perturbed == label else "Successful"
        if row["result_type"] != expected:
            raise ValueError("Outcome disagrees with recorded predictions")
        counts[expected] += 1
        n_queries, words, changed = [int(row[key]) for key in
                                     ("num_queries", "original_word_count", "changed_word_count")]
        if n_queries < 1 or words < 1 or changed < 0 or changed > words:
            raise ValueError("Invalid query or changed-word count")
        if expected != "Skipped":
            queries.append(n_queries)
        if expected == "Successful":
            perturbations.append(100 * changed / words)
    eligible = counts["Successful"] + counts["Failed"]
    return {"examples": len(rows), "successful": counts["Successful"], "failed": counts["Failed"],
        "skipped": counts["Skipped"], "eligible": eligible,
        "original_accuracy_percent": 100 * eligible / len(rows),
        "after_attack_accuracy_percent": 100 * counts["Failed"] / len(rows),
        "attack_success_rate_percent": 100 * counts["Successful"] / eligible if eligible else None,
        "mean_queries_eligible": sum(queries) / len(queries) if queries else None,
        "mean_perturbed_word_percent_successful": sum(perturbations) / len(perturbations) if perturbations else None}


def verify_run(run_dir: Path, dataset_root: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["recipe_revision_tag"] != "PRECONSTRAINTS_RESTORED_20261008":
        raise ValueError("Incomplete run or uncorrected recipe")
    dataset_dir = dataset_root / summary["dataset"]
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in summary["artifacts_sha256"].items():
        if sha256(run_dir / name) != digest:
            raise ValueError(f"Run artifact changed: {name}")
    if sha256(dataset_dir / "evaluation.csv") != summary["evaluation_sha256"]:
        raise ValueError("Evaluation hash mismatch")
    if sha256(dataset_dir / "manifest.json") != summary["dataset_manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    source, metadata = read_csv(dataset_dir / "evaluation.csv"), read_csv(dataset_dir / "evaluation_metadata.csv")
    baseline, outcomes = read_csv(run_dir / "baseline.csv"), read_csv(run_dir / "outcomes.csv")
    raw = read_csv(run_dir / "attack_results.csv")
    if not len(source) == len(metadata) == len(baseline) == len(outcomes) == len(raw) == summary["examples_planned"]:
        raise ValueError("Source, baseline and attack counts differ")
    for index, (example, meta, clean, result, logged) in enumerate(zip(source, metadata, baseline, outcomes, raw)):
        digest = hashlib.sha256(example["text"].encode("utf-8")).hexdigest()
        if meta["text_sha256"] != digest or int(meta["row"]) != index:
            raise ValueError(f"Source metadata mismatch at {index}")
        if not example["text"] == clean["text"] == result["original_text"]:
            raise ValueError(f"Source/baseline/attack text mismatch at {index}")
        if not example["label"] == clean["label"] == result["label"] == manifest["label2id"][meta["label_name"]]:
            raise ValueError(f"Label mismatch at {index}")
        if not meta["source_id"] == clean["source_id"] == result["source_id"]:
            raise ValueError(f"Source ID mismatch at {index}")
        if int(clean["prediction"]) != int(result["original_prediction"]):
            raise ValueError(f"Initial prediction mismatch at {index}")
        for name in ("original", "perturbed"):
            if int(float(logged[f"{name}_output"])) != int(result[f"{name}_prediction"]):
                raise ValueError(f"TextAttack prediction mismatch at {index}")
        if logged["result_type"] != result["result_type"] or int(float(logged["num_queries"])) != int(result["num_queries"]):
            raise ValueError(f"TextAttack outcome mismatch at {index}")
    metrics = independent_metrics(outcomes)
    for name, expected in metrics.items():
        actual = summary["metrics"][name]
        if expected is None:
            if actual is not None:
                raise ValueError(f"Conditional metric should be unavailable: {name}")
        elif actual is None or not math.isclose(expected, actual, abs_tol=1e-9):
            raise ValueError(f"Independent metric mismatch: {name}")
    semantic = json.loads((run_dir / "semantic_similarity.json").read_text(encoding="utf-8"))
    if semantic["source_outcomes_sha256"] != sha256(run_dir / "outcomes.csv"):
        raise ValueError("Semantic source outcomes hash mismatch")
    similarities = read_csv(run_dir / "semantic_similarity.csv")
    successful = [r for r in outcomes if r["result_type"] == "Successful"]
    if semantic["n"] != len(successful) or len(similarities) != len(successful):
        raise ValueError("Semantic population mismatch")
    scores = []
    for row, result in zip(similarities, successful):
        if row["row"] != result["row"] or row["source_id"] != result["source_id"]:
            raise ValueError("Semantic/source alignment mismatch")
        score = float(row["cosine_similarity"])
        if not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
            raise ValueError("Invalid cosine similarity")
        scores.append(score)
    if scores and not math.isclose(sum(scores) / len(scores), semantic["mean_cosine"], abs_tol=1e-9):
        raise ValueError("Semantic mean mismatch")
    return {"dataset": summary["dataset"], "model": summary["model"].get("id", "Locally fine-tuned DistilBERT"),
        "input_classes": len({r["label"] for r in source}), "model_outputs": len(manifest["label2id"]),
        **metrics, "mean_use_large_cosine_successful": semantic["mean_cosine"],
        "original_texts_under_15_words": summary["short_text_bypass_examples"],
        "token_truncated_examples": sum(r["original_token_truncated"] == "True" for r in outcomes),
        "semantic_label_preservation_judged": summary["human_or_llm_semantic_judging_completed"],
        "run_directory": run_dir.as_posix(), "verification": "source, baseline, TextAttack logs, outcomes, metrics, cosine rows and artifact hashes reconciled"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Verification output already exists")
    rows = [verify_run(path, args.dataset_root) for path in args.run_dirs]
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "verification.json", {"runs": rows, "all_runs_reconciled": True,
        "comparison_sha256": sha256(args.output_dir / "comparison.csv"),
        "semantic_preservation_judging_complete": False})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
