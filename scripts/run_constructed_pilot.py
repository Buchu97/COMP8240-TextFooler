"""Measure clean predictions and a stratified TextFooler pilot on constructed data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "textattack/bert-base-uncased-rotten-tomatoes"


def summarise_baseline(rows: list[dict]) -> dict:
    def totals(group):
        correct = sum(row["prediction"] == row["label"] for row in group)
        return {"examples": len(group), "correct": correct,
                "accuracy_percent": 100 * correct / len(group) if group else None}

    return totals(rows) | {
        "by_category": {
            category: totals([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        }
    }


def summarise_attacks(rows: list[dict]) -> dict:
    for row in rows:
        initially_correct = row["original_prediction"] == row["label"]
        finally_correct = row["perturbed_prediction"] == row["label"]
        expected = ("Skipped" if not initially_correct else
                    "Failed" if finally_correct else "Successful")
        if row["result_type"] != expected:
            raise ValueError(f"Inconsistent attack outcome: {row['result_type']} vs {expected}")

    def totals(group):
        counts = Counter(row["result_type"] for row in group)
        eligible = [row for row in group if row["result_type"] != "Skipped"]
        successful = [row for row in group if row["result_type"] == "Successful"]
        return {
            "examples": len(group),
            "successful": counts["Successful"],
            "failed": counts["Failed"],
            "skipped": counts["Skipped"],
            "eligible": len(eligible),
            "original_accuracy_percent": 100 * len(eligible) / len(group) if group else None,
            "after_attack_accuracy_percent": 100 * counts["Failed"] / len(group) if group else None,
            "attack_success_rate_percent": 100 * len(successful) / len(eligible) if eligible else None,
            "mean_queries_eligible": sum(r["num_queries"] for r in eligible) / len(eligible) if eligible else None,
            "mean_perturbed_word_percent_successful": (
                sum(100 * r["changed_word_count"] / r["original_word_count"] for r in successful)
                / len(successful) if successful else None
            ),
        }

    return totals(rows) | {
        "by_category": {
            category: totals([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        }
    }


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/processed/movie_sentiment_v0_1/dataset.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()

    from prepare_constructed_sentiment import read_jsonl, select_pilot, write_jsonl

    records = read_jsonl(args.dataset)
    selected = select_pilot(records, args.per_category, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "experiment_tag": "CONSTRUCTED_V0_1_PILOT_WORD_DELTA_0_7",
        "started_at_utc": started_at,
        "dataset_id": records[0]["dataset_id"],
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model_id": MODEL_ID,
        "label_map": {"negative": 0, "positive": 1},
        "label_status": "intended_unverified",
        "baseline_examples": len(records),
        "attack_examples": len(selected),
        "selection": "Complete pairs sampled by category before model predictions; no correctness filtering",
        "attack_order": "Saved selected-example order; no further shuffle",
        "per_category": args.per_category,
        "random_seed": args.seed,
        "max_word_candidates_N": 50,
        "minimum_word_embedding_cosine_delta": 0.7,
        "recipe": "TextAttack TextFoolerJin2019 with word delta changed to 0.7",
        "sentence_similarity_constraint": {
            "encoder": "UniversalSentenceEncoder", "metric": "angular",
            "threshold": 0.840845057, "window_size_words": 15,
            "skip_text_shorter_than_window": True, "compare_against_original": False,
            "behaviour": "Candidates with fewer than 15 TextAttack words receive similarity score 1 and bypass the threshold",
        },
        "independent_human_review_complete": False,
        "semantic_judging_complete": False,
    }
    _write_json(args.output_dir / "EXPERIMENT_METADATA.json", metadata)
    write_jsonl(args.output_dir / "selected_examples.jsonl", selected)

    cache_root = PROJECT_ROOT / "data/cache"
    for key, value in {
        "TA_CACHE_DIR": cache_root / "textattack",
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "NLTK_DATA": cache_root / "nltk",
        "TFHUB_CACHE_DIR": cache_root / "tensorflow_hub",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }.items():
        os.environ.setdefault(key, str(value))

    import torch
    import textattack
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from textfooler_paper_delta_0_7 import Attack

    textattack.shared.utils.set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if model.config.num_labels != 2:
        raise ValueError("The movie-sentiment pilot requires a binary sentiment checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    anchors = ["This was a dreadful movie. I hated every minute of it.",
               "This was a wonderful movie. I loved every minute of it."]
    anchor_predictions = wrapper(anchors).argmax(dim=-1).cpu().tolist()
    if anchor_predictions != [0, 1]:
        raise ValueError(f"Sentiment label-map sanity check failed: {anchor_predictions}")

    baseline = []
    for start in range(0, len(records), 8):
        batch = records[start:start + 8]
        probabilities = torch.softmax(wrapper([r["text"] for r in batch]), dim=-1).cpu().tolist()
        for record, scores in zip(batch, probabilities):
            prediction = max(range(2), key=lambda label: scores[label])
            baseline.append({
                "example_id": record["example_id"], "family_id": record["family_id"],
                "category": record["category"], "text": record["text"], "label": record["label"],
                "label_status": record["label_status"], "prediction": prediction,
                "probabilities": {"negative": scores[0], "positive": scores[1]},
            })
    write_jsonl(args.output_dir / "baseline_predictions.jsonl", baseline)
    _write_json(args.output_dir / "baseline_summary.json", summarise_baseline(baseline))
    print(f"Baseline completed on {len(baseline)} draft examples; attacking {len(selected)} selected examples.", flush=True)

    attack = Attack(wrapper)
    dataset = textattack.datasets.Dataset(
        [(r["text"], r["label"]) for r in selected], label_names=["negative", "positive"]
    )
    attack_args = textattack.AttackArgs(
        num_examples=len(selected), random_seed=args.seed, shuffle=False,
        log_to_txt=str(args.output_dir / "attack_results.txt"),
        log_to_csv=str(args.output_dir / "attack_results.csv"),
        log_summary_to_json=str(args.output_dir / "attack_summary.json"),
        disable_stdout=True,
    )
    results = textattack.Attacker(attack, dataset, attack_args).attack_dataset()
    if len(results) != len(selected):
        raise RuntimeError("Attack result count differs from the saved selected examples")
    baseline_by_id = {row["example_id"]: row for row in baseline}
    enriched = []
    for record, result in zip(selected, results):
        original = result.original_result.attacked_text
        perturbed = result.perturbed_result.attacked_text
        if original.text != record["text"]:
            raise RuntimeError("Attack results are not aligned with source-example order")
        if int(result.original_result.output) != baseline_by_id[record["example_id"]]["prediction"]:
            raise RuntimeError("Attack and baseline initial predictions disagree")
        enriched.append({
            "example_id": record["example_id"], "family_id": record["family_id"],
            "pair_id": record["pair_id"], "category": record["category"],
            "rhetorical_style": record["rhetorical_style"], "label": record["label"],
            "label_status": record["label_status"],
            "original_text": original.text, "perturbed_text": perturbed.text,
            "original_prediction": int(result.original_result.output),
            "perturbed_prediction": int(result.perturbed_result.output),
            "result_type": type(result).__name__.removesuffix("AttackResult"),
            "num_queries": int(result.num_queries), "original_word_count": original.num_words,
            "changed_word_count": original.words_diff_num(perturbed),
            "semantic_judgement": None, "human_review_status": "pending",
        })
    metrics = summarise_attacks(enriched)
    write_jsonl(args.output_dir / "attack_examples.jsonl", enriched)
    _write_json(args.output_dir / "pilot_summary.json", {
        "experiment_tag": metadata["experiment_tag"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "device": str(device),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "packages": {name: version(name) for name in ("textattack", "torch", "transformers")},
        "label_map_anchor_predictions": anchor_predictions,
        "label_status": "intended_unverified",
        "metrics": metrics,
        "semantic_judging_complete": False,
        "metric_definitions": {
            "accuracy": "All selected examples; initially wrong predictions remain in the denominator",
            "attack_success_rate": "Successful / (successful + failed); initially wrong examples skipped",
            "mean_queries": "Successful and failed attempts only; separate clean baseline queries excluded",
            "mean_perturbed_words": "Mean percentage of changed TextAttack words over successful attacks only",
        },
        "limitations": [
            "Metrics use draft intended labels, not independently verified human labels.",
            "Twelve-example default pilot is a pipeline check, not a stable effectiveness estimate.",
            "The retained sentence-similarity constraint accepts candidates under 15 words without enforcing its similarity threshold.",
            "Prediction changes have not yet been checked for semantic or label preservation.",
            "Template-family variants are dependent; no statistical independence is assumed.",
        ],
    })
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
