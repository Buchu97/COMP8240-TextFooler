"""Measure a fixed Type 1 sample with the paper-delta TextFooler recipe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from run_constructed_pilot import summarise_attacks
from type1_data import sha256, write_json
from type1_models import baseline_metrics, configure_cache, load_classifier, read_csv


def validate_alignment(rows: list[dict], metadata: list[dict], label2id: dict[str, int]) -> None:
    if len(rows) != len(metadata) or not rows:
        raise ValueError("Dataset/metadata row counts differ or are empty")
    for index, (row, meta) in enumerate(zip(rows, metadata)):
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if int(meta["row"]) != index or meta["text_sha256"] != digest:
            raise ValueError(f"Dataset/metadata text alignment mismatch at row {index}")
        if label2id.get(meta["label_name"]) != row["label"]:
            raise ValueError(f"Dataset/metadata label mismatch at row {index}")


def attack_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No attack outcomes")
    return summarise_attacks(rows)


def add_utf8_text_log(manager, path: Path) -> None:
    """Give TextAttack an explicit UTF-8 stream instead of a locale-encoded file."""
    handle = path.open("x", encoding="utf-8")
    try:
        manager.add_output_file(handle, color_method="file")
    except Exception:
        handle.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-model", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=8240)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_path = args.dataset_dir / "evaluation.csv"
    if sha256(dataset_path) != manifest["evaluation"]["sha256"]:
        raise ValueError("Evaluation file hash mismatch")
    rows = read_csv(dataset_path)
    metadata = read_csv(args.dataset_dir / "evaluation_metadata.csv")
    validate_alignment(rows, metadata, manifest["label2id"])
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise SystemExit("Limit outside dataset range")
        rows, metadata = rows[:args.limit], metadata[:args.limit]
    if args.max_length < 8:
        raise SystemExit("Invalid tokenizer maximum length")
    training = None
    if manifest["model"].get("kind") == "local_training":
        if not args.local_model or not args.training_report:
            raise SystemExit("News requires --local-model and --training-report")
        training = json.loads(args.training_report.read_text(encoding="utf-8"))
        if training["status"] != "completed" or training["test_used_for_training_or_selection"]:
            raise ValueError("Training not complete or test data used for model selection")
        if training["dataset_manifest_sha256"] != sha256(args.dataset_dir / "manifest.json"):
            raise ValueError("Training/dataset manifest mismatch")
        for filename, digest in training["checkpoint_files"].items():
            if sha256(args.local_model / filename) != digest:
                raise ValueError(f"Training checkpoint hash mismatch: {filename}")
        if args.max_length != training["max_length"]:
            raise ValueError("Training and evaluation tokenizer lengths differ")
    elif args.local_model or args.training_report:
        raise SystemExit("Published classifiers cannot be overridden with a local model")
    root = Path(__file__).resolve().parents[1]
    configure_cache(root)
    import torch
    import textattack
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textfooler_paper_delta_0_7 import Attack
    textattack.shared.utils.set_seed(args.seed)
    torch.set_num_threads(8)
    model, tokenizer, device = load_classifier(manifest, args.local_model)
    tokenizer.model_max_length = args.max_length
    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    args.output_dir.mkdir(parents=True)
    run = {"dataset": manifest["dataset"], "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "examples_planned": len(rows),
        "evaluation_sha256": sha256(dataset_path), "dataset_manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
        "model": manifest["model"], "local_checkpoint_sha256": training["checkpoint_files"] if training else None,
        "training_report_sha256": sha256(args.training_report) if training else None,
        "label2id": manifest["label2id"], "seed": args.seed, "max_length": args.max_length,
        "recipe": "TextAttack TextFoolerJin2019 with WordEmbeddingDistance(min_cos_sim=0.7)",
        "recipe_revision_tag": "PRECONSTRAINTS_RESTORED_20261008",
        "max_candidates_N": 50, "word_similarity_delta": 0.7,
        "sentence_constraint": {"encoder": "TextAttack UniversalSentenceEncoder (non-large)", "url": "https://tfhub.dev/google/universal-sentence-encoder/3", "metric": "angular",
            "threshold": 0.840845057, "window_size": 15, "compare_against_original": False,
            "skip_text_shorter_than_window": True},
        "attack_recipe_sha256": sha256(root / "scripts/textfooler_paper_delta_0_7.py"),
        "packages": {name: version(name) for name in ("torch", "transformers", "textattack")},
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "human_or_llm_semantic_judging_completed": False,
        "evaluation_selection": manifest["evaluation"]["selection"],
        "checkpoint_training_membership": manifest.get("checkpoint_training_membership",
            "Locally trained on saved disjoint train split; selected using validation only")}
    write_json(args.output_dir / "run_metadata.json", run)
    baseline = []
    names = [name for name, index in sorted(manifest["label2id"].items(), key=lambda item: item[1])]
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        scores = torch.softmax(wrapper([r["text"] for r in batch]), dim=-1).cpu().tolist()
        for offset, (row, probabilities) in enumerate(zip(batch, scores)):
            index = start + offset
            tokens = len(tokenizer(row["text"], truncation=False)["input_ids"])
            baseline.append({"row": index, "source_id": metadata[index]["source_id"],
                "text": row["text"], "label": row["label"], "label_name": metadata[index]["label_name"],
                "prediction": max(range(len(names)), key=lambda k: probabilities[k]),
                "probabilities": json.dumps(probabilities), "token_count": tokens,
                "token_truncated": tokens > args.max_length})
    with (args.output_dir / "baseline.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline[0]))
        writer.writeheader()
        writer.writerows(baseline)
    baseline_summary = baseline_metrics(baseline, manifest["label2id"])
    baseline_summary["token_truncated_examples"] = sum(r["token_truncated"] for r in baseline)
    write_json(args.output_dir / "baseline_summary.json", baseline_summary)
    print(f"{manifest['dataset']}: clean {baseline_summary['correct']}/{len(rows)}; beginning attack", flush=True)
    attack = Attack(wrapper)
    with (args.output_dir / "attack_configuration.txt").open("x", encoding="utf-8") as config_handle:
        config_handle.write(repr(attack) + "\n")
    logger = textattack.loggers.AttackLogManager(metrics=None)
    add_utf8_text_log(logger, args.output_dir / "attack_results.txt")
    logger.add_output_csv(str(args.output_dir / "attack_results.csv"), color_method="file")
    logger.add_output_summary_json(str(args.output_dir / "attack_summary.json"))
    outcomes = []
    started = time.monotonic()
    with (args.output_dir / "outcomes.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = None
        for index, (row, meta) in enumerate(zip(rows, metadata)):
            result = attack.attack(row["text"], row["label"])
            original, perturbed = result.original_result.attacked_text, result.perturbed_result.attacked_text
            if original.text != row["text"] or int(result.original_result.output) != baseline[index]["prediction"]:
                raise RuntimeError(f"Attack/baseline alignment mismatch at row {index}")
            outcome = {"row": index, "source_id": meta["source_id"], "category": meta["label_name"],
                "label": row["label"], "original_text": original.text, "perturbed_text": perturbed.text,
                "original_prediction": int(result.original_result.output),
                "perturbed_prediction": int(result.perturbed_result.output),
                "result_type": type(result).__name__.removesuffix("AttackResult"),
                "num_queries": int(result.num_queries), "original_word_count": original.num_words,
                "changed_word_count": original.words_diff_num(perturbed),
                "sentence_constraint_short_text_bypass": original.num_words < 15,
                "original_token_count": baseline[index]["token_count"],
                "perturbed_token_count": len(tokenizer(perturbed.text, truncation=False)["input_ids"]),
                "original_token_truncated": baseline[index]["token_truncated"],
                "elapsed_seconds": time.monotonic() - started}
            # Validate each recorded outcome before writing; completed rows survive interruptions.
            attack_metrics([outcome])
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(outcome))
                writer.writeheader()
            writer.writerow(outcome)
            handle.flush()
            outcomes.append(outcome)
            logger.log_result(result)
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                logger.flush()
                counts = {kind: sum(r["result_type"] == kind for r in outcomes) for kind in ("Successful", "Failed", "Skipped")}
                print(f"{manifest['dataset']}: {index + 1}/{len(rows)}, {counts}, elapsed {time.monotonic() - started:.0f}s", flush=True)
    metrics = attack_metrics(outcomes)
    if metrics["eligible"]:
        logger.log_summary()
    logger.flush()
    for output_logger in logger.loggers:
        output_logger.close()
    write_json(args.output_dir / "summary.json", {**run, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "metrics": metrics, "short_text_bypass_examples": sum(r["sentence_constraint_short_text_bypass"] for r in outcomes),
        "metric_definitions": {"accuracy": "Correct/all fixed sample examples; initially incorrect predictions stay in denominator",
            "attack_success_rate": "Successful/(successful+failed); initially incorrect examples skipped",
            "mean_queries": "Successful and failed attempts; separate clean baseline queries excluded",
            "mean_perturbed_words": "Mean changed-word percentage over successful attacks only"},
        "artifacts_sha256": {p.name: sha256(p) for p in args.output_dir.iterdir() if p.is_file()}})
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_category"}), flush=True)


if __name__ == "__main__":
    main()
