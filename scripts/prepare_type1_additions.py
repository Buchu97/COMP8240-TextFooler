"""Prepare fixed SciCite and EnvironmentalClaims inputs from pinned author files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from type1_data import (deduplicate_news, map_labels, sha256, stratified_sample,
                        text_key, write_dataset, write_json)
from type1_models import validate_split_separation


SCICITE_SHA256 = "711ece2c4e61d116c8ae5bb07e9fbb2ee9ff7bba004b4cab7fbd0ac3af499193"
CLAIMS_COMMIT = "7f3a8c1e7f689bbfb67184ab81d56f1214e9d790"
CLAIMS_HASHES = {
    "train": "c8210e69f55e8f233ad84741dc44395a9b0f9407c346051218d11f38ba4e5de1",
    "validation": "03cdb43c57c1080df4e3c879e51979e47c7bc7214a49f5a6c8a20ab783bc89e2",
    "test": "49bc5ed6b2a6d01a9be5ecf900bac94992c9084b5c9fe95ad305ac54343948d8",
}


def checked_fetch(url: str, path: Path, expected: str) -> bytes:
    if not path.exists():
        import requests
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(response.content)
    if sha256(path) != expected:
        raise ValueError(f"Pinned source hash mismatch: {path}")
    return path.read_bytes()


def prepare_scicite(root: Path, raw_dir: Path) -> tuple[dict, dict]:
    url = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
    archive = raw_dir / "scicite.tar.gz"
    checked_fetch(url, archive, SCICITE_SHA256)
    labels = {"background": 0, "method": 1, "result": 2}
    splits, cleaning, provenance = {}, {}, []
    with tarfile.open(archive, "r:gz") as source:
        for split, filename in (("train", "train"), ("validation", "dev"), ("test", "test")):
            member = source.extractfile(f"scicite/{filename}.jsonl")
            if member is None:
                raise ValueError(f"Missing author split: {split}")
            rows = [json.loads(line) for line in member if line.strip()]
            seen, records = set(), []
            for index, row in enumerate(rows):
                identity = str(row["unique_id"])
                if identity in seen:
                    continue
                seen.add(identity)
                if not isinstance(row["string"], str) or not row["string"].strip():
                    raise ValueError(f"Empty SciCite text: {split}/{identity}")
                records.append({"source_id": identity, "source_row": index,
                                "text": row["string"], "label_name": row["label"]})
                provenance.append({"split": split, "source_id": identity,
                    "citing_paper_id": row["citingPaperId"], "cited_paper_id": row["citedPaperId"],
                    "text_sha256": hashlib.sha256(row["string"].encode("utf-8")).hexdigest()})
            records, text_cleaning = deduplicate_news(records)
            cleaning[split] = {"raw_rows": len(rows), "unique_source_ids": len(seen),
                               **text_cleaning}
            splits[split] = map_labels(records, labels)
    validation_keys = {text_key(row["text"]) for row in splits["validation"]}
    test_keys = {text_key(row["text"]) for row in splits["test"]}
    for split, blocked in (("train", validation_keys | test_keys), ("validation", test_keys)):
        before = len(splits[split])
        splits[split] = [row for row in splits[split] if text_key(row["text"]) not in blocked]
        cleaning[split]["cross_split_text_rows_removed"] = before - len(splits[split])
    validate_split_separation(splits)
    manifest = {
        "dataset": "scicite", "source": "allenai/scicite",
        "source_url": "https://github.com/allenai/scicite",
        "source_revision": SCICITE_SHA256,
        "source_files": [{"url": url, "sha256": SCICITE_SHA256}],
        "source_license_metadata": "Hugging Face dataset metadata: unknown; author code repository: Apache-2.0",
        "task": "Citation intent: background, method or result",
        "label2id": labels,
        "preprocessing": "Original string field unchanged. First occurrence of each unique_id as in the pinned Hugging Face loader, then normalized-text duplicate removal; conflicting-label text groups removed. Text keys do not alter classifier inputs.",
        "cleaning": cleaning,
        "split_policy": "Published train/dev/test membership retained; overlapping text removed from train before validation and from train/validation before test.",
        "checkpoint_training_membership": "Classifier trained locally on saved training rows; checkpoint selected by validation accuracy only. Evaluation test rows and normalized texts are disjoint from training and validation.",
        "split_scope_limit": "Published partitions split citation examples rather than all citing papers; citing papers may occur in multiple splits.",
        "model": {"kind": "local_training", "id": "models/type1_scicite_scibert_20261102",
            "base": "models/scibert_base_20261102",
            "base_source_id": "allenai/scibert_scivocab_uncased",
            "base_revision": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"},
        "raw_provenance_rows": provenance,
    }
    return manifest, splits


def prepare_claims(root: Path, raw_dir: Path) -> tuple[dict, dict]:
    splits, files = {}, []
    labels = {"no": 0, "yes": 1}
    for split, filename in (("train", "train"), ("validation", "dev"), ("test", "test")):
        url = f"https://raw.githubusercontent.com/dominiksinsaarland/environmental_claims/{CLAIMS_COMMIT}/data/{filename}.jsonl"
        data = checked_fetch(url, raw_dir / f"environmental_claims_{filename}.jsonl", CLAIMS_HASHES[split])
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        records = [{"source_id": f"{split}:{i}", "source_row": i, "text": row["text"],
                    "label_name": "yes" if row["label"] == 1 else "no"} for i, row in enumerate(rows)]
        if any(row["label"] not in (0, 1) or not row["text"].strip() for row in rows):
            raise ValueError(f"Invalid EnvironmentalClaims row: {split}")
        splits[split] = map_labels(records, labels)
        files.append({"url": url, "sha256": CLAIMS_HASHES[split], "rows": len(rows)})
    validate_split_separation(splits)
    model_dir = root / "models/environmental_claims_screening_20261102"
    expected_model = "577ddac3d419dadda95238224d3f5d106f5bc7ad171d7f305d0f6143e2326ec4"
    if not (model_dir / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        snapshot_download("climatebert/environmental-claims",
            revision="9ce1fe309ed7088a2f0b7af422eb44107c335b99",
            local_dir=model_dir, allow_patterns=["*.json", "merges.txt", "model.safetensors"])
    if sha256(model_dir / "model.safetensors") != expected_model:
        raise ValueError("EnvironmentalClaims author checkpoint hash mismatch")
    manifest = {
        "dataset": "environmental_claims", "source": "climatebert/environmental_claims",
        "source_revision": "8982dbea4a595589b7ebe46b3d7eec6707eeea16",
        "source_url": "https://huggingface.co/datasets/climatebert/environmental_claims",
        "author_commit": CLAIMS_COMMIT, "source_files": files,
        "source_license_metadata": "CC-BY-NC-SA-4.0",
        "task": "Environmental claim presence, not claim truth verification",
        "label2id": labels, "preprocessing": "Original text unchanged; integer source labels retained",
        "split_policy": "Full published test split; fixed author train/dev/test partitions",
        "checkpoint_training_membership": "Author code trains the final model on train and evaluates dev/test. Exact historical training invocation is not independently reconstructed.",
        "model": {"kind": "published_local_snapshot",
            "id": "models/environmental_claims_screening_20261102",
            "source_id": "climatebert/environmental-claims",
            "revision": "9ce1fe309ed7088a2f0b7af422eb44107c335b99",
            "weights_sha256": expected_model},
    }
    return manifest, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit("Output root already exists")
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data/raw/type1_additions_20261102"
    args.output_root.mkdir(parents=True)
    for dataset, prepare, count in (("environmental_claims", prepare_claims, 265),
                                     ("scicite", prepare_scicite, 1000)):
        manifest, splits = prepare(root, raw_dir)
        directory = args.output_root / dataset
        directory.mkdir()
        manifest.update(seed=args.seed, retrieved_at_utc=datetime.now(timezone.utc).isoformat(), splits={})
        for split, rows in splits.items():
            manifest["splits"][split] = write_dataset(directory, split, rows)
        evaluation = splits["test"] if dataset == "environmental_claims" else stratified_sample(splits["test"], count, args.seed)
        if len(evaluation) != count:
            raise ValueError("Unexpected evaluation size")
        manifest["evaluation"] = write_dataset(directory, "evaluation", evaluation)
        manifest["evaluation"]["selection"] = ("Full published test split" if dataset == "environmental_claims"
            else "Fixed proportional stratified sample from deduplicated published test split; selected before training or predictions")
        provenance = manifest.pop("raw_provenance_rows", None)
        if provenance:
            with (directory / "source_provenance.csv").open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
                writer.writeheader()
                writer.writerows(provenance)
            manifest["source_provenance_sha256"] = sha256(directory / "source_provenance.csv")
        write_json(directory / "manifest.json", manifest)
        print(json.dumps({"dataset": dataset, "splits": {s: len(r) for s, r in splits.items()},
            "evaluation": len(evaluation), "labels": dict(Counter(r["label_name"] for r in evaluation))}), flush=True)


if __name__ == "__main__":
    main()
