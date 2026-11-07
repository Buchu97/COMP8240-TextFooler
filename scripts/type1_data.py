"""Dataset conversion, fixed sampling and split checks for Type 1 experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def text_key(text: str) -> str:
    return " ".join(text.casefold().split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_rows(dataset: str, rows: list[dict], label_names: list[str]) -> list[dict]:
    converted = []
    for index, row in enumerate(rows):
        if dataset == "poem":
            text, name = row["verse_text"], label_names[row["label"]]
            source_id = str(row["id"])
        elif dataset == "clinc150":
            text, name = row["text"], label_names[row["intent"]]
            if name == "oos":
                continue
            source_id = str(index)
        elif dataset == "news":
            headline, description = row["headline"], row["short_description"]
            text = headline + (" " + description if description else "")
            name, source_id = row["category"], row["link"]
            if not text.strip():
                continue
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if not isinstance(text, str) or not text.strip() or not isinstance(name, str) or not name:
            raise ValueError(f"Empty text or label at source row {index}")
        converted.append({"source_id": source_id, "source_row": index,
                          "text": text, "label_name": name})
    return converted


def map_labels(records: list[dict], label2id: dict[str, int]) -> list[dict]:
    if len(set(label2id.values())) != len(label2id):
        raise ValueError("Label IDs must be unique")
    missing = {r["label_name"] for r in records} - set(label2id)
    if missing:
        raise ValueError(f"Missing label mappings: {sorted(missing)}")
    return [dict(r, label=int(label2id[r["label_name"]])) for r in records]


def deduplicate_news(records: list[dict]) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    for row in records:
        groups[text_key(row["text"])].append(row)
    clean, duplicate, conflicting = [], 0, 0
    for group in groups.values():
        if len({r["label_name"] for r in group}) > 1:
            conflicting += len(group)
        else:
            clean.append(group[0])
            duplicate += len(group) - 1
    return clean, {"duplicate_rows_removed": duplicate, "conflicting_rows_removed": conflicting}


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    if not 1 <= n <= len(records):
        raise ValueError("Sample size must be between 1 and population size")
    groups = defaultdict(list)
    for row in records:
        groups[row["label_name"]].append(row)
    if n < len(groups):
        raise ValueError("Sample size must cover every class")
    rng = random.Random(seed)
    names = sorted(groups)
    for name in names:
        rng.shuffle(groups[name])
    # One record per class, then allocate remaining places by proportional deficit.
    counts = {name: 1 for name in names}
    for _ in range(n - len(names)):
        name = max((k for k in names if counts[k] < len(groups[k])),
                   key=lambda k: n * len(groups[k]) / len(records) - counts[k])
        counts[name] += 1
    selected = [row for name in names for row in groups[name][:counts[name]]]
    rng.shuffle(selected)
    return selected


def split_news(records: list[dict], seed: int) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in records:
        groups[row["label_name"]].append(row)
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for name in sorted(groups):
        rows = list(groups[name])
        if len(rows) < 10:
            raise ValueError(f"Not enough examples for three splits: {name}")
        rng.shuffle(rows)
        test_n, validation_n = len(rows) // 10, len(rows) // 10
        splits["test"].extend(rows[:test_n])
        splits["validation"].extend(rows[test_n:test_n + validation_n])
        splits["train"].extend(rows[test_n + validation_n:])
    keys = [set(text_key(r["text"]) for r in rows) for rows in splits.values()]
    if any(keys[i] & keys[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Text overlap between News splits")
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_dataset(directory: Path, name: str, rows: list[dict]) -> dict:
    csv_path = directory / f"{name}.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows({"text": r["text"], "label": r["label"]} for r in rows)
    with (directory / f"{name}_metadata.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "source_id", "source_row",
                                                   "label_name", "text_sha256", "words"])
        writer.writeheader()
        for index, r in enumerate(rows):
            writer.writerow({"row": index, "source_id": r["source_id"],
                "source_row": r["source_row"], "label_name": r["label_name"],
                "text_sha256": hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
                "words": len(r["text"].split())})
    return {"examples": len(rows), "sha256": sha256(csv_path),
            "label_counts": dict(sorted(Counter(r["label_name"] for r in rows).items())),
            "shorter_than_15_whitespace_words": sum(len(r["text"].split()) < 15 for r in rows)}
