"""Label contracts and clean classifier metrics for multiclass experiments."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

from type1_data import text_key


def configure_cache(root: Path) -> None:
    for name, folder in {"TA_CACHE_DIR": "textattack", "HF_HOME": "huggingface",
                         "TORCH_HOME": "torch", "NLTK_DATA": "nltk",
                         "TFHUB_CACHE_DIR": "tensorflow_hub"}.items():
        os.environ.setdefault(name, str(root / "data/cache" / folder))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def validate_label_map(expected: dict[str, int], actual: dict[int, str]) -> None:
    actual = {int(index): name for index, name in actual.items()}
    reverse = {name: index for index, name in actual.items()}
    if len(reverse) != len(actual) or expected != reverse:
        raise ValueError("Dataset/model label mappings differ")
    if set(actual) != set(range(len(actual))):
        raise ValueError("Model label IDs must be contiguous")


def validate_split_separation(splits: dict[str, list[dict]]) -> None:
    keys = {name: set(text_key(r["text"]) for r in rows) for name, rows in splits.items()}
    names = list(keys)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if keys[name] & keys[other]:
                raise ValueError(f"Text overlap between {name} and {other}")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if "label" in row:
            row["label"] = int(row["label"])
    return rows


def baseline_metrics(rows: list[dict], label2id: dict[str, int]) -> dict:
    if not rows:
        raise ValueError("No baseline examples")
    matrix = [[0] * len(label2id) for _ in label2id]
    correct = 0
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction"])
        if not 0 <= label < len(matrix) or not 0 <= prediction < len(matrix):
            raise ValueError("Prediction or label outside model label range")
        matrix[label][prediction] += 1
        correct += label == prediction
    return {"examples": len(rows), "correct": correct, "accuracy_percent": 100 * correct / len(rows),
            "confusion_matrix": matrix, "confusion_matrix_label_order":
            [name for name, index in sorted(label2id.items(), key=lambda x: x[1])],
            "by_label": {name: {"examples": sum(matrix[index]), "correct": matrix[index][index],
                "accuracy_percent": 100 * matrix[index][index] / sum(matrix[index]) if sum(matrix[index]) else None}
                for name, index in label2id.items()}}


def load_classifier(manifest: dict, local_model: Path | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    spec = manifest["model"]
    model_id = str(local_model) if local_model else spec["id"]
    revision = None if local_model else spec["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
                                                              trust_remote_code=False)
    validate_label_map(manifest["label2id"], model.config.id2label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device
