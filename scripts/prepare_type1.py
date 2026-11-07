"""Fetch pinned public sources and prepare fixed Type 1 evaluation inputs."""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from type1_data import (convert_rows, deduplicate_news, map_labels, sha256,
                        split_news, stratified_sample, write_dataset, write_json)


SOURCES = {
    "poem": ("google-research-datasets/poem_sentiment", "685b95a2787a869b7bae6c4480810f57fe23b48e"),
    "clinc150": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec"),
    "news": ("heegyu/news-category-dataset", "304a05a55bc6abc0446d8fae0d0771716b6a271a"),
}
MODELS = {
    "poem": ("nickwong64/bert-base-uncased-poems-sentiment", "ecc8e77abaf15a7f9f0e318b312786d91cae058b"),
    "clinc150": ("transformersbook/bert-base-uncased-finetuned-clinc", "795b076da71dc236dde692338e21560cbbffa6e4"),
}


def fetch(url: str, path: Path) -> bytes:
    import requests
    if path.exists():
        return path.read_bytes()
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(response.content)
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"Output already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    cache = root / "data/raw/type1_pinned"
    import pandas as pd
    for dataset, (source, revision) in SOURCES.items():
        directory = args.output_root / dataset
        directory.mkdir()
        card_url = f"https://huggingface.co/api/datasets/{source}/revision/{revision}"
        card = json.loads(fetch(card_url, cache / dataset / "card.json"))
        info = card["cardData"].get("dataset_info")
        manifest = {"dataset": dataset, "source": source, "source_revision": revision,
            "source_url": f"https://huggingface.co/datasets/{source}/tree/{revision}",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "source_license_metadata": card["cardData"].get("license"),
            "preprocessing": "Field selection and label mapping; source text unchanged",
            "splits": {}, "source_files": []}
        if dataset == "news":
            file_url = f"https://huggingface.co/datasets/{source}/resolve/{revision}/data.json"
            raw_path = cache / dataset / "data.json"
            payload = fetch(file_url, raw_path)
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
            converted = convert_rows(dataset, rows, [])
            empty_rows = len(rows) - len(converted)
            converted, cleaning = deduplicate_news(converted)
            cleaning["empty_rows_removed"] = empty_rows
            label2id = {name: index for index, name in enumerate(sorted({r["label_name"] for r in converted}))}
            splits = split_news(map_labels(converted, label2id), args.seed)
            manifest.update({"original_rows": len(rows), "cleaning": cleaning,
                "preprocessing": "headline + one space + short_description when nonempty; source fields unchanged; case/whitespace keys used only for deduplication",
                "split_policy": "Per-class 80/10/10, test and validation each floor(n/10); deduplicated before split",
                "model": {"kind": "local_training", "base": "distilbert/distilbert-base-uncased",
                          "base_revision": "12040accade4e8a0f71eabdb258fecc2e7e948be"},
                "author_source_url": "https://www.kaggle.com/datasets/rmisra/news-category-dataset",
                "dataset_paper": "https://arxiv.org/abs/2209.11429"})
            manifest["source_files"].append({"url": file_url, "sha256": sha256(raw_path)})
        else:
            config_name = "default" if dataset == "poem" else "plus"
            selected_info = info if isinstance(info, dict) else next(x for x in info if x["config_name"] == config_name)
            feature = next(x for x in selected_info["features"] if x["name"] == ("label" if dataset == "poem" else "intent"))
            names = feature["dtype"]["class_label"]["names"]
            label_names = [names[str(i)] for i in range(len(names))]
            model_id, model_revision = MODELS[dataset]
            model_url = f"https://huggingface.co/{model_id}/resolve/{model_revision}/config.json"
            model_config = json.loads(fetch(model_url, cache / dataset / "model_config.json"))
            label2id = {name: int(index) for index, name in model_config["id2label"].items()}
            if set(label2id) != set(label_names):
                raise ValueError(f"Source/model label mismatch: {dataset}")
            splits = {}
            prefix = "data" if dataset == "poem" else "plus"
            for split in ("train", "validation", "test"):
                filename = f"{split}-00000-of-00001.parquet"
                url = f"https://huggingface.co/datasets/{source}/resolve/{revision}/{prefix}/{filename}"
                raw_path = cache / dataset / filename
                payload = fetch(url, raw_path)
                rows = pd.read_parquet(io.BytesIO(payload)).to_dict("records")
                splits[split] = map_labels(convert_rows(dataset, rows, label_names), label2id)
                manifest["source_files"].append({"url": url, "sha256": sha256(raw_path), "rows": len(rows)})
            manifest["model"] = {"id": model_id, "revision": model_revision,
                "card_url": f"https://huggingface.co/{model_id}/blob/{model_revision}/README.md"}
            manifest["split_policy"] = "Published test split; CLINC OOS input rows excluded" if dataset == "clinc150" else "Published test split"
            manifest["checkpoint_training_membership"] = ("Published book notebook evaluates this checkpoint on the dataset test split; exact training rows not independently reconstructed"
                if dataset == "clinc150" else "Model card declares Poem Sentiment fine-tuning; exact training rows not published")
        manifest["label2id"] = label2id
        for split, records in splits.items():
            # Large training/test exports are reproducible locally; only fixed evaluation inputs are tracked.
            manifest["splits"][split] = write_dataset(directory, split, records)
        n = 104 if dataset == "poem" else 1000
        evaluation = splits["test"] if dataset == "poem" else stratified_sample(splits["test"], n, args.seed)
        if len(evaluation) != n:
            raise ValueError(f"Unexpected evaluation size for {dataset}: {len(evaluation)}")
        manifest["evaluation"] = write_dataset(directory, "evaluation", evaluation)
        manifest["evaluation"]["selection"] = "Full test split" if dataset == "poem" else "Fixed proportional stratified test sample, selected before predictions"
        write_json(directory / "manifest.json", manifest)
        print(json.dumps({"dataset": dataset, "split_counts": {k: len(v) for k, v in splits.items()}, "evaluation": len(evaluation)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
