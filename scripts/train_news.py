"""Fine-tune DistilBERT on the saved News training split; select on validation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from type1_data import sha256, write_json
from type1_models import configure_cache, read_csv, validate_split_separation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()
    if args.model_dir.exists() or args.report_dir.exists():
        raise SystemExit("Model or report directory already exists")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise SystemExit("Invalid training parameters")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = {name: read_csv(args.dataset_dir / f"{name}.csv") for name in ("train", "validation", "test")}
    for name in splits:
        if sha256(args.dataset_dir / f"{name}.csv") != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Split hash mismatch: {name}")
    validate_split_separation(splits)
    configure_cache(Path(__file__).resolve().parents[1])
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("News training requires the configured CUDA GPU")
    spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["base"], revision=spec["base_revision"], trust_remote_code=False)
    labels = manifest["label2id"]
    model = AutoModelForSequenceClassification.from_pretrained(spec["base"], revision=spec["base_revision"],
        num_labels=len(labels), id2label={index: name for name, index in labels.items()},
        label2id=labels, trust_remote_code=False, use_safetensors=True).to(device)
    tokenizer.model_max_length = args.max_length

    def loader(rows, shuffle, batch_size):
        encoded = tokenizer([r["text"] for r in rows], truncation=True, max_length=args.max_length,
                            padding="max_length", return_tensors="pt")
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"],
                                torch.tensor([r["label"] for r in rows]))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

    # Test texts are used only for the overlap check above, never training or checkpoint selection.
    train_loader = loader(splits["train"], True, args.batch_size)
    validation_loader = loader(splits["validation"], False, args.batch_size * 2)
    del splits
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    args.model_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    started = time.monotonic()
    settings = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": spec["base"], "base_revision": spec["base_revision"], "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "max_length": args.max_length,
        "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_fraction": 0.1,
        "mixed_precision": "float16", "selection_metric": "validation accuracy, earliest epoch wins ties",
        "test_used_for_training_or_selection": False, "dataset_manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
        "split_sha256": {name: manifest["splits"][name]["sha256"] for name in ("train", "validation", "test")},
        "split_examples": {name: manifest["splits"][name]["examples"] for name in ("train", "validation", "test")},
        "device": torch.cuda.get_device_name(0)}
    write_json(args.report_dir / "training_settings.json", settings)
    best_accuracy, best_epoch = -1.0, None
    history = []
    with (args.report_dir / "epoch_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss",
            "validation_correct", "validation_examples", "validation_accuracy_percent", "elapsed_seconds", "selected"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss, seen = 0.0, 0
            for step, batch in enumerate(train_loader, 1):
                ids, mask, target = [tensor.to(device, non_blocking=True) for tensor in batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=target).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += float(loss.detach()) * len(target)
                seen += len(target)
                if step % 100 == 0:
                    print(f"epoch {epoch}/{args.epochs}, step {step}/{len(train_loader)}, loss {train_loss / seen:.4f}, elapsed {time.monotonic() - started:.0f}s", flush=True)
            model.eval()
            correct, validation_n, validation_loss = 0, 0, 0.0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for batch in validation_loader:
                    ids, mask, target = [tensor.to(device, non_blocking=True) for tensor in batch]
                    result = model(input_ids=ids, attention_mask=mask, labels=target)
                    correct += int((result.logits.argmax(-1) == target).sum())
                    validation_n += len(target)
                    validation_loss += float(result.loss) * len(target)
            accuracy = correct / validation_n
            selected = accuracy > best_accuracy
            if selected:
                best_accuracy, best_epoch = accuracy, epoch
                model.save_pretrained(args.model_dir, safe_serialization=True)
                tokenizer.save_pretrained(args.model_dir)
            row = {"epoch": epoch, "train_loss": train_loss / seen,
                "validation_loss": validation_loss / validation_n, "validation_correct": correct,
                "validation_examples": validation_n, "validation_accuracy_percent": 100 * accuracy,
                "elapsed_seconds": time.monotonic() - started, "selected": selected}
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
    write_json(args.report_dir / "training_summary.json", {**settings, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "best_epoch": best_epoch,
        "best_validation_accuracy_percent": 100 * best_accuracy,
        "elapsed_seconds": time.monotonic() - started, "history": history,
        "checkpoint_files": {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
