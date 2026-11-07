"""Convert the pinned official SciBERT encoder to a local safetensors checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from type1_data import sha256, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "allenai/scibert_scivocab_uncased"
    revision = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
    raw = root / "data/raw/type1_additions_20261102/scibert"
    output = root / "models/scibert_base_20261102"
    report = root / "outputs/type1_ADDITIONS_20261102/scibert_base_preparation.json"
    existing = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
    if output.exists():
        if existing is None:
            raise SystemExit("Model directory exists without its preparation report")
        for filename, digest in existing["output_files"].items():
            if sha256(output / filename) != digest:
                raise ValueError(f"Prepared encoder file hash mismatch: {filename}")
        print("Pinned SciBERT encoder already prepared and verified.", flush=True)
        return
    raw.mkdir(parents=True, exist_ok=True)
    import requests
    import torch
    from transformers import BertConfig, BertModel, BertTokenizer
    files = []
    for filename in ("config.json", "vocab.txt", "pytorch_model.bin"):
        url = f"https://huggingface.co/{source}/resolve/{revision}/{filename}"
        path = raw / filename
        if not path.exists():
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with path.open("xb") as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        handle.write(chunk)
        if existing:
            expected = next(row for row in existing["source_files"] if row["url"] == url)
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Pinned SciBERT source hash mismatch: {filename}")
        files.append({"url": url, "sha256": sha256(path), "bytes": path.stat().st_size})
        print(f"Downloaded {filename}: {path.stat().st_size} bytes", flush=True)
    # Tensor-only loading; no code from the upstream checkpoint is executed.
    state = torch.load(raw / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    config = BertConfig.from_json_file(str(raw / "config.json"))
    model = BertModel(config)
    if any(key.startswith("bert.") for key in state):
        encoder_state = {key.removeprefix("bert."): tensor for key, tensor in state.items() if key.startswith("bert.")}
        discarded = sorted(key for key in state if not key.startswith("bert."))
    else:
        encoder_state, discarded = state, []
    model.load_state_dict(encoder_state, strict=True)
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    BertTokenizer(vocab_file=str(raw / "vocab.txt"), do_lower_case=True).save_pretrained(output)
    destination = output / "preparation.json" if existing else report
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {"source": source, "revision": revision,
        "source_files": files, "tensor_only_loading": True,
        "encoder_state_loaded_strictly": True,
        "discarded_pretraining_head_parameters": discarded,
        "output": output.relative_to(root).as_posix(),
        "output_files": {path.name: sha256(path) for path in output.iterdir() if path.is_file()},
        "task_fine_tuning_performed": False})
    print(json.dumps({"status": "prepared", "model": source, "revision": revision}), flush=True)


if __name__ == "__main__":
    main()
