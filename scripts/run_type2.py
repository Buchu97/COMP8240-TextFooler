"""Preflight or evaluate a prepared Type2 set with the existing fixed-set runner.

--dry-run reads and validates inputs/assets without model imports or attacks.
All live inference uses the pinned local snapshot with local_files_only=True.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from prepare_type2_evaluation import ROOT, validate_prepared, verify_local_assets


def load_type2_classifier(manifest, local_model=None):
    """Adapt verified generic BERT output IDs to the human polarity contract."""
    if local_model is not None:
        raise ValueError('Type2 checkpoint overrides are not supported')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from textattack.models.wrappers import HuggingFaceModelWrapper
    path = ROOT / manifest['model']['snapshot_path']
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False)
    if model.config.num_labels != 2 or model.config.id2label != {0: 'LABEL_0', 1: 'LABEL_1'}:
        raise ValueError('Sentiment checkpoint output contract changed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    anchors = ['This was a dreadful movie. I hated every minute of it.',
               'This was a wonderful movie. I loved every minute of it.']
    if wrapper(anchors).argmax(dim=-1).cpu().tolist() != [0, 1]:
        raise ValueError('Runtime negative=0/positive=1 anchor check failed')
    model.config.id2label = {0: 'negative', 1: 'positive'}
    model.config.label2id = {'negative': 0, 'positive': 1}
    return model, tokenizer, device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(f'Output directory already exists: {args.output_dir}')
        manifest = validate_prepared(args.dataset_dir, allow_partial=args.allow_partial)
        config = verify_local_assets()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Type2 preflight rejected: {exc}', file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(dict(status='preflight_passed_no_attack_executed',
            dataset=manifest['dataset'], examples=manifest['evaluation']['examples'],
            model=config['model'], attack=config['attack'],
            review_min_reviewers=manifest['review_provenance']['min_reviewers'],
            partial_pilot=manifest['partial_pilot'], review_counts=manifest['review_provenance']['counts'],
            output_dir=str(args.output_dir.resolve())), ensure_ascii=False, allow_nan=False))
        return 0
    for name, folder in {'TA_CACHE_DIR': 'textattack', 'HF_HOME': 'huggingface',
            'TORCH_HOME': 'torch', 'NLTK_DATA': 'nltk', 'TFHUB_CACHE_DIR': 'tensorflow_hub'}.items():
        os.environ[name] = str(ROOT / 'data/cache' / folder)
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', PYTHONIOENCODING='utf-8')
    import run_type1
    original_loader, original_argv = run_type1.load_classifier, sys.argv
    try:
        # This adapter changes model loading only; the existing runner owns the
        # fixed input order, baseline, attack recipe, metrics and output schema.
        run_type1.load_classifier = load_type2_classifier
        sys.argv = [str(ROOT / 'scripts/run_type1.py'), '--dataset-dir', str(args.dataset_dir),
            '--output-dir', str(args.output_dir), '--seed', str(config['attack']['seed']),
            '--max-length', str(config['model']['max_length'])]
        run_type1.main()
    finally:
        run_type1.load_classifier, sys.argv = original_loader, original_argv
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
