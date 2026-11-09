"""Recount Fake News label errors and compare IMDb word-ranking probe counts."""

import argparse
import ast
from collections import Counter
from inspect import getsource
import json
from pathlib import Path
from textwrap import dedent

from replication_report import verified_run
from replication_runs import (
    ROOT, check_existing, clean_predictions, configure, read, read_released, sha, state_digest,
)


def verified_rows(root, authors_data, dataset):
    directory = root / f"original_bert_{dataset}_full"
    run = verified_run(directory)
    metadata = run["metadata"]
    if metadata["model_alias"] != f"original-bert-{dataset}" or metadata["variant"] != "full":
        raise ValueError(f"Unexpected experiment for {dataset}")
    source, _ = read_released(authors_data / dataset, is_nli=False)
    check_existing(run["rows"], source)
    if len(source) != len(run["rows"]):
        raise ValueError(f"Expected the entire released {dataset} attack file")
    return run["rows"], sha(directory / "summary.json")


def fake_news_baseline(rows, baseline):
    confusion = Counter((row["label"], row["original_prediction"]) for row in rows)
    successes = Counter(row["label"] for row in rows if row["result_type"] == "Successful")
    scores = {}
    for row, clean in zip(rows, baseline, strict=True):
        if any(row[field] != clean[field] for field in ("row", "source_id", "label")):
            raise ValueError("Fake News baseline rows differ from the outcomes")
        if row["original_prediction"] != clean["prediction"]:
            raise ValueError("Fake News baseline predictions differ from the outcomes")
        scores.setdefault(clean["probabilities"][1], Counter())[row["label"]] += 1

    negatives = sum(labels[0] for labels in scores.values())
    positives = len(rows) - negatives
    negatives_below = best_correct = correct = negatives
    wins = 0
    for _, labels in sorted(scores.items(), reverse=True):
        negatives_below -= labels[0]
        wins += labels[1] * (negatives_below + labels[0] / 2)
        correct += labels[1] - labels[0]
        best_correct = max(best_correct, correct)
    return {
        "clean_confusion_rows_gold_columns_predicted": [
            [confusion[label, prediction] for prediction in (0, 1)] for label in (0, 1)
        ],
        "successful_attacks_by_gold_label": {str(label): successes[label] for label in (0, 1)},
        "class_1_score_roc_auc": wins / (positives * negatives),
        "best_correct_same_sample_threshold": best_correct,
        "scope": "The threshold sweep selects and scores on the same rows.",
    }


def imdb_ranking_probes(rows):
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.shared.utils import words_from_text

    # Read the pinned recipe's literal so the diagnostic needs no attack models.
    recipe = ast.parse(dedent(getsource(TextFoolerJin2019.build))).body[0]
    definition = next(
        node.value for node in recipe.body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "stopwords" for target in node.targets
        )
    )
    stopwords = set(ast.literal_eval(definition.args[0]))
    eligible = [row for row in rows if row["label"] == row["original_prediction"]]
    original_probes = textattack_words = textattack_probes = 0
    for row in eligible:
        text = row["original_text"]
        words = words_from_text(text)
        if len(words) != row["original_word_count"]:
            raise ValueError("TextAttack token count differs from the saved run")
        original_probes += len(text.split())
        textattack_words += len(words)
        textattack_probes += sum(word not in stopwords for word in words)
    return {
        "eligible_inputs": len(eligible),
        "authors_ranking_positions": original_probes,
        "textattack_word_positions_before_stopword_filter": textattack_words,
        "textattack_ranking_positions": textattack_probes,
        "mean_ranking_position_difference": (original_probes - textattack_probes) / len(eligible),
        "scope": "Ranking probes on matched eligible inputs. Candidate scoring and physical forward calls are excluded.",
    }


def fake_news_preprocessing(rows, released_file, metadata):
    import torch

    from replication_models import load_model

    wrapper = load_model(ROOT, "original-bert-fake")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.model.to(device).eval()
    model_hash = state_digest(wrapper.model)
    if model_hash != metadata["model_state_sha256"] or wrapper.tokenizer.model_max_length != 256:
        raise ValueError("Checkpoint or token limit differs from the archived Fake News run")

    cleaned, original = [], []
    lines = released_file.read_text(encoding="utf-8").splitlines()
    for row in rows:
        label, text = lines[row["source_id"]].split(" ", 1)
        if int(label) != row["label"]:
            raise ValueError("Released Fake News label differs from the saved row")
        sample = {key: row[key] for key in ("row", "source_id", "label")}
        cleaned.append({**sample, "inputs": {"text": row["original_text"]}})
        original.append({**sample, "inputs": {"text": text.strip().lower()}})
    cleaned_results = clean_predictions(wrapper, cleaned, 2)
    original_results = clean_predictions(wrapper, original, 2)
    if any(r["original_prediction"] != c["prediction"] for r, c in zip(rows, cleaned_results, strict=True)):
        raise ValueError("The repeated cleaned-input baseline differs from the archived run")
    transitions = Counter(
        (c["prediction"] == c["label"], o["prediction"] == o["label"])
        for c, o in zip(cleaned_results, original_results, strict=True)
    )
    return {
        "rows": len(rows),
        "correct_after_attack_cleaner": transitions[True, True] + transitions[True, False],
        "correct_without_attack_cleaner": transitions[True, True] + transitions[False, True],
        "errors_corrected": transitions[False, True],
        "new_errors": transitions[True, False],
        "token_limit": 256,
        "released_input_sha256": sha(released_file),
        "model_state_sha256": model_hash,
        "scope": "Clean inference only. The alternate preprocessing was not used for an attack run.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--authors-data", required=True, type=Path)
    parser.add_argument("--compare-fake-preprocessing", action="store_true")
    args = parser.parse_args()
    configure()
    fake, fake_hash = verified_rows(args.runs, args.authors_data, "fake")
    imdb, imdb_hash = verified_rows(args.runs, args.authors_data, "imdb")
    result = {
        "bert_fake_news": fake_news_baseline(
            fake, read(args.runs / "original_bert_fake_full" / "baseline.json")
        ),
        "bert_imdb": imdb_ranking_probes(imdb),
        "summary_sha256": {"bert_fake_news": fake_hash, "bert_imdb": imdb_hash},
    }
    if args.compare_fake_preprocessing:
        result["bert_fake_news_preprocessing"] = fake_news_preprocessing(
            fake, args.authors_data / "fake", read(args.runs / "original_bert_fake_full" / "metadata.json")
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
