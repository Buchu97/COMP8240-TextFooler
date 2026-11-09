"""Count the effect of POS/USE ordering along eight archived IMDb attacks."""

import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import time


PROJECT = Path(__file__).resolve().parents[1]
ROW_IDS = (63, 192, 313, 436, 567, 692, 817, 938)
STAGES = {
    "WordEmbeddingDistance": "embedding_rejected",
    "PartOfSpeech": "pos_rejected",
    "UniversalSentenceEncoder": "use_rejected_after_pos",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def snapshot(archive, name, filename, expected_hash):
    path = archive / filename
    require(sha(path) == expected_hash, f"Snapshot digest mismatch for {filename}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_counters(attack, counts):
    search = attack.search_method
    goal = attack.goal_function
    get_index_order = search._get_index_order
    get_results = goal.get_results
    filter_transformations = attack.filter_transformations
    ranking = False

    def count_index_order(*args, **kwargs):
        nonlocal ranking
        ranking = True
        try:
            return get_index_order(*args, **kwargs)
        finally:
            ranking = False

    def count_results(attacked_text_list, check_skip=False):
        key = "initial_queries" if check_skip else "ranking_queries" if ranking else "candidate_queries"
        before = goal.num_queries
        results = get_results(attacked_text_list, check_skip=check_skip)
        delta = goal.num_queries - before
        require(delta == len(attacked_text_list), "Unexpected query truncation")
        counts[key] += delta
        return results

    def count_filter(transformed, current, original_text=None):
        counts["raw_candidates"] += len(transformed)
        for candidate in transformed:
            if candidate.text == current.text:
                counts["unchanged_candidates"] += 1
            elif (current, candidate) in attack.constraints_cache:
                raise RuntimeError("Cannot attribute a constraint-cache hit to its filter stage")
        filtered = filter_transformations(transformed, current, original_text)
        counts["filter_survivors"] += len(filtered)
        return filtered

    def count_constraint(call_many, key):
        def count(candidates, reference):
            filtered = call_many(candidates, reference)
            counts[key] += len(candidates) - len(filtered)
            return filtered

        return count

    require([type(c).__name__ for c in attack.constraints] == list(STAGES), "Unexpected constraints")
    require(search.wir_method == "delete" and goal.query_budget == float("inf"), "Unexpected search or query budget")
    search._get_index_order = count_index_order
    search.get_goal_results = goal.get_results = count_results
    attack.filter_transformations = count_filter
    for constraint, key in zip(attack.constraints, STAGES.values(), strict=True):
        constraint.call_many = count_constraint(constraint.call_many, key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    args = parser.parse_args()
    archive, evaluation, project = args.run_dir, args.sample_dir, args.project_root.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    metadata = read_json(archive / "metadata.json")
    summary = read_json(archive / "summary.json")
    for filename, expected_hash in (
        (archive / "metadata.json", summary["metadata_sha256"]),
        (archive / "outcomes.csv", summary["outcomes_sha256"]),
        (evaluation / "manifest.json", metadata["sample_manifest_sha256"]),
        (evaluation / "evaluation.json", metadata["evaluation_sha256"]),
    ):
        require(sha(filename) == expected_hash, f"Archive digest mismatch for {filename.name}")
    with (archive / "outcomes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    samples = read_json(evaluation / "evaluation.json")
    eligible = [int(row["row"]) for row in rows if row["result_type"] != "Skipped"]
    n = len(ROW_IDS)
    require(len(eligible) == 909, "Unexpected eligible sample size")
    require(tuple(eligible[(2 * i + 1) * len(eligible) // (2 * n)] for i in range(n)) == ROW_IDS,
            "Midpoint sample changed")

    runner = snapshot(archive, "archived_runner", "runner_snapshot.py", metadata["runner_sha256"])
    runner.ROOT = project
    runner.configure()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name, expected in metadata["packages"].items():
        require(version(name) == expected, f"Package version differs for {name}")
    import tensorflow_hub as hub
    import textattack
    import torch

    recipe = snapshot(archive, "archived_recipe", "recipe_snapshot.py", metadata["recipe_sha256"])
    loader = snapshot(archive, "archived_loader", "model_loader_snapshot.py", metadata["model_loader_sha256"])
    require(torch.cuda.is_available(), "The archived CUDA run requires local CUDA for this replay")
    torch.set_num_threads(8)
    textattack.shared.utils.set_seed(metadata["seed"])
    wrapper = loader.load_model(project, metadata["model_alias"])
    wrapper.model.to("cuda").eval()
    require(runner.state_digest(wrapper.model) == metadata["model_state_sha256"], "Checkpoint state differs")
    require(wrapper.tokenizer.model_max_length == metadata["tokenizer_max_length"], "Token limit differs")
    attack = recipe.Attack(wrapper)
    require(repr(attack).strip() == (archive / "attack_configuration.txt").read_text(encoding="utf-8").strip(),
            "Attack configuration differs from archive")

    use = attack.constraints[-1]
    cache = project / "data/cache/textattack"
    module_key = hashlib.sha1(use._tfhub_url.encode()).hexdigest()
    descriptor = (cache / f"{module_key}.descriptor.txt").read_text(encoding="utf-8")
    require(descriptor.splitlines()[0] == f"Module: {use._tfhub_url}", "Cached USE module differs")
    use.model = hub.load(str(cache / module_key))

    setup_seconds = time.monotonic() - started
    print(f"Setup {setup_seconds:.2f}s", flush=True)
    counts = Counter()
    install_counters(attack, counts)
    results = []
    for index in ROW_IDS:
        sample, archived = samples[index], rows[index]
        require(sample["row"] == int(archived["row"]) == index, "Row mismatch")
        require(sample["source_id"] == int(archived["source_id"]), "Source ID mismatch")
        require(sample["label"] == int(archived["label"]), "Label mismatch")
        inputs = OrderedDict((name, sample["inputs"][name]) for name in metadata["input_columns"])
        textattack.shared.utils.set_seed(metadata["seed"] + index)
        counts.clear()
        row_started = time.monotonic()
        result = attack.attack(inputs, sample["label"])
        original = result.original_result.attacked_text
        perturbed = result.perturbed_result.attacked_text
        values = {
            "original_text": original.text,
            "perturbed_text": perturbed.text,
            "original_prediction": int(result.original_result.output),
            "perturbed_prediction": int(result.perturbed_result.output),
            "result_type": type(result).__name__.removesuffix("AttackResult"),
            "num_queries": result.num_queries,
            "original_word_count": original.num_words,
            "changed_word_count": original.words_diff_num(perturbed),
        }
        mismatches = [key for key, value in values.items() if str(value) != archived[key]]
        require(not mismatches, f"Row {index} differs from archive in {mismatches}")
        require(counts["initial_queries"] == 1, "Wrong initial-query count")
        require(counts["initial_queries"] + counts["ranking_queries"] + counts["candidate_queries"]
                == result.num_queries, "Logical-query counters do not sum to archive count")
        require(counts["candidate_queries"] == counts["filter_survivors"], "Substitutions do not reconcile")
        require(counts["candidate_queries"] + counts["unchanged_candidates"]
                + sum(counts[key] for key in STAGES.values()) == counts["raw_candidates"],
                "Pre-filter candidates do not reconcile")
        counts["ordering_only_extra_candidates"] = counts["pos_rejected"] + counts["use_rejected_after_pos"]
        counts["post_embedding_candidates"] = counts["candidate_queries"] + counts["ordering_only_extra_candidates"]
        counts["released_ranking_probes"] = len(original.text.split())
        counter_fields = ("initial_queries", "ranking_queries", "candidate_queries", "raw_candidates",
                          "unchanged_candidates", "filter_survivors", *STAGES.values(),
                          "ordering_only_extra_candidates", "post_embedding_candidates", "released_ranking_probes")
        row = {"row": index, "result_type": values["result_type"], "archived_queries": result.num_queries,
               **{key: counts[key] for key in counter_fields}, "attack_seconds": time.monotonic() - row_started}
        results.append(row)
        print(f"Row {index} matched in {row['attack_seconds']:.2f}s", flush=True)

    totals = {key: sum(row[key] for row in results) for key in ("archived_queries", *counter_fields)}
    output = {"matched_rows": len(results), "setup_seconds": setup_seconds,
              "wall_seconds": time.monotonic() - started, "totals": totals, "rows": results}
    output["scope"] = "Counterfactual scorer ordering with proposal batches and attack steps held fixed."
    output["reference_outcomes_sha256"] = summary["outcomes_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
