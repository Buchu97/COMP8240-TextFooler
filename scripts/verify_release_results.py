"""Reconcile the published aggregates with the report's denominators."""

import csv
import json
import math
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8-sig"))


def require(condition, description):
    if not condition:
        raise ValueError(f"Release check failed: {description}")


def check_attacks(rows):
    for row in rows:
        examples, eligible, successful, failed, skipped = (
            int(row[key]) for key in ("examples", "eligible", "successful", "failed", "skipped")
        )
        require(eligible == successful + failed and examples == eligible + skipped,
                f"{row['dataset']} attack counts")
        rates = {
            "original_accuracy_percent": 100 * eligible / examples,
            "after_attack_accuracy_percent": 100 * failed / examples,
            "attack_success_rate_percent": 100 * successful / eligible,
        }
        for field, expected in rates.items():
            require(math.isclose(float(row[field]), expected), f"{row['dataset']} {field}")


def check_judges(phase, rows, diagnostics):
    require(set(diagnostics) == {row["dataset"] for row in rows}, f"{phase} judge datasets")
    for row in rows:
        judged = int(row["judged"])
        counts = sum(int(row[key]) for key in ("accepted", "rejected", "uncertain"))
        require(counts == judged, f"{row['dataset']} judge counts")
        require(math.isclose(100 * int(row["accepted"]) / judged, float(row["judge_acceptance_percent"])),
                f"{row['dataset']} judge acceptance")
        check = diagnostics[row["dataset"]]
        agreement = check["paired_vs_single_label_preservation"]
        require(agreement["n"] + agreement["unresolved"] == judged, f"{row['dataset']} judge agreement population")
        require(check["uncertain_from_original_label"] <= int(row["uncertain"]),
                f"{row['dataset']} judge uncertainty")


def check_data_quality():
    quality = read_json("type2_data_quality_summary.json")
    clarity = quality["tmdb_full_clarity_screen"]
    counts = clarity["counts"]
    subsets = clarity["saved_attack_outcomes_by_clarity"]
    require(sum(counts.values()) == clarity["previous_rows"] + clarity["additional_rows"] == 412,
            "TMDb clarity population")
    require(subsets["clear"]["passages"] == counts["clear"], "clear TMDb subset size")
    for group in subsets.values():
        require(group["eligible"] == group["successful"] + group["failed"], "TMDb clarity eligible")
        require(group["passages"] == group["eligible"] + group["skipped"], "TMDb clarity passages")
        require(math.isclose(group["attack_success_rate_percent"], 100 * group["successful"] / group["eligible"]),
                "TMDb clarity attack success rate")
    require(sum(group["successful"] for group in subsets.values()) == 322, "TMDb clarity success total")
    rates = subsets["clear"]["outcomes_by_original_polarity"].values()
    macro = 50 * sum(group["successful"] / group["eligible"] for group in rates)
    require(math.isclose(macro, subsets["clear"]["equal_polarity_attack_success_rate_percent"]),
            "TMDb polarity-weighted rate")

    synthetic = quality["synthetic_blinded_rereview"]
    table = synthetic["contingency"]
    require(sum(row["count"] for row in table) == synthetic["sample_size"], "synthetic re-review population")
    agreed = sum(row["count"] for row in table if row["reviewer_1"] == row["reviewer_2"])
    require(agreed == synthetic["matched_between_reviewers"], "synthetic re-review agreement count")
    require(math.isclose(100 * agreed / synthetic["sample_size"], synthetic["agreement_percent"]),
            "synthetic re-review agreement rate")
    construction = read_json("type2_synthetic_construction.json")
    categories = construction["categories"]
    require(len(categories) == 10 and all(row["negative"] == row["positive"] == 25 for row in categories.values()),
            "synthetic design balance")
    require(sum(sum(row.values()) for row in categories.values()) == construction["frozen_rows"] == 500,
            "synthetic frozen population")


def check_query_replay():
    replay = read_json("imdb_query_stage_replay.json")
    rows, totals = replay["rows"], replay["totals"]
    require(replay["matched_rows"] == len(rows) == len({row["row"] for row in rows}) == 8,
            "IMDb replay population")
    for field, count in totals.items():
        require(count == sum(row[field] for row in rows), f"IMDb replay {field}")
    require(totals["archived_queries"] ==
            totals["initial_queries"] + totals["ranking_queries"] + totals["candidate_queries"],
            "IMDb archived query partition")
    filtered = totals["pos_rejected"] + totals["use_rejected_after_pos"]
    require(filtered == totals["ordering_only_extra_candidates"] and
            totals["post_embedding_candidates"] == filtered + totals["candidate_queries"],
            "IMDb substitution filter accounting")


def main():
    original = read_csv("original_replication.csv")
    require(len(original) == 18, "original comparisons")
    require(sum(float(row["clean_difference_points"]) == 0 for row in original) == 14, "original clean matches")
    for name in ("type1_attacks.csv", "type2_final_attacks.csv"):
        check_attacks(read_csv(name))
    diagnostics = read_json("judge_protocol_checks.json")
    check_judges("Type 1", read_csv("type1_judge.csv"), diagnostics["type1"]["datasets"])
    check_judges("Type 2", read_csv("type2_final_judge.csv"), diagnostics["type2"]["datasets"])
    check_data_quality()
    check_query_replay()
    print("Verified the published replication, attack, judge, and data-quality aggregates.")


if __name__ == "__main__":
    main()
