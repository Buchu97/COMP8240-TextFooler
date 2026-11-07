"""Collect original TMDb review sentences for independent sentiment annotation.

Authentication: TMDB_API_READ_ACCESS_TOKEN or TMDB_API_KEY environment variable.
Run: python collect_tmdb_reviews.py --output ../collected_500 --target 500
Only a new output directory is accepted. Labels are deliberately left blank.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SENTIMENT_CUES = frozenset("""
amazing awful bad beautiful believable boring brilliant charming clever compelling
confusing disappointing dull enjoyable entertaining excellent exciting fantastic
fun funny good great gripping hilarious impressive lovely memorable moving
predictable poor powerful satisfying tedious terrible thoughtful touching ugly
unconvincing underwhelming unforgettable unpleasant weak wonderful worth waste
wastes wasted loved love hate hated recommend disappointing disappointment
superb disappointing pointless ridiculous outstanding bland shallow emotional
engaging uneven heartbreaking delightful annoying absurd impressive disappointing
""".split())
SAMPLE_FIELDS = ["text", "label"]
METADATA_FIELDS = [
    "sample_id", "sample_row", "text_sha256", "source_type", "source_url", "movie_id",
    "movie_title", "review_id", "source_content", "source_content_sha256",
    "source_start", "source_end", "retrieved_at_utc", "extraction_method",
    "label_status", "review_status",
]
REVIEW_FIELDS = [
    "sample_id", "text", "human_label", "review_status", "reviewer_id",
    "reviewed_at", "notes",
]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalised_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def extract_passages(content, min_words=8, max_words=60):
    """Extract exact spans; lexical cues select candidates, never assign labels.

    Sentence boundaries are heuristic. Unclear boundaries can be rejected during
    annotation. Review-level ratings are not used as sentence-level labels.
    """
    boundaries = [0] + [
        m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z<])', content)
    ] + [len(content)]
    passages = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = content[start:end]
        start += len(segment) - len(segment.lstrip())
        end -= len(segment) - len(segment.rstrip())
        text = content[start:end]
        tokens = re.findall(r"\b[\w']+\b", text.lower())
        if not min_words <= len(tokens) <= max_words:
            continue
        if not SENTIMENT_CUES.intersection(tokens):
            continue
        if any(mark in text for mark in ("<", ">", "http://", "https://", "**", "__")):
            continue
        if text.startswith(("=", "+", "-", "@")):
            continue
        passages.append({"text": text, "start": start, "end": end})
    return passages


def collect_samples(fetch, target=500, popular_pages=20, max_per_review=2,
                    max_per_movie=12):
    if target < 1 or popular_pages < 1 or max_per_review < 1 or max_per_movie < 1:
        raise ValueError("Collection limits must be positive")
    rows, seen_texts, seen_reviews, seen_movies = [], set(), set(), set()
    for popular_page in range(1, popular_pages + 1):
        movies = fetch("/movie/popular", popular_page)
        for movie in movies.get("results", []):
            movie_id = movie["id"]
            if movie_id in seen_movies:
                continue
            seen_movies.add(movie_id)
            review_page, accepted_movie = 1, 0
            while accepted_movie < max_per_movie:
                response = fetch(f"/movie/{movie_id}/reviews", review_page)
                retrieved_at = utc_now()
                for review in response.get("results", []):
                    review_id = review.get("id", "")
                    if not review_id or review_id in seen_reviews:
                        continue
                    seen_reviews.add(review_id)
                    content = review.get("content", "")
                    accepted_review = 0
                    for passage in extract_passages(content):
                        key = normalised_key(passage["text"])
                        if key in seen_texts:
                            continue
                        seen_texts.add(key)
                        rows.append({
                            "text": passage["text"], "label": "",
                            "movie_id": movie_id, "movie_title": movie.get("title", ""),
                            "review_id": review_id,
                            "source_url": review.get("url") or
                                f"https://www.themoviedb.org/review/{review_id}",
                            "source_content": content,
                            "source_start": passage["start"], "source_end": passage["end"],
                            "retrieved_at_utc": retrieved_at,
                        })
                        accepted_movie += 1
                        accepted_review += 1
                        if len(rows) == target:
                            return rows
                        if accepted_review == max_per_review or accepted_movie == max_per_movie:
                            break
                    if accepted_movie == max_per_movie:
                        break
                if review_page >= response.get("total_pages", 1):
                    break
                review_page += 1
        if popular_page >= movies.get("total_pages", 1):
            break
    return rows


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(rows, destination, target=500):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    metadata, reviews = [], []
    for index, row in enumerate(rows, 1):
        sample_id = f"TMDB{index:04d}"
        metadata.append({
            "sample_id": sample_id, "sample_row": index,
            "text_sha256": sha256(row["text"]), "source_type": "tmdb_review_sentence",
            **{field: row[field] for field in (
                "source_url", "movie_id", "movie_title", "review_id", "source_content",
                "source_start", "source_end", "retrieved_at_utc",
            )},
            "source_content_sha256": sha256(row["source_content"]),
            "extraction_method": "exact_sentence_span_8_60_words_sentiment_cue_filter_v1",
            "label_status": "unlabelled", "review_status": "pending",
        })
        reviews.append({"sample_id": sample_id, "text": row["text"],
                        "human_label": "", "review_status": "pending",
                        "reviewer_id": "", "reviewed_at": "", "notes": ""})
    write_csv(destination / "samples.csv", SAMPLE_FIELDS,
              [{"text": r["text"], "label": ""} for r in rows])
    write_csv(destination / "sample_metadata.csv", METADATA_FIELDS, metadata)
    write_csv(destination / "human_review.csv", REVIEW_FIELDS, reviews)
    facts = {
        "dataset_category": "source_derived_movie_review_sentences",
        "source": "TMDb API movie reviews",
        "target_samples": target, "collected_samples": len(rows),
        "collection_status": "target_collected" if len(rows) == target else "partial",
        "created_at_utc": utc_now(), "human_verified_samples": 0,
        "label_mapping": "0=negative;1=positive;blank=not_assigned",
        "label_status": "pending_independent_human_annotation",
        "selection": "popular_movies;English_API_language;lexical_sentiment_cues",
        "selection_limitations": "convenience_sample;English_not_language_verified;cue_filter_bias",
        "review_level_ratings_used_as_labels": "false",
        "split": "evaluation_candidate_pool_no_training_split",
        "source_api": "https://developer.themoviedb.org/reference/movie-reviews",
        "attribution": "This product uses the TMDB API but is not endorsed or certified by TMDB.",
        "public_redistribution_rights": "not_confirmed",
    }
    write_csv(destination / "dataset_metadata.csv", ["field", "value"],
              [{"field": key, "value": value} for key, value in facts.items()])


def api_fetcher():
    token = os.environ.get("TMDB_API_READ_ACCESS_TOKEN")
    key = os.environ.get("TMDB_API_KEY")
    if not token and not key:
        raise RuntimeError("TMDB_API_READ_ACCESS_TOKEN or TMDB_API_KEY is not configured locally")

    def fetch(endpoint, page):
        params = {"language": "en-US", "page": page}
        headers = {"accept": "application/json", "User-Agent": "COMP8240-review-collection/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            params["api_key"] = key
        request = Request("https://api.themoviedb.org/3" + endpoint + "?" + urlencode(params),
                          headers=headers)
        for attempt in range(4):
            try:
                with urlopen(request, timeout=20) as response:
                    result = json.load(response)
                time.sleep(0.15)
                return result
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f"TMDb HTTP {error.code} at {endpoint}") from None
                time.sleep(min(2 ** attempt, 8))
            except URLError:
                if attempt == 3:
                    raise RuntimeError(f"TMDb network request failed at {endpoint}") from None
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("TMDb request attempts exhausted")
    return fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--popular-pages", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    try:
        rows = collect_samples(api_fetcher(), args.target, args.popular_pages)
        write_dataset(rows, args.output, args.target)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Collected {len(rows)}/{args.target} passages; sentiment labels await human annotation.")
    if len(rows) < args.target:
        parser.exit(2, "Partial collection saved; target count not reached.\n")


if __name__ == "__main__":
    main()
