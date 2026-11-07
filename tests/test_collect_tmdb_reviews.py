import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_tmdb_reviews import collect_samples, extract_passages, write_dataset


class PassageTests(unittest.TestCase):
    def test_passages_keep_exact_source_offsets(self):
        content = (
            "A note. The actors make every quiet conversation feel warm and believable. "
            "The film was released in the summer of that year. "
            "Unfortunately, the ending is dull and wastes the promising opening."
        )
        passages = extract_passages(content)
        self.assertEqual([p["text"] for p in passages], [
            "The actors make every quiet conversation feel warm and believable.",
            "Unfortunately, the ending is dull and wastes the promising opening.",
        ])
        for p in passages:
            self.assertEqual(content[p["start"]:p["end"]], p["text"])

    def test_short_neutral_markup_and_oversized_passages_are_excluded(self):
        content = "Great! The movie was filmed in a town near the coast. "
        content += "<b>The wonderful actors deliver a fantastic and memorable performance.</b> "
        content += "The wonderful film " + "excellent " * 70 + "ends."
        self.assertEqual(extract_passages(content), [])

    def test_duplicate_passages_and_review_ratings_do_not_become_labels(self):
        sentence = "The excellent cast makes this gentle film a pleasure to watch."
        calls = []

        def fetch(endpoint, page):
            calls.append((endpoint, page))
            if endpoint == "/movie/popular":
                return {"total_pages": 1, "results": [{"id": 42, "title": "Fixture"}]}
            return {"total_pages": 1, "results": [
                {"id": "a", "content": sentence, "author_details": {"rating": 10},
                 "url": "https://www.themoviedb.org/review/a"},
                {"id": "b", "content": sentence.upper(), "author_details": {"rating": 1},
                 "url": "https://www.themoviedb.org/review/b"},
            ]}

        rows = collect_samples(fetch, target=500, popular_pages=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "")
        self.assertEqual(rows[0]["review_id"], "a")
        self.assertEqual(rows[0]["text"], sentence)
        self.assertEqual(calls, [("/movie/popular", 1), ("/movie/42/reviews", 1)])

    def test_target_count_is_a_hard_cap(self):
        def fetch(endpoint, page):
            if endpoint == "/movie/popular":
                return {"total_pages": 1, "results": [{"id": 9, "title": "Fixture"}]}
            return {"total_pages": 1, "results": [
                {"id": "x", "content": (
                    "The wonderful performances make the entire film worth seeing again. "
                    "The beautiful ending brings a satisfying conclusion to the story."
                )}
            ]}
        self.assertEqual(len(collect_samples(fetch, target=1, popular_pages=1)), 1)


class ExportTests(unittest.TestCase):
    def test_csv_schemas_empty_labels_and_no_overwrite(self):
        row = {
            "text": "The lovely performances make this charming film worth watching again.",
            "label": "", "movie_id": 7, "movie_title": "Fixture", "review_id": "r",
            "source_url": "https://www.themoviedb.org/review/r", "source_content": "x",
            "source_start": 0, "source_end": 1, "retrieved_at_utc": "2026-10-06T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "collected"
            write_dataset([row], dest, target=500)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), [
                "dataset_metadata.csv", "human_review.csv", "sample_metadata.csv", "samples.csv"
            ])
            with (dest / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(samples, [{"text": row["text"], "label": ""}])
            with (dest / "human_review.csv").open(encoding="utf-8-sig", newline="") as stream:
                review = list(csv.DictReader(stream))[0]
            self.assertEqual(review["human_label"], "")
            self.assertEqual(review["review_status"], "pending")
            with self.assertRaises(FileExistsError):
                write_dataset([row], dest, target=500)


if __name__ == "__main__":
    unittest.main()
