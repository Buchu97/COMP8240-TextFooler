import importlib.util
import sys
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts/type1_data.py"


class Type1DataTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), "Type 1 dataset adapter is missing")
        spec = importlib.util.spec_from_file_location("type1_data", PATH)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def test_poem_preserves_text_and_maps_label_by_name(self):
        records = self.module.convert_rows("poem", [
            {"id": 9, "verse_text": "  Bright dawn,\nagain. ", "label": 1}],
            ["negative", "positive", "no_impact", "mixed"])
        self.assertEqual(records[0]["text"], "  Bright dawn,\nagain. ")
        self.assertEqual(records[0]["label_name"], "positive")
        self.assertEqual(records[0]["source_id"], "9")
        mapped = self.module.map_labels(records, {"positive": 0, "negative": 1,
                                                 "mixed": 2, "no_impact": 3})
        self.assertEqual(mapped[0]["label"], 0)

    def test_clinc_excludes_oos_inputs_without_collapsing_intents(self):
        records = self.module.convert_rows("clinc150", [
            {"text": "book a flight", "intent": 0},
            {"text": "where is my dog", "intent": 1}], ["book_flight", "oos"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["label_name"], "book_flight")

    def test_news_join_is_exact_and_keeps_source_link(self):
        records = self.module.convert_rows("news", [{"headline": "A headline",
            "short_description": "More detail.", "category": "WORLD NEWS",
            "link": "https://example.org/a", "date": "2022-01-01"}], [])
        self.assertEqual(records[0]["text"], "A headline More detail.")
        self.assertEqual(records[0]["label_name"], "WORLD NEWS")
        self.assertEqual(records[0]["source_id"], "https://example.org/a")

    def test_empty_news_rows_are_excluded_without_shifting_source_row_ids(self):
        rows = [{"headline": "", "short_description": "", "category": "A", "link": "empty"},
                {"headline": "News", "short_description": "", "category": "B", "link": "valid"}]
        records = self.module.convert_rows("news", rows, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_row"], 1)

    def test_news_duplicate_and_conflicting_texts_are_removed_before_split(self):
        records = [{"source_id": str(i), "text": text, "label_name": label}
                   for i, (text, label) in enumerate([
                       ("Same text", "A"), ("same  TEXT", "A"),
                       ("Unclear", "A"), ("unclear", "B"),
                       ("Unique", "B")])]
        clean, report = self.module.deduplicate_news(records)
        self.assertEqual([r["source_id"] for r in clean], ["0", "4"])
        self.assertEqual(report["duplicate_rows_removed"], 1)
        self.assertEqual(report["conflicting_rows_removed"], 2)

    def test_stratified_sample_is_fixed_and_covers_each_class(self):
        records = [{"source_id": str(i), "text": str(i), "label_name": str(i // 10)}
                   for i in range(30)]
        sample = self.module.stratified_sample(records, 8, 8240)
        self.assertEqual(len(sample), 8)
        self.assertEqual({r["label_name"] for r in sample}, {"0", "1", "2"})
        self.assertEqual(sample, self.module.stratified_sample(records, 8, 8240))
        self.assertEqual(len({r["source_id"] for r in sample}), 8)
        with self.assertRaises(ValueError):
            self.module.stratified_sample(records, 31, 8240)

    def test_news_splits_are_disjoint_and_keep_all_classes(self):
        records = [{"source_id": str(i), "text": str(i), "label_name": str(i // 20)}
                   for i in range(60)]
        splits = self.module.split_news(records, 8240)
        self.assertEqual({k: len(v) for k, v in splits.items()},
                         {"train": 48, "validation": 6, "test": 6})
        ids = [{r["source_id"] for r in splits[k]} for k in splits]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertTrue(all({r["label_name"] for r in rows} == {"0", "1", "2"}
                            for rows in splits.values()))

    def test_invalid_text_and_missing_label_mapping_are_rejected(self):
        with self.assertRaises(ValueError):
            self.module.convert_rows("poem", [{"id": 0, "verse_text": " ", "label": 0}],
                                     ["negative"])
        with self.assertRaises(ValueError):
            self.module.map_labels([{"text": "test", "label_name": "missing"}], {"x": 0})


if __name__ == "__main__":
    unittest.main()
