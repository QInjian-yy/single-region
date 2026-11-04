import csv
import hashlib
import tempfile
import unittest
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from scripts.create_fivefold_splits import (existing_participants, make_folds,
                                          validate_labels, write_folds)


class FiveFoldTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"case_id": f"{kind}_{i:03}", "slide_id": f"{kind}_{i:03}", "label": label}
            for kind, label in (("normal", "0"), ("tumor", "1"))
            for i in range(1, 11)
        ]

    def test_exact_stratified_split_and_no_overwrite(self):
        cases = [row["case_id"] for row in self.rows]
        labels = [int(row["label"]) for row in self.rows]
        expected = [(sorted(cases[i] for i in train), sorted(cases[i] for i in val))
                    for train, val in StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    .split(cases, labels)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            label_path = root / "camelyon16_labels.csv"
            with label_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("case_id", "slide_id", "label"))
                writer.writeheader()
                writer.writerows(self.rows)
            for fold in range(3):
                val = cases[fold::3]
                train = [case for case in cases if case not in val]
                with (root / f"splits_{fold}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("train", "val"))
                    writer.writerows(zip_longest(train, val, fillvalue=""))
            original = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in root.glob("*.csv")}
            validate_labels(self.rows)
            self.assertEqual(existing_participants(root, self.rows), set(cases))
            folds, grouped = make_folds(self.rows)
            self.assertFalse(grouped)
            self.assertEqual(folds, expected)
            write_folds(root, folds)
            self.assertEqual(len(list(root.glob("splits_5fold_*.csv"))), 5)
            self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
                                for path, digest in original.items()))
            before = (root / "splits_5fold_0.csv").read_bytes()
            with self.assertRaises(FileExistsError):
                write_folds(root, folds)
            self.assertEqual((root / "splits_5fold_0.csv").read_bytes(), before)

    def test_grouped_cases_do_not_cross_folds_and_test_ids_are_rejected(self):
        self.rows[1]["case_id"] = self.rows[0]["case_id"]
        validate_labels(self.rows)
        folds, grouped = make_folds(self.rows)
        self.assertTrue(grouped)
        self.assertTrue(all(not set(train) & set(val) for train, val in folds))
        self.assertEqual(Counter(case for _, val in folds for case in val),
                         Counter({row["case_id"]: 1 for row in self.rows}))
        self.rows[0]["slide_id"] = "test_001"
        with self.assertRaisesRegex(ValueError, "training slide"):
            validate_labels(self.rows)

    def test_packaged_native512_folds_match_server_label_snapshot(self):
        excluded = {16, 19, 27, 42, 45, 86, 144}
        rows = ([{"case_id": f"normal_{i:03}", "slide_id": f"normal_{i:03}", "label": "0"}
                 for i in range(1, 161) if i not in excluded] +
                [{"case_id": f"tumor_{i:03}", "slide_id": f"tumor_{i:03}", "label": "1"}
                 for i in range(1, 112)])
        cases = [row["case_id"] for row in rows]
        labels = [int(row["label"]) for row in rows]
        expected = [(sorted(cases[i] for i in train), sorted(cases[i] for i in val))
                    for train, val in StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    .split(cases, labels)]
        directory = Path(__file__).resolve().parents[1] / "downstream_train/native512_264"
        validation_counts = Counter()
        for fold, (train, val) in enumerate(expected):
            with (directory / f"splits_5fold_{fold}.csv").open(newline="", encoding="utf-8") as handle:
                actual = list(csv.DictReader(handle))
            self.assertEqual([row["train"] for row in actual if row["train"]], train)
            self.assertEqual([row["val"] for row in actual if row["val"]], val)
            self.assertFalse(set(train) & set(val))
            self.assertEqual(set(train) | set(val), set(cases))
            validation_counts.update(val)
        self.assertEqual(validation_counts, Counter({case: 1 for case in cases}))


if __name__ == "__main__":
    unittest.main()
