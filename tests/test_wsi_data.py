"""Temporary CPU fixtures: manifest completeness, split safety and lazy decoding."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsi_data import (Camelyon16WSI, collate_one_wsi, discover_fold_ids,
                      load_fold_datasets, load_images)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class WSIDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.labels = [{"case_id": "case_{}".format(i), "slide_id": "slide_{}".format(i),
                        "label": i % 2} for i in range(5)]
        self.label_path = self.root / "downstream_train" / "camelyon16_labels.csv"
        write_csv(self.label_path, ("case_id", "slide_id", "label"), self.labels)
        self.regions = []
        for index in range(5):
            for region in range(2 if index == 1 else 1):
                filename = "slide_{}_r{}.png".format(index, region)
                self.regions.append({"filename": filename, "slide_id": "slide_{}".format(index),
                                     "split": "test"})  # Old SR split is deliberately unrelated.
                for directory in ("images_256", "images_8192"):
                    path = self.root / directory / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"not decoded by Dataset")
        self.manifest = self.root / "manifests" / "patch_manifest.csv"
        write_csv(self.manifest, ("filename", "slide_id", "split"), self.regions)
        for fold in range(5):
            train = [row["case_id"] for i, row in enumerate(self.labels) if i != fold]
            self.write_split(fold, train, ["case_{}".format(fold)])

    def write_split(self, fold, train, val):
        rows = [{"train": train[i] if i < len(train) else "",
                 "val": val[i] if i < len(val) else ""}
                for i in range(max(len(train), len(val)))]
        write_csv(self.root / "downstream_train" / "splits_{}.csv".format(fold),
                  ("train", "val"), rows)

    def test_all_regions_grouped_by_slide_and_split_by_case(self):
        with patch("wsi_data.Image.open", side_effect=AssertionError("Dataset decoded image")):
            train, val = load_fold_datasets(self.root, 0)
        self.assertEqual((len(train), len(val)), (4, 1))
        self.assertEqual(train[0]["slide_id"], "slide_1")
        self.assertEqual(train[0]["n_regions"], 2)
        self.assertEqual(train[0]["label"], 1)
        self.assertEqual([p.name for p in train[0]["lr_paths"]],
                         ["slide_1_r0.png", "slide_1_r1.png"])
        self.assertEqual(val[0]["case_id"], "case_0")
        self.assertIs(collate_one_wsi([train[0]]), train[0])
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            collate_one_wsi([train[0], train[1]])

    def test_missing_region_fails_instead_of_reducing_wsi(self):
        (self.root / "images_8192" / "slide_1_r1.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "all 2 regions required"):
            load_fold_datasets(self.root, 0)

    def test_region_limit_preserves_pairs_and_full_evaluation(self):
        sample = {
            "slide_id": "slide_large", "label": 1, "n_regions": 25,
            "lr_paths": [self.root / "images_256" / f"region_{i}.png" for i in range(25)],
            "hr_paths": [self.root / "images_8192" / f"region_{i}.png" for i in range(25)],
        }
        dataset = Camelyon16WSI([sample], max_regions=24)
        selected = dataset[0]
        self.assertEqual(selected["n_regions"], 24)
        self.assertEqual(len(set(selected["lr_paths"])), 24)
        self.assertEqual([path.name for path in selected["lr_paths"]],
                         [path.name for path in selected["hr_paths"]])
        self.assertEqual(selected["lr_paths"],
                         [path for path in sample["lr_paths"] if path in selected["lr_paths"]])
        self.assertEqual(sample["n_regions"], 25)
        self.assertIs(dataset.evaluation_view()[0], sample)

    def test_sr_off_does_not_require_hr_files(self):
        (self.root / "images_8192" / "slide_1_r1.png").unlink()
        train, val = load_fold_datasets(self.root, 0, require_hr=False)
        self.assertEqual((len(train), len(val)), (4, 1))
        self.assertEqual(train[0]["n_regions"], 2)

    def test_train_val_overlap_rejected(self):
        self.write_split(0, ["case_0", "case_1", "case_2", "case_3", "case_4"], ["case_0"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_fold_datasets(self.root, 0)

    def test_unknown_case_and_omitted_case_rejected(self):
        self.write_split(0, ["case_1", "case_2", "case_3", "unknown"], ["case_0"])
        with self.assertRaisesRegex(ValueError, "unknown cases"):
            load_fold_datasets(self.root, 0)
        self.write_split(0, ["case_1", "case_2", "case_3"], ["case_0"])
        with self.assertRaisesRegex(ValueError, "omits labelled cases"):
            load_fold_datasets(self.root, 0)

    def test_missing_and_invalid_label_rejected(self):
        write_csv(self.label_path, ("case_id", "slide_id", "label"), self.labels[1:])
        with self.assertRaisesRegex(ValueError, "missing real labels"):
            load_fold_datasets(self.root, 0)
        self.labels[0]["label"] = ""
        write_csv(self.label_path, ("case_id", "slide_id", "label"), self.labels)
        with self.assertRaisesRegex(ValueError, "Invalid C16 label"):
            load_fold_datasets(self.root, 0)

    def test_duplicate_region_rejected(self):
        write_csv(self.manifest, ("filename", "slide_id", "split"), self.regions + self.regions[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate region"):
            load_fold_datasets(self.root, 0)

    def test_ambiguous_label_file_requires_explicit_selection(self):
        self.label_path = self.label_path.rename(self.label_path.with_name("first_labels.csv"))
        other = self.label_path.with_name("another_labels.csv")
        write_csv(other, ("case_id", "slide_id", "label"), self.labels)
        with self.assertRaisesRegex(ValueError, "labels-csv explicitly"):
            load_fold_datasets(self.root, 0)
        train, _ = load_fold_datasets(self.root, 0, labels_csv=self.label_path)
        self.assertEqual(len(train), 4)

    def test_available_folds_auto_discovered(self):
        (self.root / "downstream_train" / "splits_3.csv").unlink()
        (self.root / "downstream_train" / "splits_4.csv").unlink()
        self.assertEqual(discover_fold_ids(self.root / "downstream_train"), [0, 1, 2])
        train, _ = load_fold_datasets(self.root, 0)
        self.assertEqual(len(train), 4)

    def test_fivefold_is_explicit_and_preserves_legacy_default(self):
        split_dir = self.root / "downstream_train"
        for fold in range(5):
            val_case = "case_{}".format((fold + 1) % 5)
            train = [row["case_id"] for row in self.labels if row["case_id"] != val_case]
            rows = [{"train": case, "val": val_case if i == 0 else ""}
                    for i, case in enumerate(train)]
            write_csv(split_dir / "splits_5fold_{}.csv".format(fold), ("train", "val"), rows)
        self.assertEqual(discover_fold_ids(split_dir, "5fold"), [0, 1, 2, 3, 4])
        self.assertEqual(discover_fold_ids(split_dir), [0, 1, 2, 3, 4])
        _, old_val = load_fold_datasets(self.root, 0)
        _, new_val = load_fold_datasets(self.root, 0, split_scheme="5fold")
        self.assertEqual(old_val.samples[0]["case_id"], "case_0")
        self.assertEqual(new_val.samples[0]["case_id"], "case_1")
        self.assertEqual(new_val.split_csv.name, "splits_5fold_0.csv")
        with self.assertRaisesRegex(ValueError, "3fold or 5fold"):
            load_fold_datasets(self.root, 0, split_scheme="unknown")

    def test_manifest_split_column_ignored(self):
        rows = []
        for row in self.regions:
            updated = dict(row)
            updated["split"] = "test"
            rows.append(updated)
        write_csv(self.manifest, ("filename", "slide_id", "split"), rows)
        with patch("wsi_data.Image.open", side_effect=AssertionError("Dataset decoded image")):
            train, val = load_fold_datasets(self.root, 0)
        self.assertEqual(len(train), 4)
        self.assertEqual(train[0]["slide_id"], "slide_1")

    def test_load_images_rgb_float32_strict_size(self):
        path = self.root / "real.png"
        Image.new("RGB", (4, 4), (255, 128, 0)).save(path)
        images = load_images([path], 4)
        self.assertEqual(tuple(images.shape), (1, 3, 4, 4))
        self.assertEqual(images.dtype, torch.float32)
        torch.testing.assert_close(images[0, :, 0, 0], torch.tensor([1., 128 / 255., 0.]))
        with self.assertRaisesRegex(ValueError, "no resize allowed"):
            load_images([path], 8)


if __name__ == "__main__":
    unittest.main()
