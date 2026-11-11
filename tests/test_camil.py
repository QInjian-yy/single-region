import csv
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
import yaml

from camil_data import FOLDS, MANIFESTS, PARTITIONS, load_camil_samples, make_camil_datasets, read_camil_protocol
from models.shared_resnet_sr import model_from_checkpoint
from native_data import NATIVE_COLUMNS
import train_e2e_camil as camil

ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class CamilTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.protocol = self.root / "protocol"
        self.protocol.mkdir()
        self.parts = [[f"slide_{i}_{label}" for label in (0, 1)] for i in FOLDS]
        slides = {
            slide_id: {
                "source": tuple(MANIFESTS)[(i + label) % 2], "label": label,
                "group_id": slide_id, "n_regions": 2,
            }
            for i, part in enumerate(self.parts) for label, slide_id in enumerate(part)
        }
        self.roots = {}
        sources = {}
        for name, rel in MANIFESTS.items():
            root = self.roots[name] = self.root / name
            manifest, report = root / rel, root / "verification/report.json"
            for directory in (manifest.parent, report.parent, root / "images_512"):
                directory.mkdir(parents=True)
            entries, rows = [], []
            for slide_id, slide in slides.items():
                if slide["source"] != name:
                    continue
                entries.append({
                    "slide_id": slide_id, "pairs": 2, "patch_size_level0": 16384,
                    "level_downsamples": [1, 2, 4, 8, 16, 32],
                    "level_dimensions": [[32768 // 2**i] * 2 for i in range(6)],
                })
                for region in range(2):
                    filename = f"{slide_id}_{region}.jpg"
                    lr = f"images_512/{filename}"
                    Image.new("RGB", (512, 512), (80 + 90 * slide["label"], 120, 150)).save(root / lr)
                    rows.append((slide_id, region * 16384, 0, 1, 5, 16, f"images_8192/{filename}", lr))
            _csv(manifest, NATIVE_COLUMNS, rows)
            _json(report, {
                "status": "complete", "hr_level": 1, "lr_level": 5, "hr_size": 8192, "lr_size": 512,
                "scale": 16, "sampling": {"method": "tissue_windows_nonoverlap_v2", "overlap_allowed": False},
                "slides": entries,
            })
            sources[name] = {
                "root": str(root), "manifest": str(manifest), "verification": str(report),
                "manifest_sha256": _digest(manifest), "verification_sha256": _digest(report),
                "wsis": len(entries), "regions": len(rows),
            }
        index, labels = self.protocol / "dataset_sources.json", self.protocol / "labels_all.csv"
        _json(index, {"root_by_source": sources, "slides": slides})
        _csv(labels, ("case_id", "slide_id", "label", "source", "group_id"),
             [(slide_id, slide_id, s["label"], s["source"], s["group_id"]) for slide_id, s in slides.items()])
        self.record = {
            "fold_count": 5, "total_wsis": len(slides), "total_regions": 2 * len(slides),
            "source_wsi_counts": dict(Counter(s["source"] for s in slides.values())),
            "source_index_sha256": _digest(index), "labels_all_sha256": _digest(labels),
            "patient_grouping": "slide_id_proxy_NO_PATIENT_LEVEL_GUARANTEE",
            "folds": [],
        }
        for i in FOLDS:
            val, test = self.parts[(i + 1) % 5], self.parts[i]
            train = [slide for slide in slides if slide not in val + test]
            path = self.protocol / f"splits_5fold_{i}.csv"
            _csv(path, PARTITIONS, zip_longest(train, val, test, fillvalue=""))
            self.record["folds"].append({"fold": i, "file": path.name, "sha256": _digest(path)})
        _json(self.protocol / "protocol.json", self.record)

    def _config(self, name="mean", **changes):
        config = yaml.safe_load((ROOT / "configs/native512_gcsam.yaml").read_text(encoding="utf-8"))
        config["shared_encoder"]["pretrained"] = False
        config["classification"].update(pooling=[name], embedding_dim=8, attention_hidden_dim=8)
        config["metasr"].update(channels=8, hr_tile=16)
        config["augmentation"] = {"enabled": False}
        config.update(precision="fp32", epochs=1, min_epochs=1, scheduler="none")
        config.update(changes)
        path = self.root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def _args(self, config):
        return ["--config", str(config), "--protocol-dir", str(self.protocol), "--device", "cpu"]

    def _input_hashes(self):
        inputs = [p for path in (self.protocol, *self.roots.values()) for p in path.rglob("*") if p.is_file()]
        return {p: _digest(p) for p in inputs}

    def test_dual_roots_and_every_wsi_held_out_once(self):
        before = self._input_hashes()
        protocol = read_camil_protocol(self.protocol)
        samples, _ = load_camil_samples(protocol)
        seen_test = Counter()
        for fold in FOLDS:
            train, val, test = make_camil_datasets(
                protocol, samples, fold, augmentation={"enabled": True}, max_train_regions=1,
            )
            self.assertEqual([len(train), len(val), len(test)], [6, 2, 2])
            self.assertTrue(train.augmentation.enabled)
            self.assertEqual(train[0]["n_regions"], 1)
            self.assertFalse(val.augmentation.enabled)
            self.assertFalse(test.augmentation.enabled)
            self.assertEqual(val[0]["n_regions"], 2)
            self.assertEqual(test[0]["n_regions"], 2)
            ids = [s["slide_id"] for dataset in (train, val, test) for s in dataset.samples]
            self.assertEqual(len(ids), len(set(ids)))
            seen_test.update(s["slide_id"] for s in test.samples)
        self.assertEqual(seen_test, Counter(samples.keys()))
        for sample in samples.values():
            self.assertEqual(sample["lr_paths"][0].parent.parent, self.roots[sample["source"]])
        self.assertEqual(self._input_hashes(), before)
        with self.assertRaisesRegex(FileNotFoundError, "images_8192"):
            load_camil_samples(protocol, require_hr=True)

    def test_rejects_digest_changes_wsi_overlap_and_repeated_outer_test(self):
        path = self.protocol / "splits_5fold_0.csv"
        original = path.read_bytes()
        rows = list(csv.reader(io.StringIO(original.decode())))
        rows[1][0] = rows[1][2]
        _csv(path, rows[0], rows[1:])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            read_camil_protocol(self.protocol)
        self.record["folds"][0]["sha256"] = _digest(path)
        _json(self.protocol / "protocol.json", self.record)
        with self.assertRaisesRegex(ValueError, "overlapping WSI"):
            read_camil_protocol(self.protocol)

        rows = list(csv.reader(io.StringIO(original.decode())))
        _csv(path, rows[0], [(train, test, val) for train, val, test in rows[1:]])
        self.record["folds"][0]["sha256"] = _digest(path)
        _json(self.protocol / "protocol.json", self.record)
        with self.assertRaisesRegex(ValueError, "exactly one outer Test"):
            read_camil_protocol(self.protocol)

    def test_main_reloads_val_best_then_tests_once_and_aggregates_without_overwriting(self):
        config = self._config()
        recipe = yaml.safe_load(config.read_text(encoding="utf-8"))
        recipe["epochs"] = 2
        config.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        folds = read_camil_protocol(self.protocol).folds
        order = iter((fold, part, epoch) for fold in FOLDS
                     for part, epoch in (("train", 1), ("val", 1), ("train", 2), ("val", 2), ("test", 1)))
        best_weights = {}

        class SmallModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.encoder = nn.Linear(1, 1)
                self.head = nn.Linear(1, 2)

            def architecture(self):
                return {"type": "synthetic-model"}

        def step(model, sample, optimizer, device, current):
            with torch.no_grad():
                model.encoder.weight.add_(1.0)
            return dict(train_loss_cls=1.0, train_loss_sr=0.0, train_loss_total=1.0, grad_norm=0.0)

        def evaluate(model, loader, device, current):
            fold, part, epoch = next(order)
            self.assertEqual([s["slide_id"] for s in loader.dataset.samples], folds[fold][part])
            if part == "val" and epoch == 1:
                best_weights[fold] = model.encoder.weight.detach().clone()
            if part == "test":
                torch.testing.assert_close(model.encoder.weight, best_weights[fold], atol=0, rtol=0)
                self.assertAlmostEqual(current["decision_threshold"], 0.1)
            predictions = [
                dict(slide_id=s["slide_id"], label=s["label"], p_tumor=0.02 + 0.16 * s["label"],
                     prediction=0, n_regions=s["n_regions"])
                for s in loader.dataset.samples
            ]
            return dict(loss=1.0, auc=(0.5 + 0.1 * fold if part == "test" else 1.0 / epoch),
                        acc=0.5, bacc=0.5, sensitivity=0.0, specificity=1.0, predictions=predictions)

        before = self._input_hashes()
        output = self.root / "experiment"
        args = [*self._args(config), "--output", str(output)]
        with patch.object(camil, "SharedResNetSRMIL", SmallModel), \
             patch.object(camil.engine_gcsam, "train_wsi", side_effect=step), \
             patch.object(camil.engine, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
            camil.main(args)
            self.assertIsNone(next(order, None))
            first_test = output / "fold_0/test_predictions.csv"
            test_hash = _digest(first_test)
            with self.assertRaises(FileExistsError):
                camil.main(args)
            self.assertEqual(_digest(first_test), test_hash)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(summary["test_auc_mean"], 0.7)
        self.assertAlmostEqual(summary["test_auc_std"], 0.158113883)
        self.assertEqual([f["checkpoint_epoch"] for f in summary["folds"]], [1] * 5)
        self.assertEqual(self._input_hashes(), before)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    def test_real_native512_gcsam_and_ema_for_both_poolings(self):
        for pooling in ("mean", "attention"):
            with self.subTest(pooling=pooling):
                config = self._config(pooling)
                recipe = yaml.safe_load(config.read_text(encoding="utf-8"))
                recipe["precision"] = "bf16"
                config.write_text(yaml.safe_dump(recipe), encoding="utf-8")
                output = self.root / f"actual-{pooling}"
                args = ["--config", str(config), "--protocol-dir", str(self.protocol), "--fold", "0",
                        "--output", str(output)]
                with redirect_stdout(io.StringIO()):
                    camil.main(args)
                best = torch.load(output / "fold_0/best.pth", map_location="cpu", weights_only=True)
                self.assertEqual(best["ema_updates"], 6)
                self.assertEqual(best["optimizer_state"]["gcsam"]["normalization"], "after_gc")
                self.assertEqual(best["architecture"]["mil_pooling"], [pooling])
                self.assertEqual(model_from_checkpoint(best).architecture(), best["architecture"])
                metrics = json.loads((output / "fold_0/test_metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["slide_count"], 2)
                self.assertEqual(metrics["checkpoint_epoch"], 1)


    def test_adamw_reuses_the_joint_sr_training_path(self):
        hr_image = self.root / "synthetic_hr.jpg"
        Image.new("L", (8192, 8192), 128).save(hr_image)
        for root in self.roots.values():
            (root / "images_8192").mkdir()
            for lr in (root / "images_512").glob("*.jpg"):
                shutil.copyfile(hr_image, root / "images_8192" / lr.name)
        config = self._config(optimizer="adamw", lambda_sr=0.05, sr_train_crop=8)
        output = self.root / "joint"
        with redirect_stdout(io.StringIO()):
            camil.main([*self._args(config), "--fold", "0", "--output", str(output)])
        with (output / "fold_0/history.csv").open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertGreater(float(row["train_loss_sr"]), 0.0)
        self.assertAlmostEqual(float(row["train_loss_total"]),
                               float(row["train_loss_cls"]) + 0.05 * float(row["train_loss_sr"]), places=6)
        best = torch.load(output / "fold_0/best.pth", map_location="cpu", weights_only=True)
        self.assertEqual(best["config"]["optimizer"], "adamw")
        self.assertEqual(best["epoch_lambda_sr"], 0.05)
        self.assertTrue((output / "fold_0/test_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()

