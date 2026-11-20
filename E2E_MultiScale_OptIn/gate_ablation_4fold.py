import os
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from E2E_MultiScale_OptIn import multiscale_train as baseline
from E2E_MultiScale_OptIn.multiscale_relation_model import model_from_checkpoint

ROOT = Path("runs/s123_relation_topk_5fold_20261117_110419")
PROTOCOL = Path("data_protocols/camil_5fold_seed123_sourcestrat")
CACHE = Path("/home/ub/wsi_fastcache/lr512_hr256_k2_v1")
DEVICE = torch.device(os.environ.get("DEVICE", "cpu"))

protocol = baseline.camil_data.read_camil_protocol(PROTOCOL)
samples, _ = baseline.camil_data.load_camil_samples(
    protocol, train_root=None, test_root=None, require_hr=True
)

results = []

for fold in range(4):
    path = ROOT / f"fold_{fold}" / "best.pth"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]

    model = model_from_checkpoint(saved, DEVICE).eval()

    ms = config["multiscale"]
    cache = baseline.open_cache(
        samples, CACHE,
        patch_size=ms["hr_patch_size"],
        scale=config["metasr"]["scale"],
        k=ms["patches_per_region"],
    )

    _, _, test = baseline.camil.make_camil_datasets(
        protocol, samples, fold,
        augmentation=config.get("augmentation"),
        max_train_regions=config["training"]["max_train_regions"],
    )
    loader = baseline.camil._loader(test)

    def set_seed():
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)

    # 正常 TopK
    set_seed()
    with torch.inference_mode():
        normal = baseline.evaluate(
            model, loader, cache, DEVICE, config
        )["auc"]

    gate = torch.sigmoid(
        model.region_relation.gate_logit.detach().float()
    ).item()

    # 关闭 TopK：只修改内存中的 Gate
    with torch.no_grad():
        model.region_relation.gate_logit.fill_(-30.0)

    set_seed()
    with torch.inference_mode():
        disabled = baseline.evaluate(
            model, loader, cache, DEVICE, config
        )["auc"]

    original_file = ROOT / f"fold_{fold}" / "test_metrics.json"
    if original_file.exists():
        original = json.loads(original_file.read_text())["auc"]
        if abs(normal - original) > 1e-4:
            print(
                f"WARNING Fold {fold}: "
                f"复现AUC {normal:.6f} != "
                f"原始AUC {original:.6f}",
                flush=True
            )

    delta = disabled - normal
    results.append([fold, gate, normal, disabled, delta])

    print(
        f"Fold {fold} | Gate={gate:.6f} | "
        f"Normal={normal:.6f} | "
        f"GateOFF={disabled:.6f} | "
        f"Delta={delta:+.6f}",
        flush=True
    )

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

out = Path("logs/gate_ablation_4fold.csv")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["fold", "gate", "normal_auc", "gate_off_auc", "delta"]
    )
    writer.writerows(results)

print("\n===== 四折平均 =====")
print(f"Normal AUC : {np.mean([r[2] for r in results]):.6f}")
print(f"GateOFF AUC: {np.mean([r[3] for r in results]):.6f}")
print(f"结果保存: {out}")
