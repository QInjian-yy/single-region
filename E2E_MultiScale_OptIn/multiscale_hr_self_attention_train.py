"""Train multiscale MIL with HR patch self-attention before region aggregation."""
from __future__ import annotations

import argparse
from inspect import signature
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

import camil_data
from .multiscale_relation_train import baseline
from .ssd_support.relation_paths import remap_lr_samples


FORMAT = "c16-e2e-lrhr-hr-self-attention-v1"
HR_SELF_ATTENTION = {
    "layers": 1,
    "num_heads": 4,
    "attention_dropout": 0.0,
    "residual_init": 0.0,
}


class HRSelfAttentionPool(nn.Module):
    """Update [N,K,D] within each region before applying the original HR scorer."""

    def __init__(self, scorer):
        super().__init__()
        dim = scorer.in_features
        self.scorer = scorer
        self.norm = nn.LayerNorm(dim)
        # Attention initialization must not change the baseline shuffle RNG.
        with torch.random.fork_rng(devices=[]):
            self.attention = nn.MultiheadAttention(
                dim, HR_SELF_ATTENTION["num_heads"],
                dropout=HR_SELF_ATTENTION["attention_dropout"], batch_first=True,
            )
        self.gamma = nn.Parameter(torch.tensor(HR_SELF_ATTENTION["residual_init"]))

    def interact(self, embeddings):
        tokens = embeddings.float()
        normalized = self.norm(tokens)
        update, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return tokens + self.gamma.float() * update.float()

    def forward(self, embeddings):
        updated = self.interact(embeddings)
        weights = self.scorer(updated).softmax(dim=1)
        return (weights * updated).sum(dim=1)


class HRSelfAttentionMIL(baseline.MultiScaleSRMIL):
    def __init__(self, config):
        super().__init__(config)
        # The original head optimizer already includes all patch_score parameters.
        self.patch_score = HRSelfAttentionPool(self.patch_score)

    def encode_multiscale(self, lr, hr, *, need_sr=False):
        if lr.ndim != 4 or hr.ndim != 5:
            raise ValueError("Expected LR [N,3,512,512] and HR [N,K,3,P,P]")
        n, k, channels, height, width = hr.shape
        if (n != lr.shape[0] or k != self.k or channels != 3
                or (height, width) != (self.patch_size, self.patch_size)):
            raise ValueError("HR local patch dimensions mismatch config")

        stages = self.base.encoder(lr)
        lr_embedding = self.base.classifier_head(stages[-1])
        hr_stages = self.base.encoder(hr.reshape(n * k, 3, height, width))
        hr_embedding = self.base.classifier_head(hr_stages[-1]).reshape(n, k, -1)
        hr_aggregate = self.patch_score(hr_embedding)

        residual = self.fusion(torch.cat((lr_embedding.float(), hr_aggregate), dim=1))
        fused = lr_embedding.float() + torch.sigmoid(self.fusion_logit) * residual
        sr_features = (self.base.sr.fuse_stages(stages, target_size=lr.shape[-2:])
                       if need_sr else None)
        return fused, sr_features

    def architecture(self):
        return {
            **super().architecture(),
            "type": FORMAT,
            "hr_self_attention": dict(HR_SELF_ATTENTION),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ssd-root", type=Path, default=Path("/home/ub/wsi_fastcache"))
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--fold", type=int, choices=camil_data.FOLDS)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-data", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if "model_factory" not in signature(baseline.train_fold).parameters:
        parser.error("Use the relation-enabled multiscale trainer from the Seed123 SSD runs")

    config = baseline.load_config(args.config)
    relation = config.get("region_relation") or {}
    if (relation.get("enabled", False)
            or config.get("hr_aggregation") not in (None, "patch_score")
            or config.get("fusion_drop_path")):
        parser.error("Use the original patch_score multiscale configuration")
    config["hr_self_attention"] = dict(HR_SELF_ATTENTION)

    print("[startup] Reading the protocol and checking original LR/HR image headers", flush=True)
    protocol = camil_data.read_camil_protocol(args.protocol_dir)
    original, provenance = camil_data.load_camil_samples(
        protocol, train_root=args.train_root, test_root=args.test_root, require_hr=True,
    )
    print("[startup] Checking SSD LR mappings and the HR pixel cache", flush=True)
    samples, ssd = remap_lr_samples(original, args.ssd_root)
    provenance["ssd_lr"] = ssd
    multiscale = config["multiscale"]
    cache = baseline.open_cache(
        original, args.cache_dir, patch_size=multiscale["hr_patch_size"],
        scale=config["metasr"]["scale"], k=multiscale["patches_per_region"],
    )
    print(f"[data] {len(samples)} WSI, {ssd['regions']} Regions; "
          f"grouping={protocol.patient_grouping}", flush=True)
    print(f"[hr_self_attention] layers=1 heads=4 gamma_init=0.0 HR cache K={cache.k}",
          flush=True)
    if args.check_data:
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 not supported on GPU")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    out = args.output.resolve()
    input_roots = [args.protocol_dir, args.cache_dir, args.ssd_root, *(
        Path(provenance[f"{source}_manifest"]["data_root"]) for source in protocol.sources
    )]
    if any(out.is_relative_to(root.resolve()) for root in input_roots):
        raise ValueError("Output must be outside the data, protocol and cache roots")
    out.mkdir(parents=True, exist_ok=False)

    folds = camil_data.FOLDS if args.fold is None else (args.fold,)
    results = [
        baseline.train_fold(
            config, protocol, samples, provenance, cache, fold, out / f"fold_{fold}",
            device, model_factory=HRSelfAttentionMIL, training_entry=Path(__file__).name,
        )
        for fold in folds
    ]
    if args.fold is None:
        aucs = [result["auc"] for result in results]
        baseline.camil._write_json(out / "summary.json", dict(
            fold_count=len(results), test_auc_mean=mean(aucs), test_auc_std=stdev(aucs),
            folds=results,
        ))
        print(f"[five folds] Test AUC {mean(aucs):.6f} +/- {stdev(aucs):.6f}", flush=True)


if __name__ == "__main__":
    main()
