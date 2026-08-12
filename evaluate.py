import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from castnet.config import load_yaml, model_config
from castnet.data import ManifestDataset
from castnet.metrics import masked_edge_mae, masked_psnr, standard_ssim
from castnet.model import CASTNet
from castnet.reproducibility import environment_record, git_revision, seed_everything, sha256, write_json


def metric_summary(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def load_perceptual_metrics(device, include_fid):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise RuntimeError("LPIPS/FID require torchmetrics and torch-fidelity; install requirements.txt") from exc
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, reduction="none").to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device) if include_fid else None
    return lpips, fid


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAST-Net and archive per-image evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="CAST-Net")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--allow-generated-masks", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    config = load_yaml(config_path)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CASTNet(model_config(config)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    data_config = config["data"]
    mask_manifest = data_config.get("test_mask_manifest")
    if not mask_manifest and not args.allow_generated_masks:
        raise ValueError("data.test_mask_manifest is required for publishable evaluation; create a fixed mask bank first")
    dataset = ManifestDataset(
        data_config["test_manifest"],
        data_config["image_size"],
        data_config["mask_seed"],
        mask_manifest=mask_manifest,
        verify_images=data_config.get("verify_image_hashes", False),
        data_root=data_config.get("test_root"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    lpips_metric, fid_metric = (None, None)
    if not args.skip_lpips:
        lpips_metric, fid_metric = load_perceptual_metrics(device, not args.skip_fid)

    image_root = output.parent / f"{output.stem}_images"
    completed_dir = image_root / "completed"
    target_dir = image_root / "target"
    if args.save_images:
        completed_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = sha256(checkpoint_path)
    config_hash = sha256(config_path)
    dataset_name = config.get("dataset", config_path.stem)
    run_id = config.get("experiment", checkpoint_path.parent.name)
    records = []

    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device)
            corrupted = batch["corrupted"].to(device)
            valid = batch["valid_mask"].to(device)
            completed = model(corrupted, valid)["completed"]
            prediction_01 = ((completed + 1.0) / 2.0).clamp(0, 1)
            target_01 = ((target + 1.0) / 2.0).clamp(0, 1)
            image_id = batch["image_id"][0]
            record = {
                "dataset": dataset_name,
                "method": args.method,
                "run_id": run_id,
                "seed": int(config["seed"]),
                "image_id": image_id,
                "mask_id": batch["mask_id"][0],
                "mask_coverage": float(batch["mask_coverage"].item()),
                "mask_sha256": batch["mask_sha256"][0],
                "psnr_hole": float(masked_psnr(completed, target, valid)),
                "ssim_full": float(standard_ssim(completed, target, reduction="elementwise_mean")),
                "edge_mae_hole": float(masked_edge_mae(completed, target, valid).mean()),
                "checkpoint_sha256": checkpoint_hash,
                "config_sha256": config_hash,
            }
            if lpips_metric is not None:
                record["lpips_full"] = float(lpips_metric(prediction_01, target_01).mean())
            records.append(record)
            if fid_metric is not None:
                fid_metric.update(target_01, real=True)
                fid_metric.update(prediction_01, real=False)
            if args.save_images:
                TF.to_pil_image(prediction_01[0].cpu()).save(completed_dir / f"{image_id}.png")
                TF.to_pil_image(target_01[0].cpu()).save(target_dir / f"{image_id}.png")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metric_names = ["psnr_hole", "ssim_full", "edge_mae_hole"]
    if records and "lpips_full" in records[0]:
        metric_names.append("lpips_full")
    summary = {
        "dataset": dataset_name,
        "method": args.method,
        "run_id": run_id,
        "seed": int(config["seed"]),
        "count": len(records),
        "metrics": {name: metric_summary(record[name] for record in records) for name in metric_names},
        "fid_full": float(fid_metric.compute()) if fid_metric is not None and len(records) >= 2 else None,
        "manifest_sha256": sha256(data_config["test_manifest"]),
        "mask_manifest_sha256": sha256(mask_manifest) if mask_manifest else None,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "git_commit": git_revision(),
        "environment": environment_record(),
        "saved_images": str(image_root) if args.save_images else None,
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
