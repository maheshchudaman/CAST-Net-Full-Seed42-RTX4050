import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from castnet.config import load_yaml, model_config
from castnet.data import ManifestDataset
from castnet.discriminator import PatchDiscriminator, discriminator_hinge, generator_hinge
from castnet.losses import generator_objective
from castnet.metrics import masked_psnr
from castnet.model import CASTNet
from castnet.reproducibility import environment_record, git_revision, seed_everything, sha256, write_json


def evaluate_validation(model, loader, device, amp_enabled=False):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device)
            corrupted = batch["corrupted"].to(device)
            valid = batch["valid_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                completed = model(corrupted, valid)["completed"]
            values.append(float(masked_psnr(completed, target, valid)))
    model.train()
    return sum(values) / len(values) if values else float("-inf")


def main():
    parser = argparse.ArgumentParser(description="Train CAST-Net with checkpoint provenance.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a CAST-Net training checkpoint.")
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help="Overwrite latest.pt every N optimizer steps. Defaults to the configuration value or epoch-only saves.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (for example cuda, mps, or cpu). Defaults to the best available device.",
    )
    args = parser.parse_args()

    source_config = Path(args.config).resolve()
    config = load_yaml(source_config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.experiment:
        config["experiment"] = args.experiment
    elif args.seed is not None:
        base = config.get("dataset", source_config.stem)
        config["experiment"] = f"{base}_seed{args.seed}"

    seed_everything(config["seed"])
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    run_dir = Path("runs") / config["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.resolved.yaml"
    snapshot.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    data_config = config["data"]
    manifest_hashes = {
        "train": sha256(data_config["train_manifest"]),
        "validation": sha256(data_config["validation_manifest"]),
    }
    if data_config.get("validation_mask_manifest"):
        manifest_hashes["validation_masks"] = sha256(data_config["validation_mask_manifest"])
    metadata = {
        "git_commit": git_revision(),
        "source_config": str(source_config),
        "resolved_config_sha256": sha256(snapshot),
        "manifest_sha256": manifest_hashes,
        "environment": environment_record(),
        "training_device": str(device),
        "seed": config["seed"],
        "status": "started",
        "checkpoint_selection": "maximum validation hole PSNR",
    }
    write_json(run_dir / "run_metadata.json", metadata)

    train_dataset = ManifestDataset(
        data_config["train_manifest"],
        data_config["image_size"],
        data_config["mask_seed"],
        verify_images=data_config.get("verify_image_hashes", False),
        data_root=data_config.get("train_root"),
    )
    validation_dataset = ManifestDataset(
        data_config["validation_manifest"],
        data_config["image_size"],
        data_config.get("validation_mask_seed", data_config["mask_seed"] + 1_000_000),
        mask_manifest=data_config.get("validation_mask_manifest"),
        verify_images=data_config.get("verify_image_hashes", False),
        data_root=data_config.get("validation_root"),
    )
    workers = int(config["training"].get("num_workers", 0))
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=workers)

    model = CASTNet(model_config(config)).to(device)
    discriminator = PatchDiscriminator(config["model"]["base_channels"]).to(device)
    optimizer_g = torch.optim.Adam(
        model.parameters(),
        config["training"]["learning_rate"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        config["training"]["learning_rate"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )

    amp_enabled = bool(config["training"].get("amp", False) and device.type == "cuda")
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_every_steps = (
        args.checkpoint_every_steps
        if args.checkpoint_every_steps is not None
        else int(config["training"].get("checkpoint_every_steps", 0))
    )
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must not be negative")
    metadata.update(
        {
            "amp_enabled": amp_enabled,
            "micro_batch_size": int(config["training"]["batch_size"]),
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": int(config["training"]["batch_size"]) * accumulation_steps,
            "checkpoint_every_steps": checkpoint_every_steps,
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)

    step = 0
    start_epoch = 0
    resume_batch_index = 0
    resume_epoch_generator_state = None
    best_validation_psnr = float("-inf")
    best_epoch = None
    best_checkpoint_hash = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        if checkpoint.get("grad_scaler"):
            scaler.load_state_dict(checkpoint["grad_scaler"])
        step = int(checkpoint["step"])
        start_epoch = int(checkpoint["epoch"])
        resume_batch_index = int(checkpoint.get("next_batch_index", 0))
        resume_epoch_generator_state = checkpoint.get("epoch_generator_state")
        if resume_epoch_generator_state is not None:
            resume_epoch_generator_state = resume_epoch_generator_state.cpu()
        best_validation_psnr = float(checkpoint.get("best_validation_psnr", float("-inf")))
        best_epoch = checkpoint.get("best_epoch")
        best_checkpoint_hash = checkpoint.get("best_checkpoint_hash")
        metadata.update(
            {
                "status": "resumed",
                "resumed_from": str(resume_path),
                "resumed_from_step": step,
            }
        )
        write_json(run_dir / "run_metadata.json", metadata)

    def checkpoint_payload(epoch, next_batch_index, epoch_generator_state, validation_psnr=None):
        return {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "epoch_generator_state": epoch_generator_state,
            "step": step,
            "validation_psnr_hole": validation_psnr,
            "best_validation_psnr": best_validation_psnr,
            "best_epoch": best_epoch,
            "best_checkpoint_hash": best_checkpoint_hash,
            "config": config,
            "git_commit": metadata["git_commit"],
            "resolved_config_sha256": metadata["resolved_config_sha256"],
            "manifest_sha256": manifest_hashes,
        }

    log_path = run_dir / "metrics.jsonl"
    log_mode = "a" if args.resume else "w"
    with log_path.open(log_mode, encoding="utf-8") as log:
        for epoch in range(start_epoch, config["training"]["epochs"]):
            model.train()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            micro_step = 0
            micro_in_group = 0
            group_size = accumulation_steps
            if epoch == start_epoch and resume_epoch_generator_state is not None:
                generator.set_state(resume_epoch_generator_state)
            epoch_generator_state = generator.get_state()
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < resume_batch_index:
                    continue
                if micro_in_group == 0:
                    group_size = min(accumulation_steps, len(train_loader) - batch_index)
                target = batch["target"].to(device)
                corrupted = batch["corrupted"].to(device)
                valid = batch["valid_mask"].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output = model(corrupted, valid)
                    loss_d = discriminator_hinge(
                        discriminator(target, valid),
                        discriminator(output["completed"].detach(), valid),
                    )
                scaler.scale(loss_d / group_size).backward()

                discriminator.requires_grad_(False)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    losses = generator_objective(
                        output,
                        target,
                        valid,
                        config["training"]["lambda_frequency"],
                        config["training"]["lambda_edge"],
                    )
                    adversarial = generator_hinge(discriminator(output["completed"], valid))
                    loss_g = losses["total"] + config["training"].get("lambda_adversarial", 0.1) * adversarial
                scaler.scale(loss_g / group_size).backward()
                discriminator.requires_grad_(True)

                micro_step += 1
                micro_in_group += 1
                if micro_in_group != group_size:
                    continue

                scaler.step(optimizer_d)
                scaler.step(optimizer_g)
                scaler.update()
                optimizer_d.zero_grad(set_to_none=True)
                optimizer_g.zero_grad(set_to_none=True)
                micro_in_group = 0
                step += 1
                record = {
                    "type": "training_step",
                    "epoch": epoch,
                    "step": step,
                    "micro_step": micro_step,
                    "effective_batch_size": int(config["training"]["batch_size"]) * accumulation_steps,
                    "amp_enabled": amp_enabled,
                    "loss_g": float(loss_g.detach()),
                    "loss_d": float(loss_d.detach()),
                    "loss_adversarial": float(adversarial.detach()),
                    **{key: float(value.detach()) for key, value in losses.items()},
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                if checkpoint_every_steps and step % checkpoint_every_steps == 0:
                    torch.save(
                        checkpoint_payload(epoch, batch_index + 1, epoch_generator_state),
                        run_dir / "latest.pt",
                    )
                if args.max_steps and step >= args.max_steps:
                    break

            validation_psnr = evaluate_validation(model, validation_loader, device, amp_enabled)
            log.write(json.dumps({"type": "validation_epoch", "epoch": epoch, "step": step, "psnr_hole": validation_psnr}, sort_keys=True) + "\n")
            log.flush()
            if validation_psnr > best_validation_psnr:
                best_validation_psnr = validation_psnr
                best_epoch = epoch
                best_checkpoint = run_dir / "best.pt"
                torch.save(
                    checkpoint_payload(epoch + 1, 0, generator.get_state(), validation_psnr),
                    best_checkpoint,
                )
                best_checkpoint_hash = sha256(best_checkpoint)
            epoch_completed = batch_index + 1 == len(train_loader)
            next_epoch = epoch + 1 if epoch_completed else epoch
            next_batch_index = 0 if epoch_completed else batch_index + 1
            next_generator_state = generator.get_state() if epoch_completed else epoch_generator_state
            payload = checkpoint_payload(next_epoch, next_batch_index, next_generator_state, validation_psnr)
            latest_checkpoint = run_dir / "latest.pt"
            torch.save(payload, latest_checkpoint)
            if args.max_steps and step >= args.max_steps:
                break

    metadata.update(
        {
            "status": "complete",
            "steps": step,
            "latest_checkpoint_sha256": sha256(latest_checkpoint),
            "best_checkpoint_sha256": best_checkpoint_hash,
            "best_epoch": best_epoch,
            "best_validation_psnr_hole": best_validation_psnr,
            "metrics_log_sha256": sha256(log_path),
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
