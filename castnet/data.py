import json
import random
import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def free_form_validity_mask(size, seed, min_strokes=4, max_strokes=12):
    rng = random.Random(seed)
    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(min_strokes, max_strokes)):
        points = []
        x, y = rng.randrange(size), rng.randrange(size)
        for _ in range(rng.randint(2, 6)):
            x = max(0, min(size - 1, x + rng.randint(-size // 3, size // 3)))
            y = max(0, min(size - 1, y + rng.randint(-size // 3, size // 3)))
            points.append((x, y))
        draw.line(points, fill=0, width=rng.randint(max(2, size // 40), max(3, size // 10)))
    return TF.pil_to_tensor(image).float() / 255.0


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FixedMaskBank:
    """Immutable mapping from image identifiers to stored validity masks."""

    def __init__(self, registry):
        registry = Path(registry).resolve()
        self.registry = registry
        self.records = {}
        for line in registry.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in self.records:
                raise ValueError(f"duplicate image_id in mask registry: {image_id}")
            mask_path = Path(record["path"])
            if not mask_path.is_absolute():
                mask_path = (registry.parent / mask_path).resolve()
            self.records[image_id] = {**record, "resolved_path": mask_path}

    def load(self, image_id, size):
        image_id = str(image_id)
        if image_id not in self.records:
            raise KeyError(f"mask registry has no record for image_id={image_id}")
        record = self.records[image_id]
        path = record["resolved_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = record.get("sha256")
        if expected_hash and _sha256(path) != expected_hash:
            raise ValueError(f"mask checksum mismatch: {path}")
        mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
        valid = (TF.pil_to_tensor(mask).float() >= 127.5).float()
        coverage = float((1.0 - valid).mean())
        return valid, {
            "mask_id": str(record.get("mask_id", image_id)),
            "mask_coverage": coverage,
            "mask_sha256": expected_hash or _sha256(path),
        }


class ManifestDataset(Dataset):
    def __init__(self, manifest, image_size=256, mask_seed=2026, mask_manifest=None, verify_images=False, data_root=None):
        self.manifest = Path(manifest).resolve()
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image_size = image_size
        self.mask_seed = mask_seed
        self.mask_bank = FixedMaskBank(mask_manifest) if mask_manifest else None
        self.verify_images = verify_images
        self.data_root = Path(data_root).resolve() if data_root else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_id = str(record.get("id", index))
        if self.data_root and record.get("relative_path"):
            image_path = self.data_root / record["relative_path"]
        else:
            image_path = Path(record["path"])
        if not image_path.is_absolute():
            image_path = (self.manifest.parent / image_path).resolve()
        if self.verify_images and record.get("sha256") and _sha256(image_path) != record["sha256"]:
            raise ValueError(f"image checksum mismatch: {image_path}")
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        target = TF.pil_to_tensor(image).float() / 127.5 - 1.0
        if self.mask_bank:
            valid, mask_metadata = self.mask_bank.load(image_id, self.image_size)
        else:
            seed = self.mask_seed + index
            valid = free_form_validity_mask(self.image_size, seed)
            mask_metadata = {
                "mask_id": f"generated-{seed}",
                "mask_coverage": float((1.0 - valid).mean()),
                "mask_sha256": "not-stored",
            }
        corrupted = target * valid
        return {
            "target": target,
            "corrupted": corrupted,
            "valid_mask": valid,
            "image_id": image_id,
            **mask_metadata,
        }
