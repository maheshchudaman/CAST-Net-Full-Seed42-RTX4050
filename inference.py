import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from castnet.config import load_yaml, model_config
from castnet.model import CASTNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--valid-mask", required=True, help="White=observed, black=missing")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    size = config["data"]["image_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(args.image).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    mask = Image.open(args.valid_mask).convert("L").resize((size, size), Image.Resampling.NEAREST)
    target = TF.pil_to_tensor(image).float().unsqueeze(0).to(device) / 127.5 - 1.0
    valid = TF.pil_to_tensor(mask).float().unsqueeze(0).to(device) / 255.0
    model = CASTNet(model_config(config)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    with torch.no_grad():
        completed = model(target * valid, valid)["completed"][0].cpu()
    output = TF.to_pil_image(((completed + 1.0) / 2.0).clamp(0, 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)


if __name__ == "__main__":
    main()
