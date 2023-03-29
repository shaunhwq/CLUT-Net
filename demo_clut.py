import os
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

from models import CLUTNet


def pre_process(image: np.array, device: str) -> torch.tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(0)
    # image = torch.stack([image])
    image = image.to(device)
    return image


def post_process(output_tensor):
    image_rgb = output_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image_rgb = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("--model", type=str, default="20+05+20", help="model configuration, n+s+w. Options: '20+05+10', '20+05+20'")
    args = parser.parse_args()

    model_weight_path = {
        "20+05+10": "FiveK/20+05+10_models/model0310.pth",
        "20+05+20": "FiveK/20+05+20_models/model0361.pth",
    }.get(args.model, None)

    assert model_weight_path is not None, f"Invalid option for --model: '{args.model}'"
    assert os.path.exists(model_weight_path), "Model path does not exist"

    # Prepare output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    model = CLUTNet(args.model, dim=33)
    model.load_state_dict(torch.load(model_weight_path))
    model.to(args.device)
    model.eval()

    # Prepare images
    image_paths = [os.path.join(args.input_dir, img_path) for img_path in os.listdir(args.input_dir) if img_path[0] != "."]

    # Model inference
    with torch.no_grad():
        for img_path in tqdm(image_paths, total=len(image_paths), desc=f"Running CLUT-Net({args.model})"):
            in_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            model_input = pre_process(in_image, args.device)

            model_output, _ = model(model_input, model_input)

            enhanced_image = post_process(model_output)

            output_path = os.path.join(args.output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, enhanced_image)
            