#!/usr/bin/env python3
"""
run_grounded_sam2_to_yolo.py

End-to-end pipeline: input_dir → GroundedSAM2 + Grounding DINO segmentation → YOLO segmentation labels.

For each image in input_dir, this script:
  1. Runs the grounding-dino model to detect objects given a text prompt.
  2. Uses SAM2 to generate precise masks for each detected box.
  3. Extracts polygon contours from masks via OpenCV.
  4. Normalizes coordinates and writes YOLO-segmentation .txt for each image.
  5. Copies images to output_dir/images and writes classes.txt in output_dir.

Usage:
  pip install opencv-python torch torchvision transformers pycocotools supervision
  # Ensure sam2 package and utils are on PYTHONPATH.
  python run_grounded_sam2_to_yolo.py \
    --input-dir path/to/images \
    --output-dir path/to/output \
    --prompt "food" \
    --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --grounding-model IDEA-Research/grounding-dino-tiny \
    --device cuda
"""
import os
import json
import shutil
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def mask_to_polygons(mask: np.ndarray):
    """
    Convert a binary mask (H×W bool) to a list of flattened polygons [x1,y1,x2,y2,...].
    """
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue
        flat = []
        for x, y in pts:
            flat.extend([int(x), int(y)])
        polys.append(flat)
    return polys


def main(args):
    # Prepare output directories
    out = Path(args.output_dir)
    imgs_out = out / "images"
    labels_out = out / "labels"
    imgs_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Load Grounding DINO processor and model
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model).to(args.device)

    # Load SAM2 model and predictor
    sam2 = build_sam2(args.sam_config, args.sam_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2)

    # Class-name to index mapping
    class_map = {}
    next_idx = 0

    # Process each image
    for img_path in sorted(Path(args.input_dir).iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Load
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        # 1) Grounding DINO detection
        inputs = processor(images=image_pil, text=args.prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(H, W)]
        )
        if not results or len(results[0]["boxes"]) == 0:
            print(f"No detections for {img_path.name}")
            continue
        boxes = results[0]["boxes"].cpu().numpy()
        labels = results[0]["labels"]
        scores = results[0]["scores"].cpu().numpy()

        # 2) SAM2 mask prediction
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )
        # masks: (N, H, W) boolean

        # Copy image to output
        shutil.copy2(str(img_path), imgs_out / img_path.name)

        lines = []
        # Per-detection
        for cls_name, score, mask in zip(labels, scores, masks):
            name = cls_name  # textual label
            if name not in class_map:
                class_map[name] = next_idx
                next_idx += 1
            cls_idx = class_map[name]
            # polygons
            polys = mask_to_polygons(mask)
            for poly in polys:
                norm = []
                for i, v in enumerate(poly):
                    norm.append(v / (W if i % 2 == 0 else H))
                coord_str = " ".join(f"{c:.6f}" for c in norm)
                lines.append(f"{cls_idx} {coord_str}")

        # Write YOLO .txt for this image
        txt_file = labels_out / f"{img_path.stem}.txt"
        txt_file.write_text("\n".join(lines))

    # Write classes.txt
    names = [None] * len(class_map)
    for name, idx in class_map.items():
        names[idx] = name
    (out / "classes.txt").write_text("\n".join(names))

    print(f"Processed {len(list(imgs_out.iterdir()))} images; found {len(names)} classes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundedSAM2 + Grounding DINO → YOLO segmentation"
    )
    parser.add_argument("--input-dir", required=True, help="Folder of images (.jpg/.png)")
    parser.add_argument("--output-dir", required=True, help="Output dir for images/, labels/, classes.txt")
    parser.add_argument("--prompt", default="food", help="Text prompt for grounded detection")
    parser.add_argument("--sam-checkpoint", required=True, help="Path to SAM2 checkpoint .pt")
    parser.add_argument("--sam-config", required=True, help="Path to SAM2 YAML config")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny",
                        help="HuggingFace model for grounding DINO")
    parser.add_argument("--box-threshold", type=float, default=0.3, help="Box score threshold")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="Text score threshold")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    args = parser.parse_args()
    main(args)
