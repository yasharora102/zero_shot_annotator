#!/usr/bin/env python3
"""
run_grounded_sam2_to_yolo.py

End-to-end pipeline: input_dir → GroundedSAM2 + Grounding DINO segmentation → Ultralytics YOLO Segmentation v1.0 output (for CVAT import).

For each image under input_dir (recursively), this script:
  1. Runs Grounding DINO + SAM2 to detect and mask objects given a text prompt.
  2. Extracts polygon contours from masks via OpenCV.
  3. Normalizes coordinates and writes YOLO-segmentation .txt labels preserving folder structure.
  4. Copies images into data/images/... preserving hierarchy.
  5. Generates data.yaml and train.txt at output_dir for Ultralytics format.

Output structure:
  output_dir/
    data.yaml
    train.txt
    data/
      images/...
      labels/...

Usage:
  pip install opencv-python torch torchvision transformers pycocotools supervision pyyaml
  # Ensure sam2 package and utils are on PYTHONPATH.
  python run_grounded_sam2_to_yolo.py \
    --input-dir path/to/images_root \
    --output-dir path/to/output \
    --prompt "food" \
    --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --grounding-model IDEA-Research/grounding-dino-tiny \
    --device cuda
"""
import os
import shutil
import argparse
from pathlib import Path
import yaml

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
    Convert a binary mask (H×W bool/float) to a list of flattened polygons [x1,y1,x2,y2,...].
    """
    # Ensure mask is 2D
    if mask is None or mask.size == 0:
        return []
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.ndim != 2:
        print(f"  [WARN] mask shape after squeeze: {mask.shape} (should be 2D)")
        return []
    mask = np.ascontiguousarray(mask)
    # Binarize for OpenCV
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    if np.count_nonzero(mask_uint8) == 0:
        return []
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
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    data_root = output_root / "data"
    images_out = data_root / "images"
    labels_out = data_root / "labels"

    # Create output dirs
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Load Grounding DINO
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model).to(args.device)

    # Load SAM2
    sam2 = build_sam2(args.sam_config, args.sam_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2)

    # Track classes
    class_map = {}
    next_idx = 0

    # Process images recursively
    for img_path in input_root.rglob("*.jpg"):  # also matches .jpeg, .png if needed
        # Load image
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        # Grounding DINO detection
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
            continue
        boxes = results[0]["boxes"].cpu().numpy()
        labels = results[0]["labels"]
        scores = results[0]["scores"].cpu().numpy()

        # SAM2 mask generation
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )
        # masks: (N, H, W) bool

        # Compute relative output paths
        rel = img_path.relative_to(input_root)
        img_dest = images_out / rel
        label_dest = labels_out / rel.with_suffix(".txt")
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy image
        shutil.copy2(img_path, img_dest)

        # Build YOLO lines
        lines = []
        for cls_name, mask in zip(labels, masks):
            name = cls_name
            if name not in class_map:
                class_map[name] = next_idx
                next_idx += 1
            cls_idx = class_map[name]
            for poly in mask_to_polygons(mask):
                norm = [v / (W if i % 2 == 0 else H) for i, v in enumerate(poly)]
                coord_str = " ".join(f"{c:.6f}" for c in norm)
                lines.append(f"{cls_idx} {coord_str}")

        # Write label file
        label_dest.write_text("\n".join(lines))

    # Create train.txt listing all image paths
    train_file = output_root / "train.txt"
    with open(train_file, "w") as f:
        for img in sorted(images_out.rglob("*.jpg")):
            # paths relative to output_root
            rel_path = img.relative_to(output_root)
            f.write(str(rel_path).replace('\\', '/') + "\n")

    # Write data.yaml
    names = {idx: name for name, idx in class_map.items()}
    yaml_dict = {
        'names': names,
        'path': '.',
        'train': 'train.txt'
    }
    with open(output_root / 'data.yaml', 'w') as f:
        yaml.dump(yaml_dict, f)

    print(f"Done: processed {len(class_map)} classes, outputs in {output_root}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="GroundedSAM2 + Grounding DINO → Ultralytics YOLO Segmentation v1.0"
    )
    p.add_argument("--input-dir", required=True, help="Root folder of input images")
    p.add_argument("--output-dir", required=True, help="Folder to write dataset (contains data.yaml, train.txt, data/)")
    p.add_argument("--prompt", default="food.", help="Text prompt for grounded detection")
    p.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base",
                   help="HuggingFace grounding DINO model name or path")
    p.add_argument("--box-threshold", type=float, default=0.3, help="Detection box threshold")
    p.add_argument("--text-threshold", type=float, default=0.3, help="Detection text threshold")
    p.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    args = p.parse_args()
    main(args)
# "
# SAM2_MODEL_CONFIG = 
# GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"