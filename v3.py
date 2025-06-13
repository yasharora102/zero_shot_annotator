#!/usr/bin/env python3
"""
run_grounded_sam2_to_yolo.py

End-to-end pipeline:
  input_dir → GroundedSAM2 + Grounding DINO segmentation → Ultralytics YOLO Segmentation v1.0 folder structure (for CVAT import).

Produces:
  output_root/
    data.yaml
    train.txt
    images/
      train/outs/images/*.jpg
    labels/
      train/outs/images/*.txt

Usage:
  pip install opencv-python torch torchvision transformers pycocotools supervision pyyaml
  python run_grounded_sam2_to_yolo.py \
    --input-dir path/to/images \
    --output-dir path/to/output \
    --prompt "food" \
    --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --grounding-model IDEA-Research/grounding-dino-tiny \
    --device cuda
"""
import shutil
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
import yaml

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# def mask_to_polygons(mask: np.ndarray):
#     """
#     Convert a binary mask (H×W bool) to list of flattened polygons [x1,y1,x2,y2,...].
#     """
#     m = (mask.astype(np.uint8) * 255)
#     contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     polys = []
#     for cnt in contours:
#         pts = cnt.squeeze()
#         if pts.ndim != 2 or len(pts) < 3:
#             continue
#         flat = []
#         for x, y in pts:
#             flat.extend([int(x), int(y)])
#         polys.append(flat)
#     return polys



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
    out = Path(args.output_dir)

    # Define the required folder structure
    img_folder = out / "images" / "train" / "outs" / "images"
    lbl_folder = out / "labels" / "train" / "outs" / "images"
    img_folder.mkdir(parents=True, exist_ok=True)
    lbl_folder.mkdir(parents=True, exist_ok=True)

    # Load Grounding DINO
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_model
    ).to(args.device)

    # Load SAM2
    sam2 = build_sam2(args.sam_config, args.sam_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(sam2)

    class_map = {}
    next_idx = 0
    image_paths = []

    # Process each image
    for img_path in input_root.rglob("*.jpg"):
        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)
        H, W = img_np.shape[:2]

        # Grounding DINO detection
        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(args.device)
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

        # SAM2 mask generation
        predictor.set_image(img_np)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )

        # Copy image
        dest_img = img_folder / img_path.name
        shutil.copy2(img_path, dest_img)
        image_paths.append(dest_img)

        # Write YOLO .txt
        lines = []
        for cls_name, mask in zip(labels, masks):
            if cls_name not in class_map:
                class_map[cls_name] = next_idx
                next_idx += 1
            cls_idx = class_map[cls_name]
            for poly in mask_to_polygons(mask):
                norm = [v/(W if i%2==0 else H) for i, v in enumerate(poly)]
                coord_str = " ".join(f"{c:.6f}" for c in norm)
                lines.append(f"{cls_idx} {coord_str}")
        (lbl_folder / f"{img_path.stem}.txt").write_text("\n".join(lines))

    # Write train.txt listing image paths
    train_txt = out / "train.txt"
    with open(train_txt, "w") as f:
        for p in sorted(image_paths):
            rel = p.relative_to(out)
            f.write(str(rel).replace('\\', '/') + "\n")

    # Write data.yaml
    names = {idx: name for name, idx in class_map.items()}
    yaml_dict = {
        'names': names,
        'path': '.',
        'train': 'train.txt'
    }
    (out / 'data.yaml').write_text(yaml.dump(yaml_dict))

    print(f"Done: processed {len(image_paths)} images, {len(names)} classes. Output at {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundedSAM2 + Grounding DINO → YOLO Segmentation v1.0 structure"
    )
    parser.add_argument("--input-dir", required=True, help="Folder of input .jpg images")
    parser.add_argument("--output-dir", required=True, help="Root output folder for dataset")
    parser.add_argument("--prompt", default="food.", help="Grounding DINO text prompt")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-base",
                   help="HuggingFace grounding DINO model name or path")
    parser.add_argument("--box-threshold", type=float, default=0.3, help="Box score threshold")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="Text score threshold")
    parser.add_argument("--device", default="cuda", help="Torch device: cuda or cpu")
    args = parser.parse_args()
    main(args)
