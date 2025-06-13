
#!/usr/bin/env python3
"""
run_grounded_sam2_to_yolo.py

End-to-end pipeline: input images → GroundedSAM 2 segmentation → YOLO segmentation labels.
Generates:
  - output_dir/images/   (copies of input images)
  - output_dir/labels/   (YOLO-format .txt files)
  - output_dir/classes.txt  (list of detected class names)

Usage:
  pip install opencv-python torch torchvision
  # ensure grounded_sam2_tool.py is on PYTHONPATH
  python run_grounded_sam2_to_yolo.py \
    --input-dir path/to/images \
    --output-dir path/to/output \
    --device cuda
"""
import os
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from grounded_sam2_tool import GroundedSAM2Tool

# Utility: convert binary mask → list of flattened [x1,y1,x2,y2,...]
def mask_to_polygons(mask: np.ndarray):
    mask_uint8 = (mask * 255).astype(np.uint8)
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


def main(input_dir: str, output_dir: str, device: str):
    inp = Path(input_dir)
    out = Path(output_dir)
    images_out = out / "images"
    labels_out = out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Initialize GroundedSAM2 segmentation
    segger = GroundedSAM2Tool(device=device)

    class_names = {}  # label -> idx
    next_idx = 0

    # Process each image
    for img_path in sorted(inp.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # Run GroundedSAM2
        results = segger.predict(img)
        # results: list of {"mask": HxW bool array, "label": str}

        # Copy original image
        shutil.copy2(str(img_path), images_out / img_path.name)

        lines = []
        for ann in results:
            label = ann["label"]
            if label not in class_names:
                class_names[label] = next_idx
                next_idx += 1
            cls_idx = class_names[label]

            mask = ann["mask"]
            polys = mask_to_polygons(mask)
            for poly in polys:
                norm = []
                for i, v in enumerate(poly):
                    if i % 2 == 0:
                        norm.append(v / W)
                    else:
                        norm.append(v / H)
                coord_str = " ".join(f"{c:.6f}" for c in norm)
                lines.append(f"{cls_idx} {coord_str}")

        # Write YOLO .txt file
        txt_path = labels_out / f"{img_path.stem}.txt"
        txt_path.write_text("\n".join(lines))

    # Write classes.txt
    names = [None] * len(class_names)
    for lbl, idx in class_names.items():
        names[idx] = lbl
    (out / "classes.txt").write_text("\n".join(names))

    print(f"Done! Processed {len(list(images_out.iterdir()))} images, found {len(names)} classes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroundedSAM2 → YOLO segmentation end-to-end"
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Folder containing input images (.jpg/.png)"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Output directory for images/, labels/, and classes.txt"
    )
    parser.add_argument(
        "--device", "-d", default="cuda",
        help="Torch device for segmentation: cuda or cpu"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.device)
