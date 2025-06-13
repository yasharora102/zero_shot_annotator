# import os
# import json
# from pathlib import Path
# import shutil

# OUTPUT_DIR = "output_annotations"
# YOLO_OUTPUT_DIR = "yolo_dataset"

# def load_annotations():
#     """Loads all individual annotation JSON files."""
#     data = []
#     for root, dirs, files in os.walk(OUTPUT_DIR):
#         for f in files:
#             # Avoid reading the old coco dataset file
#             if f.endswith(".json") and f != "cvat_dataset.json":
#                 with open(os.path.join(root, f), "r") as infile:
#                     data.append(json.load(infile))
#     return data

# def convert_to_yolo():
#     """Converts the annotations to YOLO segmentation format."""
#     annotations_data = load_annotations()
    

#     yolo_labels_dir = Path(YOLO_OUTPUT_DIR) / "labels"
#     yolo_images_dir = Path(YOLO_OUTPUT_DIR) / "images"

#     # Create the necessary directories
#     yolo_labels_dir.mkdir(parents=True, exist_ok=True)
#     yolo_images_dir.mkdir(parents=True, exist_ok=True)

#     class_names = []

#     for entry in annotations_data:
#         image_height = entry["height"]
#         image_width = entry["width"]
        
#         yolo_txt_filename = Path(entry["file_name"]).stem + ".txt"
        
#         with open(yolo_labels_dir / yolo_txt_filename, "w") as f:
#             for ann in entry["annotations"]:
#                 class_name = ann["label"]
#                 if class_name not in class_names:
#                     class_names.append(class_name)
                
#                 class_id = class_names.index(class_name)
                
#                 # Assuming segmentation is a list of polygons, we take the first one
#                 if ann["segmentation"]:
#                     segmentation = ann["segmentation"][0]
                    
#                     normalized_polygon = []
#                     for i in range(0, len(segmentation), 2):
#                         x = segmentation[i] / image_width
#                         y = segmentation[i+1] / image_height
#                         normalized_polygon.extend([x, y])
                    
#                     f.write(f"{class_id} " + " ".join(map(str, normalized_polygon)) + "\n")
        
#         # Copy the original image to the yolo_dataset/images directory
#         shutil.copy(entry["file_name"], yolo_images_dir)

#     # Create the data.yaml file required by YOLO
#     with open(Path(YOLO_OUTPUT_DIR) / "data.yaml", "w") as f:
#         f.write(f"train: ../{YOLO_OUTPUT_DIR}/images\n")
#         f.write(f"val: ../{YOLO_OUTPUT_DIR}/images\n")
#         f.write(f"nc: {len(class_names)}\n")
#         f.write(f"names: {class_names}\n")

#     print(f"✅ Conversion to YOLO format complete. Dataset saved in '{YOLO_OUTPUT_DIR}'")

# if __name__ == '__main__':
#     convert_to_yolo()


#!/usr/bin/env python3
"""
convert_to_yolo.py

Convert COCO-format instance segmentation (polygon) to YOLO segmentation format
(normalized polygon coordinates), suitable for CVAT import.

Usage:
  python convert_to_yolo.py --coco path/to/coco.json --out-dir labels
"""

import os
import json
import argparse

def convert(coco_json_path, output_dir):
    # --- load COCO file ---
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # --- build image lookup: id -> {file_name, width, height} ---
    images = {img["id"]: img for img in data["images"]}

    # --- build category_id -> zero-based class index, and classes list ---
    # sort categories by their id to keep ordering consistent
    categories = sorted(data["categories"], key=lambda c: c["id"])
    id2idx = {c["id"]: i for i, c in enumerate(categories)}
    classes = [c["name"] for c in categories]

    # --- write classes.txt for CVAT ---
    os.makedirs(output_dir, exist_ok=True)
    classes_txt = os.path.join(output_dir, "classes.txt")
    with open(classes_txt, 'w') as f:
        f.write("\n".join(classes))
    print(f"Wrote class names to {classes_txt}")

    # --- group annotations by image_id ---
    anns_per_image = {}
    for ann in data["annotations"]:
        anns_per_image.setdefault(ann["image_id"], []).append(ann)

    # --- for each image, write a .txt file ---
    for img_id, anns in anns_per_image.items():
        img = images[img_id]
        width, height = img["width"], img["height"]
        base = os.path.splitext(os.path.basename(img["file_name"]))[0]
        txt_path = os.path.join(output_dir, base + ".txt")

        lines = []
        for ann in anns:
            class_idx = id2idx[ann["category_id"]]
            segs = ann.get("segmentation", [])
            # each seg in segs is a flat list [x1,y1,x2,y2,...]
            for seg in segs:
                if not isinstance(seg, list) or len(seg) < 6:
                    # need at least 3 points
                    continue
                # normalize and flatten
                norm_coords = []
                for i, v in enumerate(seg):
                    if i % 2 == 0:
                        norm_coords.append(v / width)
                    else:
                        norm_coords.append(v / height)
                coord_str = " ".join(f"{c:.6f}" for c in norm_coords)
                lines.append(f"{class_idx} {coord_str}")

        # write file (even if empty, CVAT will skip)
        with open(txt_path, 'w') as out:
            out.write("\n".join(lines))

    print(f"YOLO segmentation labels written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert polygon COCO → YOLO segmentation (CVAT format)"
    )
    parser.add_argument(
        "--coco", required=True,
        help="path to input COCO JSON file"
    )
    parser.add_argument(
        "--out-dir", default="labels",
        help="directory to write YOLO .txt files (and classes.txt)"
    )
    args = parser.parse_args()
    convert(args.coco, args.out_dir)
