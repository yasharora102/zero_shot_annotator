import os
import json
from pathlib import Path

OUTPUT_DIR = "output_annotations"
COCO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cvat_dataset.json")

def load_annotations():
    data = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith(".json") and f != "cvat_dataset.json":
                with open(os.path.join(root, f), "r") as infile:
                    data.append(json.load(infile))
    return data

def convert_to_coco():
    annotations_data = load_annotations()

    images = []
    annotations = []
    categories = {}
    category_id_map = {}
    ann_id = 1
    cat_id = 1

    for img_id, entry in enumerate(annotations_data, 1):
        image_id = img_id
        images.append({
            "id": image_id,
            "file_name": entry["file_name"],
            "width": entry["width"],
            "height": entry["height"]
        })

        for ann in entry["annotations"]:
            cat_name = ann["category_name"]
            if cat_name not in categories:
                categories[cat_name] = {
                    "id": cat_id,
                    "name": cat_name,
                    "supercategory": "none"
                }
                category_id_map[cat_name] = cat_id
                cat_id += 1

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id_map[cat_name],
                "bbox": ann["bbox"],
                "segmentation": ann["segmentation"],
                "area": ann["segmentation"].get("size", 0),
                "iscrowd": 0
            })
            ann_id += 1

    coco_format = {
        "info": {
            "description": "GroundedSAM2 CVAT Dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": list(categories.values())
    }

    with open(COCO_OUTPUT_PATH, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"✅ COCO dataset written to {COCO_OUTPUT_PATH}")

if __name__ == "__main__":
    convert_to_coco()
