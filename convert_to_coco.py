import os
import json
from tqdm import tqdm

def convert_to_coco(input_dir, output_json):
    images = []
    annotations = []
    label_set = set()

    ann_id = 1
    # 1) Gather images + raw labels
    for img_idx, fname in enumerate(sorted(os.listdir(input_dir))):
        if not fname.lower().endswith((".jpg", ".png")): 
            continue

        img_id = img_idx + 1
        # you’ll want to load the true width/height here:
        # e.g. via PIL.Image.open(os.path.join(input_dir,fname)).size
        width, height = 1024, 768  

        images.append({
            "id": img_id,
            "file_name": fname,
            "width": width,
            "height": height,
        })

        # assume per-image JSON with polygons in “shapes”
        ann_path = os.path.join(input_dir, fname.rsplit(".",1)[0] + ".json")
        with open(ann_path) as f:
            raw = json.load(f)

        for shape in raw["shapes"]:
            label = shape["label"]
            label_set.add(label)

            # flatten segmentation for COCO
            seg = [coord for point in shape["points"] for coord in point]

            # compute bbox & area
            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w, h = xmax - xmin, ymax - ymin
            area = w * h

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                # placeholder, fix below once you know all labels → IDs
                "category_id": label,
                "segmentation": [seg],
                "bbox": [xmin, ymin, w, h],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    # 2) Build categories list
    classes = sorted(label_set)
    categories = [
        {"id": idx+1, "name": name, "supercategory": ""}
        for idx, name in enumerate(classes)
    ]
    name2id = {c["name"]: c["id"] for c in categories}

    # 3) Swap out the placeholder label → numeric category_id
    for ann in annotations:
        ann["category_id"] = name2id[ann["category_id"]]

    # 4) Dump full COCO JSON
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True, help="folder with .jpg/.json pairs")
    p.add_argument("--output",     required=True, help="where to write coco.json")
    args = p.parse_args()

    convert_to_coco(args.input_dir, args.output)
