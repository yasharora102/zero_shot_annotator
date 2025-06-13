import os
import json
from pathlib import Path
import shutil

OUTPUT_DIR = "output_annotations"
YOLO_OUTPUT_DIR = "yolo_dataset"

def load_annotations():
    """Loads all individual annotation JSON files."""
    data = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            # Avoid reading the old coco dataset file
            if f.endswith(".json") and f != "cvat_dataset.json":
                with open(os.path.join(root, f), "r") as infile:
                    data.append(json.load(infile))
    return data

def convert_to_yolo():
    """Converts the annotations to YOLO segmentation format."""
    annotations_data = load_annotations()
    

    yolo_labels_dir = Path(YOLO_OUTPUT_DIR) / "labels"
    yolo_images_dir = Path(YOLO_OUTPUT_DIR) / "images"

    # Create the necessary directories
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    yolo_images_dir.mkdir(parents=True, exist_ok=True)

    class_names = []

    for entry in annotations_data:
        image_height = entry["height"]
        image_width = entry["width"]
        
        yolo_txt_filename = Path(entry["file_name"]).stem + ".txt"
        
        with open(yolo_labels_dir / yolo_txt_filename, "w") as f:
            for ann in entry["annotations"]:
                class_name = ann["label"]
                if class_name not in class_names:
                    class_names.append(class_name)
                
                class_id = class_names.index(class_name)
                
                # Assuming segmentation is a list of polygons, we take the first one
                if ann["segmentation"]:
                    segmentation = ann["segmentation"][0]
                    
                    normalized_polygon = []
                    for i in range(0, len(segmentation), 2):
                        x = segmentation[i] / image_width
                        y = segmentation[i+1] / image_height
                        normalized_polygon.extend([x, y])
                    
                    f.write(f"{class_id} " + " ".join(map(str, normalized_polygon)) + "\n")
        
        # Copy the original image to the yolo_dataset/images directory
        shutil.copy(entry["file_name"], yolo_images_dir)

    # Create the data.yaml file required by YOLO
    with open(Path(YOLO_OUTPUT_DIR) / "data.yaml", "w") as f:
        f.write(f"train: ../{YOLO_OUTPUT_DIR}/images\n")
        f.write(f"val: ../{YOLO_OUTPUT_DIR}/images\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"✅ Conversion to YOLO format complete. Dataset saved in '{YOLO_OUTPUT_DIR}'")

if __name__ == '__main__':
    convert_to_yolo()