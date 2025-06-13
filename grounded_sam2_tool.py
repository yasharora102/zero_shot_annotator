# import os
# import json
# import cv2
# import torch
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from utils.supervision_utils import CUSTOM_COLOR_MAP
# import pycocotools.mask as mask_util
# import supervision as sv

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from supervision.draw.color import ColorPalette

# # ----------- Configuration -------------
# OUTPUT_DIR = "output_annotations"
# SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
# SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ---------------------------------------

# Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
# if DEVICE == "cuda":
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

# # Load models once
# sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
# sam2_predictor = SAM2ImagePredictor(sam2_model)
# processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
# grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)

# def mask_to_rle(mask):
#     rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
#     rle["counts"] = rle["counts"].decode("utf-8")
#     return rle

# def annotate_image(img_path: str, prompt: str = "food."):
#     print(f"🔍 Processing: {img_path} with prompt '{prompt}'")
#     image = Image.open(img_path).convert("RGB")
#     img_np = np.array(image)
#     sam2_predictor.set_image(img_np)

#     inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outputs = grounding_model(**inputs)

#     results = processor.post_process_grounded_object_detection(
#         outputs,
#         inputs.input_ids,
#         box_threshold=0.29,
#         text_threshold=0.3,
#         target_sizes=[image.size[::-1]]
#     )

#     if len(results) == 0 or len(results[0]["boxes"]) == 0:
#         print(f"⚠️ No detections found for {img_path}")
#         return

#     boxes = results[0]["boxes"].cpu().numpy()
#     class_names = results[0]["labels"]
#     scores = results[0]["scores"].cpu().numpy()
#     print(f"Found {len(boxes)} detections.")
#     masks, _, _ = sam2_predictor.predict(
#         point_coords=None, point_labels=None, box=boxes, multimask_output=False
#     )
#     if masks.ndim == 4:
#         masks = masks.squeeze(1)

#     # class_ids = list(range(len(class_names)))
#     class_ids = np.arange(len(class_names))
#     labels = [f"{cls} {score:.2f}" for cls, score in zip(class_names, scores)]

#     detections = sv.Detections(xyxy=boxes, mask=masks.astype(bool), class_id=class_ids)
#     frame = cv2.imread(img_path)
#     frame = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)).annotate(scene=frame, detections=detections)
#     frame = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)).annotate(scene=frame, detections=detections, labels=labels)
#     frame = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)).annotate(scene=frame, detections=detections)

#     # Save visualized image
#     filename = os.path.basename(img_path)
#     cv2.imwrite(f"{OUTPUT_DIR}/{filename}_vis.jpg", frame)

#     # Save annotations in COCO-style
#     mask_rles = [mask_to_rle(mask) for mask in masks]
#     annotations = [
#         {
#             "image_id": filename,
#             "bbox": box.tolist(),
#             "segmentation": rle,
#             "category_id": 0,
#             "category_name": class_name,
#             "score": float(score)
#         }
#         for box, rle, class_name, score in zip(boxes, mask_rles, class_names, scores)
#     ]

#     json_data = {
#         "file_name": filename,
#         "width": image.width,
#         "height": image.height,
#         "annotations": annotations,
#         "prompt": prompt
#     }

#     with open(f"{OUTPUT_DIR}/{filename}.json", "w") as f:
#         json.dump(json_data, f, indent=4)

#     print(f"✅ Done: {filename}")

# # Optional batch runner
# def batch_process(folder="input_images", prompt="food."):
#     for fname in os.listdir(folder):
#         if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             annotate_image(os.path.join(folder, fname), prompt=prompt)

# if __name__ == "__main__":
#     batch_process()



import os
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import pycocotools.mask as mask_util
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP


# ----------- Configuration -------------
OUTPUT_ROOT = Path("output_annotations")
IMAGES_DIR = OUTPUT_ROOT / "images"
MASKS_DIR = OUTPUT_ROOT / "masks"
JSON_DIR = OUTPUT_ROOT / "json"

CVAT_EXPORT_DIR = Path("cvat_export")
CVAT_IMAGES_DIR = CVAT_EXPORT_DIR / "images"
CVAT_MASKS_DIR = CVAT_EXPORT_DIR / "masks"
CVAT_LABELS_FILE = CVAT_EXPORT_DIR / "labels.txt"

# Ensure all directories exist
for d in [
    OUTPUT_ROOT, IMAGES_DIR, MASKS_DIR, JSON_DIR,
    CVAT_EXPORT_DIR, CVAT_IMAGES_DIR, CVAT_MASKS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load models once
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)


def mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def save_masks_as_png(masks, base_filename):
    for i, mask in enumerate(masks):
        mask_path = MASKS_DIR / f"{base_filename}_mask_{i}.png"
        # Save mask as 0/255 uint8 PNG
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))


def annotate_image(img_path: str, prompt: str = "food."):
    print(f"🔍 Processing: {img_path} with prompt '{prompt}'")
    image = Image.open(img_path).convert("RGB")
    img_np = np.array(image)
    sam2_predictor.set_image(img_np)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    if len(results) == 0 or len(results[0]["boxes"]) == 0:
        print(f"⚠️ No detections found for {img_path}")
        return

    boxes = results[0]["boxes"].cpu().numpy()
    class_names = results[0]["labels"]
    scores = results[0]["scores"].cpu().numpy()

    masks, _, _ = sam2_predictor.predict(
        point_coords=None, point_labels=None, box=boxes, multimask_output=False
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    if masks.shape[0] != len(boxes):
        print("❌ Mismatch between boxes and masks. Skipping this image.")
        return

    class_ids = np.arange(len(class_names))  # Make sure it's a numpy array
    labels = [f"{cls} {score:.2f}" for cls, score in zip(class_names, scores)]

    detections = sv.Detections(xyxy=boxes, mask=masks.astype(bool), class_id=class_ids)

    frame = cv2.imread(img_path)
    frame = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)).annotate(scene=frame, detections=detections)
    frame = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)).annotate(scene=frame, detections=detections, labels=labels)
    frame = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP)).annotate(scene=frame, detections=detections)

    filename_stem = Path(img_path).stem

        # Save RGB input for CVAT
    image.save(CVAT_IMAGES_DIR / f"{filename_stem}.jpg")

    # Create label-indexed full mask for CVAT (shape: H x W)
    # Each instance will be filled with its class index
    label_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for class_id, mask in zip(class_ids, masks.astype(bool)):
        label_mask[mask] = class_id + 1  # CVAT expects 0 = background

    # Save mask
    mask_out_path = CVAT_MASKS_DIR / f"{filename_stem}.png"
    cv2.imwrite(str(mask_out_path), label_mask)

    # Save visualized annotated image
    vis_path = IMAGES_DIR / f"{filename_stem}_vis.jpg"
    cv2.imwrite(str(vis_path), frame)

    # Save masks individually as PNGs
    save_masks_as_png(masks, filename_stem)

    # Prepare JSON annotations
    mask_rles = [mask_to_rle(mask) for mask in masks]
    annotations = [
        {
            "image_id": filename_stem,
            "bbox": box.tolist(),
            "segmentation": rle,
            "category_id": 0,
            "category_name": class_name,
            "score": float(score)
        }
        for box, rle, class_name, score in zip(boxes, mask_rles, class_names, scores)
    ]

    json_data = {
        "file_name": filename_stem,
        "width": image.width,
        "height": image.height,
        "annotations": annotations,
        "prompt": prompt
    }

    # Save JSON annotation
    json_path = JSON_DIR / f"{filename_stem}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    write_cvat_labels_file(class_names)

    print(f"✅ Saved annotated image to: {vis_path}")
    print(f"✅ Saved {len(masks)} masks to: {MASKS_DIR}")
    print(f"✅ Saved JSON annotations to: {json_path}")


def batch_process(folder="input_images", prompt="food."):
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            annotate_image(os.path.join(folder, fname), prompt=prompt)


def write_cvat_labels_file(class_names):
    existing = set()
    if CVAT_LABELS_FILE.exists():
        with open(CVAT_LABELS_FILE, "r") as f:
            existing.update([line.strip() for line in f.readlines()])
    new_labels = existing.union(set(class_names))
    with open(CVAT_LABELS_FILE, "w") as f:
        for name in sorted(new_labels):
            f.write(name + "\n")


def binary_mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Must have at least 3 points
            polygon = contour.squeeze().tolist()
            if isinstance(polygon[0], list):
                polygons.append(polygon)
    return polygons



if __name__ == "__main__":
    batch_process()
