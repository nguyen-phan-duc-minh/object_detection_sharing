import json
import importlib
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
	mask_utils = importlib.import_module("pycocotools.mask")
except Exception:
	mask_utils = None


DATA_ROOT = Path("/kaggle/input/datasets/nguynphancminh/snack-segmentation")
WORK_ROOT = Path("/kaggle/working/snack_seg_yolo")
MODEL_PATH = "/kaggle/input/models/nguynphancminh/yolo26n-seg/pytorch/default/1/yolo26n-seg.pt"


def _bbox_to_polygon(ann: dict) -> list[list[float]]:
	bbox = ann.get("bbox")
	if not isinstance(bbox, list) or len(bbox) != 4:
		return []
	x, y, w, h = bbox
	if w <= 1 or h <= 1:
		return []
	return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def _rle_to_polygons(segmentation: dict, h: int, w: int) -> list[list[float]]:
	if mask_utils is None:
		return []
	try:
		rle = segmentation
		if isinstance(segmentation.get("counts"), list):
			rle = mask_utils.frPyObjects(segmentation, h, w)
		mask = mask_utils.decode(rle)
		if mask.ndim == 3:
			mask = mask[:, :, 0]
		mask = (mask > 0).astype("uint8")
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		polygons = []
		for c in contours:
			if c.shape[0] < 3:
				continue
			flat = c.reshape(-1, 2).astype(float).flatten().tolist()
			if len(flat) >= 6:
				polygons.append(flat)
		return polygons
	except Exception:
		return []


def _ann_to_polygons(ann: dict, h: int, w: int) -> list[list[float]]:
	segmentation = ann.get("segmentation")

	if isinstance(segmentation, list):
		if segmentation and isinstance(segmentation[0], (int, float)):
			polygons = [segmentation]
		else:
			polygons = [p for p in segmentation if isinstance(p, list)]
	elif isinstance(segmentation, dict):
		polygons = _rle_to_polygons(segmentation, int(h), int(w))
	else:
		polygons = []

	if not polygons:
		polygons = _bbox_to_polygon(ann)

	return [p for p in polygons if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]


def _resolve_split_dir(split_name: str) -> str:
	aliases = {
		"train": ["train"],
		"val": ["val", "valid"],
		"test": ["test"],
	}
	for candidate in aliases.get(split_name, [split_name]):
		ann = DATA_ROOT / candidate / "_annotations.coco.json"
		if ann.exists():
			return candidate
	raise FileNotFoundError(
		f"Khong tim thay _annotations.coco.json cho split '{split_name}' (da thu {aliases.get(split_name, [split_name])})"
	)


def convert_split(split_name: str) -> None:
	real_split = _resolve_split_dir(split_name)
	ann_path = DATA_ROOT / real_split / "_annotations.coco.json"
	with ann_path.open("r", encoding="utf-8") as f:
		coco = json.load(f)

	images_by_id = {img["id"]: img for img in coco["images"]}
	anns_by_image = defaultdict(list)
	for ann in coco["annotations"]:
		if ann.get("iscrowd", 0) == 0:
			anns_by_image[ann["image_id"]].append(ann)

	img_out = WORK_ROOT / "images" / split_name
	lbl_out = WORK_ROOT / "labels" / split_name
	img_out.mkdir(parents=True, exist_ok=True)
	lbl_out.mkdir(parents=True, exist_ok=True)

	for image_id, img in images_by_id.items():
		w = float(img["width"])
		h = float(img["height"])
		name = Path(img["file_name"]).name
		src = DATA_ROOT / real_split / name
		if not src.exists():
			src = DATA_ROOT / real_split / "images" / name
		if not src.exists():
			raise FileNotFoundError(f"Missing image: {name} in {split_name}")

		dst_img = img_out / name
		if not dst_img.exists():
			dst_img.symlink_to(src)

		lines = []
		for ann in anns_by_image.get(image_id, []):
			polygons = _ann_to_polygons(ann, int(h), int(w))
			for seg in polygons:
				points = []
				for i in range(0, len(seg), 2):
					x = max(0.0, min(1.0, float(seg[i]) / w))
					y = max(0.0, min(1.0, float(seg[i + 1]) / h))
					points.append(f"{x:.6f}")
					points.append(f"{y:.6f}")
				lines.append("0 " + " ".join(points))

		(lbl_out / f"{Path(name).stem}.txt").write_text(
			"\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
		)


def write_yaml() -> str:
	yaml_path = WORK_ROOT / "dataset.yaml"
	yaml_path.parent.mkdir(parents=True, exist_ok=True)
	yaml_path.write_text(
		"path: /kaggle/working/snack_seg_yolo\n"
		"train: images/train\n"
		"val: images/val\n"
		"test: images/test\n"
		"names:\n"
		"  0: Coconut_belt\n",
		encoding="utf-8",
	)
	return str(yaml_path)


def train() -> None:
	convert_split("train")
	convert_split("val")
	convert_split("test")
	data_yaml = write_yaml()

	model = YOLO(MODEL_PATH)
	model.train(
		task="segment",
		data=data_yaml,
		epochs=100,
		patience=20,
		imgsz=640,
		batch=8,
		verbose=True,
		project="/kaggle/working/runs",
		name="train_seg",
	)
	model.val(
		task="segment",
		data=data_yaml,
		split="test",
		project="/kaggle/working/runs",
		name="test_seg",
	)


if __name__ == "__main__":
	train()