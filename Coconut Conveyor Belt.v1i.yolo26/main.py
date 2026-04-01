from pathlib import Path
from ultralytics import YOLO

def train() -> None:
	root = Path(__file__).resolve().parent
	data_yaml = root / "data.yaml"
	model_path = root / "models" / "yolo26n.pt"
	model = YOLO(str(model_path))
	model.train(
		data=str(data_yaml),
		epochs=100,
		patience=20,
		imgsz=640,
		batch=16,
		verbose=True,
		project=str(root / "runs"),
		name="train",
	)

	model.val(
		data=str(data_yaml),
		split="test",
		project=str(root / "runs"),
		name="test",
	)

if __name__ == "__main__":
	train()