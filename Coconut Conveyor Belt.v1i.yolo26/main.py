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
		patience=20, # dừng sớm - early stopping - nếu sau 20 epochs mà không cải thiện độ chính xác trên tập validation thì sẽ dừng việc huấn luyện
		imgsz=640,
		batch=2, # kích thước batch, số lượng mẫu được xử lý trước khi cập nhật trọng số của mô hình. Batch size lớn hơn có thể giúp tăng tốc độ huấn luyện nhưng cũng yêu cầu nhiều bộ nhớ hơn.
		verbose=True, # hiển thị thông tin chi tiết về quá trình huấn luyện, bao gồm loss, độ chính xác và các chỉ số khác sau mỗi epoch.
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