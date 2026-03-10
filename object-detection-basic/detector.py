from ultralytics import YOLO

class YOLODetector:
    def __init__(
        self,
        weight_path: str,
        conf: float = 0,  # Giảm từ 0.5 xuống 0.25 để detect xe nhỏ tốt hơn
        iou: float = 0,    # Giảm từ 0.7 xuống 0.5
        classes: list | None = None,
        max_det: int = 100,
        device: str | None = None,
    ):
        self.model = YOLO(weight_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.max_det = max_det
        self.device = device

    def detect(self, frame):
        # Đổi từ track() sang predict() để tăng tốc đáng kể (nhanh hơn 2-3 lần)
        results = self.model.predict(
            frame,
            imgsz=640,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        
        if len(results.boxes) == 0:
            return detections
        
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class": cls,
                "track_id": i,  # Tạm dùng index vì không có tracking
            })

        return detections
