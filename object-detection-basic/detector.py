from ultralytics import YOLO

class YOLODetector:
    def __init__(
        self,
        weight_path: str,
        conf: float = 0.5,
        iou: float = 0.7,
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
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class": cls,
            })

        return detections
