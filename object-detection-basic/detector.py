from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weight_path: str):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class": cls
            })

        return detections