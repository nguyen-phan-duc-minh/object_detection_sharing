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
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            max_det=self.max_det,
            device=self.device,
            persist=True, # giữ model trên GPU để tăng tốc cho các frame tiếp theo, neu khong co GPU thi cung khong sao
            tracker="bytetrack.yaml", # bytetrack và sort, sort chỉ lưu các object có conf > value. là 2 tracker phổ biến, có thể thử nghiệm để xem cái nào phù hợp với bài toán hơn, "bytetrack.yaml" là cấu hình mặc định của ByteTrack, có thể tùy chỉnh nếu cần
            # sử dụng tracker ByteTrack để gán ID cho các object, giúp theo dõi chúng qua các frame, "bytetrack.yaml" 
            # là cấu hình mặc định của ByteTrack, có thể tùy chỉnh nếu cần
            verbose=False,
        )[0]

        detections = []
        
        if results.boxes.id is None:
            return detections
        
        for box, track_id in zip(results.boxes, results.boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "class": cls,
                "track_id": int(track_id),
            })

        return detections
