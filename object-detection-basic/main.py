import cv2
from detector import YOLODetector
from counter import LineCounter
from counter import ZoneCounter
from counter import LaneZoneCounter
from visualize import draw_boxes
from saver import ResultSaver

VIDEO_PATH = "inputs/videos/videoplayback.mp4"
VID_STRIDE = 1

def main():
    detector = YOLODetector(
        "models/yolo26n.pt",
        # conf=0.4,
        # iou=0.6,
        classes=[2],  # chỉ detect person
        # max_det=50,
    )
    saver = ResultSaver(
        save_dir="outputs",
        save_frame=False,
        save_txt=False,
        save_crop=False,
    )
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        return
    
    counter = LineCounter((100, 200), (500, 200))
    # counter = ZoneCounter((100, 100), (500, 300))
    # points = [(180, 100), (400, 100), (550, 300), (50, 300)]
    # counter = LaneZoneCounter(points, frame.shape)
    frame_id = 0
    detections = []

    while True:
        frame_id += 1
        
        if frame_id == 1 or frame_id % VID_STRIDE == 0:
            detections = detector.detect(frame)
            saver.save(frame, detections, frame_id)

        counter.update(detections)
        frame = draw_boxes(frame, detections, detector.model.names)
        frame = counter.draw(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) == 27:
            break
        
        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()