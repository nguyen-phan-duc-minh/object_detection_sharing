import cv2
from detector import YOLODetector
# from counter import LineCounter
# from counter import ZoneCounter
from visualize import draw_boxes
from saver import ResultSaver

VIDEO_PATH = "inputs/videos/videoplayback.mp4"
VID_STRIDE = 3

def main():
    detector = YOLODetector(
        "models/yolo26n.pt",
        # conf=0.4,
        # iou=0.6,
        # classes=[0],  # chỉ detect person
        # max_det=50,
    )
    saver = ResultSaver(
        save_dir="outputs",
        save_frame=False,
        save_txt=False,
        save_crop=False,
    )
    # counter = LineCounter((100, 300), (500, 300))
    # counter = ZoneCounter((200, 200), (500, 500))
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_id = 0
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        if frame_id == 1 or frame_id % VID_STRIDE == 0:
            detections = detector.detect(frame)
            saver.save(frame, detections, frame_id)

        # counter.update(detections)
        frame = draw_boxes(frame, detections, detector.model.names)
        # frame = counter.draw(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()