import cv2
from detector import YOLODetector
from counter import LineCounter
from counter import ZoneCounter
from counter import LaneZoneCounter
from visualize import draw_boxes
from saver import ResultSaver

VIDEO_PATH = "inputs/videos/video_playback_2.mp4"
VID_STRIDE = 1  # Xử lý mỗi 2 frame để tăng tốc

def main():
    detector = YOLODetector(
        "models/best.pt",
        conf=0.5,  # Mặc định đã tốt, có thể uncomment để tùy chỉnh
        iou=0.5,
        classes=None,  # Model custom: để None để detect tất cả class trong model
        max_det=100,
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
    
    # Counter với tọa độ phù hợp 1920x1080
    counter = LineCounter((300, 600), (1600, 600))  # Line ngang giữa frame
    # counter = ZoneCounter((300, 200), (1600, 900))  # Zone hình chữ nhật lớn
    # points = [(500, 200), (1400, 200), (1800, 900), (100, 900)]  # Lane Zone toàn bộ đường
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