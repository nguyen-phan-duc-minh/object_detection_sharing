import cv2
from detector import YOLODetector
from counter import LineCounter
from visualize import draw_boxes

VIDEO_PATH = "inputs/videos/sample.mp4"

def main():
    detector = YOLODetector("models/yolo26n.pt")
    counter = LineCounter((100, 300), (500, 300))
    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        counter.update(detections)
        frame = draw_boxes(frame, detections)
        frame = counter.draw(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()