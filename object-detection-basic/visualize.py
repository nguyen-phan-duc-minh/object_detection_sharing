import cv2

def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame