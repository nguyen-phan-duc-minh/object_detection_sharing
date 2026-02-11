import cv2

def draw_boxes(frame, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        conf = detection["conf"]
        cls_id = detection["class"]

        label = class_names[cls_id]
        text = f"{label} {conf:.2f}"

        # Vẽ box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Kích thước text
        (w, h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        # Background text
        cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w, y1), (255, 0, 0), -1)

        # Text
        cv2.putText(
            frame,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame
