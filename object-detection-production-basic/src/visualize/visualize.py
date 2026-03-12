import cv2

def draw_boxes(frame, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        conf = detection["conf"]
        cls_id = detection["class"]

        label = class_names[cls_id]
        
        text = f"{label} - {detection['track_id']} - {conf:.2f}"
        if 'speed' in detection and detection['speed'] > 0:
            text += f" - {detection['speed']:.1f} km/h"
        
        # Kiểm tra xem xe có vượt quá tốc độ không
        is_speeding = detection.get('speeding', False)
        
        # Màu bounding box: ĐỎ nếu vượt quá tốc độ, XANH DA TRỜI nếu bình thường
        box_color = (0, 0, 255) if is_speeding else (255, 0, 0)  # BGR: Red or Blue
        
        # Thêm thông tin vi phạm vào text nếu vượt quá tốc độ
        if is_speeding:
            max_speed = detection.get('max_speed', 0)
            text += f" - VUOT QUA {max_speed} km/h!"

        # Vẽ box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2) # frame, vi tri bat dau, vi tri ket thuc, mau sac, do day

        # Kích thước text
        # (w, h), _ la de bo phan goc
        (width_text, height_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # text, font, scale, do day

        # Background text
        cv2.rectangle(frame, (x1, y1 - height_text - 6), (x1 + width_text, y1), box_color, -1) # frame, vi tri bat dau, vi tri ket thuc, mau sac, do day (-1 de fill background)

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
        ) # frame, text, vi tri text, font, scale, color, do day, mau sac, line type

    return frame
