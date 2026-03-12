import cv2
import numpy as np

class ZoneCounter:
    def __init__(self, top_left, bottom_right, color=None):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.color = tuple(color) if color else (0, 255, 0)  # Màu mặc định: Green
        self.count = 0
        self.inside_ids = set()  # tránh đếm trùng

    def update(self, detections):
        current_inside = set()
        self.centers = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.centers.append((cx, cy))

            if self._inside_zone(cx, cy):
                current_inside.add(track_id)

        self.count = len(current_inside)
        self.inside_ids = current_inside

    def _inside_zone(self, x, y):
        return (
            self.top_left[0] <= x <= self.bottom_right[0]
            and self.top_left[1] <= y <= self.bottom_right[1]
        )

    def draw(self, frame):
        for cx, cy in self.centers:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
            
        cv2.rectangle(
            frame,
            self.top_left,
            self.bottom_right,
            self.color,
            2,
        )

        cv2.putText(
            frame,
            f"Zone Count: {self.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.color,
            2,
        )

        return frame
    
class LaneZoneCounter:
    def __init__(self, points, frame_shape, colors=None):
        if len(points) > 0 and isinstance(points[0][0], (list, tuple)):
            self.zones = [np.array(zone, dtype=np.int32) for zone in points]
            self.is_multi_zone = True
        else: # points[0]
            self.zones = [np.array(points, dtype=np.int32)] # np.array de chuyen list thanh array, dtype=np.int32 de chuyen cac toa do thanh so nguyen
            self.is_multi_zone = False
        
        if colors is None:
            default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            self.colors = [default_colors[i % len(default_colors)] for i in range(len(self.zones))]
        elif isinstance(colors[0], (list, tuple)):
            self.colors = [tuple(color) for color in colors]
        else:
            self.colors = [tuple(colors)] * len(self.zones)
        
        self.count = 0
        self.inside_ids = set()  # tránh đếm trùng
        self.zone_counts = [0] * len(self.zones) # dem so luong object trong moi zone (neu co nhieu zone)
        self.zone_inside_ids = [set() for _ in range(len(self.zones))] # luu track_id cua cac object da dem trong moi zone (neu co nhieu zone)

        # tạo mask theo kích thước frame
        h, w = frame_shape[:2] # [:2] la de lay height va width thoi, khong can den so kenh
        self.mask = [] # tao mask den, co cung kich thuoc voi frame, dtype=np.uint8 de chuyen mask thanh so nguyen 8 bit (0-255)

        for zone_points in self.zones:
            zone_mask = np.zeros((h, w), dtype=np.uint8) # tao mask den moi zone
            # tô polygon vào mask
            cv2.fillPoly(zone_mask, [zone_points], 255) # fillPoly de to polygon vao mask, [zone_points] de chuyen points thanh dang list de fillPoly co the nhan, 255 la gia tri de to (trang)
            self.mask.append(zone_mask)

    def update(self, detections):
        current_inside = set()
        current_zone_inside = [set() for _ in range(len(self.zones))] # luu track_id cua cac object hien dang o trong moi zone (neu co nhieu zone)
        self.centers = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.centers.append((cx, cy))
                
            for zone_idx, mask in enumerate(self.mask):
                # tránh out-of-bounds
                if cy >= mask.shape[0] or cx >= mask.shape[1]:
                    continue

                # nếu nằm trong lane mask
                if mask[cy, cx] == 255:
                    current_zone_inside[zone_idx].add(track_id) # them track_id vao zone hien tai
                    current_inside.add(track_id) # them track_id vao zone chung (de dem tong so luong trong lane)

        for zone_idx in range(len(self.zones)):
            self.zone_counts[zone_idx] = len(current_zone_inside[zone_idx]) # cap nhat count cho moi zone
            self.zone_inside_ids[zone_idx] = current_zone_inside[zone_idx] # cap nhat track_id da dem cho moi zone
            
        self.count = len(current_inside)
        self.inside_ids = current_inside

    def draw(self, frame):
        overlay = frame.copy() # tạo bản sao của frame để vẽ overlay (vùng lane mờ) mà không làm thay đổi frame gốc
        
        for cx, cy in self.centers:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        for zone_idx, zone_points in enumerate(self.zones):
            colors = self.colors[zone_idx % len(self.colors)] # lấy màu tương ứng cho zone hiện tại
            # tô vùng lane mờ
            cv2.fillPoly(overlay, [zone_points], colors) # tô polygon vào overlay, colors là màu của zone
            
            # vẽ viền lane
            cv2.polylines(frame, [zone_points], True, colors, 2) # vẽ viền polygon, True là để đóng polygon, colors là màu của zone, 2 là độ dày của viền

            # text counter
            if self.is_multi_zone:
                M = cv2.moments(zone_points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        frame,
                        f"Zone {zone_idx + 1}: {self.zone_counts[zone_idx]}",
                        (cx-50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colors,
                        2,
                    )
                    
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0) # gộp overlay với frame gốc để tạo hiệu ứng mờ cho vùng lane   

        return frame
