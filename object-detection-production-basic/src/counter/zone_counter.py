import cv2
import numpy as np
    
class ZoneCounter:
    def __init__(self, top_left, bottom_right, color=None):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.color = tuple(color) if color else (255, 0, 0)  # Màu mặc định: Blue
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
        # Hỗ trợ nhiều zones: points có thể là list of zones [[zone1_points], [zone2_points], ...]
        # hoặc một zone duy nhất [points]
        if len(points) > 0 and isinstance(points[0][0], (list, tuple)): # nếu nhiều zone [[(x1, y1), (x2, y2), ...], [(x3, y3), (x4, y4), ...], ...]
            # Nhiều zones - points là list 3 chiều
            # points = [
                # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], 
                # [(x5, y5), (x6, y6), (x7, y7), (x8, y8)], 
            # ...]
            self.zones = [np.array(zone, dtype=np.int32) for zone in points]
            self.is_multi_zone = True
        else:
            # Một zone duy nhất - points là list 2 chiều - [(x1, y1), (x2, y2), ...]
            self.zones = [np.array(points, dtype=np.int32)]
            self.is_multi_zone = False
        
        # Xử lý colors từ settings
        if colors is None:
            # Màu mặc định nếu không có trong settings
            # self.zones = [zone1, zone2, zone3, zone4] -> len(self.zones) = 4 -> 0,1,2,3 -> 4 màu lặp lại
            default_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            self.colors = [default_colors[i % len(default_colors)] for i in range(len(self.zones))]
        elif isinstance(colors[0], (list, tuple)):
            # Nhiều màu cho nhiều zones
            self.colors = [tuple(color) for color in colors]
        else:
            # Một màu duy nhất cho tất cả zones
            self.colors = [tuple(colors)] * len(self.zones)
            
        self.count = 0
        self.inside_ids = set()  # tránh đếm trùng
        self.zone_counts = [0] * len(self.zones)  # đếm riêng cho từng zone
        self.zone_inside_ids = [set() for _ in range(len(self.zones))]  # track IDs cho từng zone

        # tạo mask cho tất cả các zones
        h, w = frame_shape[:2]
        self.masks = []
        for zone_points in self.zones:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [zone_points], 255)
            self.masks.append(mask)

    def update(self, detections):
        current_inside = set()
        current_zone_inside = [set() for _ in range(len(self.zones))]
        self.centers = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.centers.append((cx, cy))
            
            # Kiểm tra từng zone
            for zone_idx, mask in enumerate(self.masks):
                # tránh out-of-bounds
                if cy >= mask.shape[0] or cx >= mask.shape[1]:
                    continue

                # nếu nằm trong zone mask này
                if mask[cy, cx] == 255:
                    current_zone_inside[zone_idx].add(track_id)
                    current_inside.add(track_id)

        # Cập nhật count cho từng zone
        # current_zone_inside = [
        #     {1, 5},        # zone 0 có object ID 1 và 5 -> self.zone_counts[0] = 2, self.zone_inside_ids[0] = {1,5}
        #     {2},           # zone 1 có object ID 2 -> self.zone_counts[1] = 1, self.zone_inside_ids[1] = {2}
        #     {3, 4, 6}      # zone 2 có 3 object -> self.zone_counts[2] = 3, self.zone_inside_ids[2] = {3, 4, 6}   
        # ]
        for zone_idx in range(len(self.zones)):
            self.zone_counts[zone_idx] = len(current_zone_inside[zone_idx])
            self.zone_inside_ids[zone_idx] = current_zone_inside[zone_idx]
            
        # Tổng count của tất cả zones
        self.count = len(current_inside)
        self.inside_ids = current_inside

    def draw(self, frame):
        overlay = frame.copy()
        
        for cx, cy in self.centers:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        
        # Vẽ tất cả các zones với màu từ settings
        for zone_idx, zone_points in enumerate(self.zones):
            # Sử dụng modulo để cycle màu nếu không đủ
            color = self.colors[zone_idx % len(self.colors)]
            
            # tô vùng zone mờ
            cv2.fillPoly(overlay, [zone_points], color)
            
            # vẽ viền zone
            cv2.polylines(frame, [zone_points], True, color, 2)
            
            # Text counter cho từng zone (nếu có nhiều zones)
            if self.is_multi_zone:
                # Tính vị trí text ở giữa zone
                # Moments giúp ta tính: 
                #   + Diện tích 
                #   + Tâm (centroid) 
                #   + Các đặc trưng hình học
                # M = {
                #   "m00": diện tích,
                #   "m10": tổng x có trọng số,
                #   "m01": tổng y có trọng số,
                #   ...
                # }
                # Công thức tọa độ tâm:
                #   + cx = int(M["m10"] / M["m00"])
                #   + cy = int(M["m01"] / M["m00"])     
                M = cv2.moments(zone_points)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        frame,
                        f"Zone {zone_idx+1}: {self.zone_counts[zone_idx]}",
                        (cx-50, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
        
        frame = cv2.addWeighted(overlay, 0.1, frame, 1.0, 0)

        return frame