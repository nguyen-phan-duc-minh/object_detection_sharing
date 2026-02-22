import cv2
import numpy as np

class LineCounter:
    def __init__(self, start, end):
        self.start = start # (x, y) start[1] la lay y ra
        self.end = end
        self.count = 0

    def update(self, detections):
        self.centers = [] # [(300,500) , (400, 600)]
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cx = int((x1 + x2) / 2) 
            cy = int((y1 + y2) / 2)

            self.centers.append((cx, cy))
            
            if self._crossed_line(cx, cy):
                self.count += 1

    def _crossed_line(self, x, y):
        return abs(y - self.start[1]) < 5

    def draw(self, frame):
        for cx, cy in self.centers:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            
        cv2.line(frame, self.start, self.end, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Count: {self.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return frame
    
class ZoneCounter:
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.count = 0
        self.inside_ids = set()  # tránh đếm trùng

    def update(self, detections):
        current_inside = set() # lưu id của các object đang nằm trong zone ở frame hiện tại
        self.centers = []

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            self.centers.append((cx, cy))

            if self._inside_zone(cx, cy):
                current_inside.add(i)

                # chỉ đếm khi object mới vào zone
                if i not in self.inside_ids:
                    self.count += 1

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
            (255, 0, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Zone Count: {self.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        return frame
    
class LaneZoneCounter:
    def __init__(self, points, frame_shape):
        self.points = np.array(points, dtype=np.int32) # np.array de chuyen list thanh array, dtype=np.int32 de chuyen cac toa do thanh so nguyen
        self.count = 0
        self.inside_ids = set()  # tránh đếm trùng

        # tạo mask theo kích thước frame
        h, w = frame_shape[:2] # [:2] la de lay height va width thoi, khong can den so kenh
        self.mask = np.zeros((h, w), dtype=np.uint8) # tao mask den, co cung kich thuoc voi frame, dtype=np.uint8 de chuyen mask thanh so nguyen 8 bit (0-255)

        # tô polygon vào mask
        cv2.fillPoly(self.mask, [self.points], 255) # fillPoly de to polygon vao mask, [self.points] de chuyen points thanh dang list de fillPoly co the nhan, 255 la gia tri de to (trang)

    def update(self, detections):
        current_inside = set()
        self.centers = []

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            self.centers.append((cx, cy))
            
            # tránh out-of-bounds
            if cy >= self.mask.shape[0] or cx >= self.mask.shape[1]:
                continue

            # nếu nằm trong lane mask
            if self.mask[cy, cx] == 255:
                current_inside.add(i)

                # chỉ đếm khi object mới vào lane
                if i not in self.inside_ids:
                    self.count += 1

        self.inside_ids = current_inside

    def draw(self, frame):
        overlay = frame.copy() # tạo bản sao của frame để vẽ overlay (vùng lane mờ) mà không làm thay đổi frame gốc
        
        for cx, cy in self.centers:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # tô vùng lane mờ
        cv2.fillPoly(overlay, [self.points], (255, 0, 0)) # tô polygon vào overlay, (255, 0, 0) là màu xanh dương
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0) # kết hợp overlay với frame gốc, 0.2 là alpha của overlay (độ mờ), 0.8 là alpha của frame gốc,

        # vẽ viền lane
        cv2.polylines(frame, [self.points], True, (255, 0, 0), 2) # vẽ viền polygon, True là để đóng polygon, (255, 0, 0) là màu xanh dương, 2 là độ dày của viền

        # text counter
        cv2.putText(
            frame,
            f"Lane Count: {self.count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        return frame