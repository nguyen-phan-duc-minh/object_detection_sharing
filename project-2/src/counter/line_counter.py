import cv2

class LineCounter:
    def __init__(self, start, end, color=None):
        self.start = start # (x, y) start[1] la lay y ra
        self.end = end
        self.color = tuple(color) if color else (0, 255, 0)  # Màu mặc định: Green
        self.count = 0
        self.counted_ids = set() # lưu track_id của các object đã đếm để tránh đếm trùng
        self.prev_side_by_id = {}

    def update(self, detections):
        self.centers = [] # [(300,500) , (400, 600)]
        active_ids = set()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]
            cx = int((x1 + x2) / 2) 
            cy = int((y1 + y2) / 2)
            active_ids.add(track_id)

            self.centers.append((cx, cy))
            
            prev_side = self.prev_side_by_id.get(track_id)
            curr_side = cy - self.start[1]
            
            crossed = False
            if prev_side is not None:
                crossed = (prev_side < 0 <= curr_side) or (prev_side > 0 >= curr_side)
                
            if crossed and track_id not in self.counted_ids:
                self.count += 1
                self.counted_ids.add(track_id)
                
            self.prev_side_by_id[track_id] = curr_side
        
        stale_ids = [tid for tid in self.prev_side_by_id if tid not in active_ids]
        for tid in stale_ids:
            del self.prev_side_by_id[tid]
            
    def _crossed_line(self, x, y):
        return abs(y - self.start[1]) < 5

    def draw(self, frame):
        for cx, cy in self.centers:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
            
        cv2.line(frame, self.start, self.end, self.color, 4)
        cv2.putText(
            frame,
            f"Count: {self.count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            3, # scale
            self.color,
            5, # do day
        )
        return frame
