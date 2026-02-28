import cv2

class LineCounter:
    def __init__(self, start, end):
        self.start = start # (x, y) start[1] la lay y ra
        self.end = end
        self.count = 0
        self.counted_ids = set() # lưu track_id của các object đã đếm để tránh đếm trùng

    def update(self, detections):
        self.centers = [] # [(300,500) , (400, 600)]
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            track_id = detection["track_id"]
            cx = int((x1 + x2) / 2) 
            cy = int((y1 + y2) / 2)

            self.centers.append((cx, cy))
            
            if self._crossed_line(cx, cy) and track_id not in self.counted_ids:
                self.count += 1
                self.counted_ids.add(track_id)

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
