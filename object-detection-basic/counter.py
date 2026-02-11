import cv2

class LineCounter:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.count = 0

    def update(self, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if self._crossed_line(cx, cy):
                self.count += 1

    def _crossed_line(self, x, y):
        return abs(y - self.start[1]) < 5

    def draw(self, frame):
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
        current_inside = set()

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]

            # tâm object
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

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