import cv2

class LineCounter:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.count = 0

    def update(self, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
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