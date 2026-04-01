import os
import cv2

class ResultSaver:
    def __init__(
        self,
        save_dir="outputs",
        save_frame=True,
        save_txt=False,
        save_crop=False,
    ):
        self.save_dir = save_dir
        self.save_frame_flag = save_frame
        self.save_txt_flag = save_txt
        self.save_crop_flag = save_crop

        os.makedirs(save_dir, exist_ok=True)

    def save(self, frame, detections, frame_id):
        if self.save_frame_flag:
            self._save_frame(frame, frame_id)

        if self.save_txt_flag:
            self._save_txt(detections, frame_id)

        if self.save_crop_flag:
            self._save_crop(frame, detections, frame_id)

    def _save_frame(self, frame, frame_id):
        path = f"{self.save_dir}/frame_{frame_id:05d}.jpg"
        cv2.imwrite(path, frame)

    def _save_txt(self, detections, frame_id):
        path = f"{self.save_dir}/frame_{frame_id:05d}.txt"

        with open(path, "w") as f:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cls = det["class"]
                conf = det["conf"]

                f.write(f"{cls} {conf:.3f} {x1} {y1} {x2} {y2}\n")

    def _save_crop(self, frame, detections, frame_id):
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = frame[y1:y2, x1:x2]

            path = f"{self.save_dir}/crop_{frame_id:05d}_{i}.jpg"
            cv2.imwrite(path, crop)
