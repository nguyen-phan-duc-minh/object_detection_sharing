import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class SpeedEstimator:
    def __init__(self, fps=30, pixel_per_meter=21.7, smooth_window=5):
        self.fps = fps
        self.pixel_per_meter = pixel_per_meter
        self.smooth_window = smooth_window
        
        # Lưu lịch sử vị trí: {track_id: [(x1, y1, frame1), (x2, y2, frame2), ...]}
        self.track_history = defaultdict(list)
        
        # Lưu speed hiện tại: {track_id: speed_kmh}
        self.current_speeds = {}
        
        self.frame_count = 0
        
        logger.info(f"SpeedEstimator initialized: fps={fps}, pixel_per_meter={pixel_per_meter}, smooth_window={smooth_window}")
        
    def update(self, detections):
        self.frame_count += 1
        
        for detection in detections:
            track_id = detection['track_id']
            x1, y1, x2, y2 = detection['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            self.track_history[track_id].append((cx, cy, self.frame_count))
            
            max_history = max(self.smooth_window * 2, 10)
            if len(self.track_history[track_id]) > max_history:
                self.track_history[track_id].pop(0)
                
            speed = self._calculate_speed(track_id)
            self.current_speeds[track_id] = speed
            
            detection['speed'] = speed
            
        return detections

    def _calculate_speed(self, track_id):
        history = self.track_history[track_id]
        
        if len(history) < 2:
            return 0.0
        
        recent_points = history[-min(self.smooth_window, len(history)):]
        
        if len(recent_points) < 2:
            return 0.0
        
        total_distance_px = 0
        for i in range(1, len(recent_points)):
            x1, y1, _ = recent_points[i-1]
            x2, y2, _ = recent_points[i]
            distance_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_distance_px += distance_px
            
        distance_meters = total_distance_px / self.pixel_per_meter
        
        # thoi gian
        num_intervals = len(recent_points) - 1
        time_seconds = num_intervals / self.fps
        
        if time_seconds == 0:
            return 0.0

        speed_m_s = distance_meters / time_seconds
        speed_kmh = speed_m_s * 3.6

        return round(speed_kmh, 1)
    
    def get_speed(self, track_id):
        return self.current_speeds.get(track_id, 0.0)
    
    def reset(self):
        self.track_history.clear()
        self.current_speeds.clear()
        self.frame_count = 0
        logger.info("SpeedEstimator reset: cleared history and speeds.")