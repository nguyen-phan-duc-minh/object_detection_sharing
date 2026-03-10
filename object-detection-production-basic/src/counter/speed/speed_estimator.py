import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class SpeedEstimator:
    """
    Estimator để tính vận tốc của các đối tượng được tracking.
    
    Logic:
    1. Lưu vị trí (x, y) của mỗi track_id qua các frame
    2. Tính khoảng cách di chuyển giữa các frame (pixels)
    3. Chuyển đổi sang km/h dựa trên:
       - FPS video
       - Pixel-to-meter ratio (cần calibration)
    
    Ví dụ calibration:
    - Video 1920x1080, góc nhìn từ trên, chiều cao 50m
    - => 1 pixel ~ 50/1080 = 0.046 meters
    - hoặc đo đường thực tế: 10m road / 217 pixels = 0.046 m/pixel
    """
    
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
            
            # Giữ history trong phạm vi smooth_window
            max_history = max(self.smooth_window * 2, 10)  # Ít nhất 10 frame
            if len(self.track_history[track_id]) > max_history:
                self.track_history[track_id].pop(0)
            
            # Tính speed
            speed = self._calculate_speed(track_id)
            self.current_speeds[track_id] = speed
            
            # Thêm speed vào detection
            detection['speed'] = speed
            
        return detections
    
    def _calculate_speed(self, track_id):
        """
        Tính speed cho một track_id cụ thể.
        
        Method:
        1. Lấy N vị trí gần nhất (N = smooth_window)
        2. Tính tổng khoảng cách di chuyển
        3. Tính thời gian: N frames / FPS
        4. Speed (m/s) = distance / time
        5. Speed (km/h) = speed_m_s * 3.6
        """
        history = self.track_history[track_id]
        
        # Cần ít nhất 2 điểm để tính speed
        if len(history) < 2:
            return 0.0
        
        # Lấy smooth_window điểm gần nhất
        recent_points = history[-min(len(history), self.smooth_window):] # vi du: lay tu frame 10 den frame 14 neu smooth_window=5
        
        if len(recent_points) < 2:
            return 0.0
        
        # Tính tổng khoảng cách (pixels)
        total_distance_px = 0
        for i in range(1, len(recent_points)):
            x1, y1, _ = recent_points[i-1]
            x2, y2, _ = recent_points[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance_px += distance
        
        # Tính thời gian (seconds)
        num_intervals = len(recent_points) - 1
        time_seconds = num_intervals / self.fps
        
        if time_seconds == 0:
            return 0.0
        
        # Chuyển đổi distance sang meters
        distance_meters = total_distance_px / self.pixel_per_meter
        
        # Tính speed (m/s)
        speed_m_s = distance_meters / time_seconds
        
        # Chuyển sang km/h
        speed_kmh = speed_m_s * 3.6
        
        return round(speed_kmh, 1)
    
    def get_speed(self, track_id):
        """
        Lấy speed hiện tại của một track_id.
        
        Returns:
            Speed in km/h, or 0 if not available
        """
        return self.current_speeds.get(track_id, 0.0)
    
    def reset(self):
        """Reset tất cả tracking data"""
        self.track_history.clear()
        self.current_speeds.clear()
        self.frame_count = 0
        logger.info("SpeedEstimator reset")
