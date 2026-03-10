import cv2 
from pathlib import Path
import logging

from src.detector.detector import YOLODetector
from src.saver import ResultSaver
from src.visualize.visualize import draw_boxes
from src.counter.line_counter import LineCounter
from src.counter.zone_counter import ZoneCounter, LaneZoneCounter
from src.counter.speed import SpeedEstimator

logger = logging.getLogger(__name__)

class DetectionPipeline:
    def __init__(self, config):
        self.config = config
        
        model_path = Path(config['data']['model']) / f"{config['detector']['type']}.pt"
        self.detector = YOLODetector(
            weight_path=str(model_path),
            conf=config['detector']['conf_threshold'],
            iou=config['detector']['iou_threshold'],
            classes=config['detector'].get('classes', None),
            max_det=config['detector'].get('max_det', 100),
            device=config['detector'].get('device', None),
        )
        
        self.saver = ResultSaver(
            save_dir=config['data']['output'],
            save_frame=config['saver'].get('save_images', True),
            save_txt=config['saver'].get('save_detections', False),
            save_crop=config['saver'].get('save_crops', False),
        )
        
        self.counter = None
        
        # Initialize speed estimator if enabled
        self.speed_estimator = None
        if config.get('speed', {}).get('enabled', False):
            self.speed_estimator = SpeedEstimator(
                fps=config['speed'].get('fps', 30),
                pixel_per_meter=config['speed'].get('pixel_per_meter', 21.7),
                smooth_window=config['speed'].get('smooth_window', 5),
            )
            logger.info("SpeedEstimator initialized")
        
        logger.info("Detection pipeline initialized")
        
    def initialize_counter(self, counter_type, **kwargs):
        if counter_type == 'line':
            start = kwargs.get('start', (100, 200))
            end = kwargs.get('end', (500, 200))
            color = kwargs.get('color', None)
            self.counter = LineCounter(start, end, color=color)
            logger.info(f"LineCounter initialized: {start} to {end}")
            
        elif counter_type == 'zone':
            top_left = kwargs.get('top_left', (100, 100))
            bottom_right = kwargs.get('bottom_right', (500, 300))
            color = kwargs.get('color', None)
            self.counter = ZoneCounter(top_left, bottom_right, color=color)
            logger.info(f"ZoneCounter initialized: {top_left} to {bottom_right}")
            
        elif counter_type == 'lane':
            points = kwargs.get('points', [(180, 100), (400, 100), (550, 300), (50, 300)])
            frame_shape = kwargs.get('frame_shape')
            colors = kwargs.get('colors', None)
            if frame_shape is None:
                raise ValueError("frame_shape is required for LaneZoneCounter")
            self.counter = LaneZoneCounter(points, frame_shape, colors=colors)
            num_zones = len(points) if isinstance(points[0][0], (list, tuple)) else 1
            logger.info(f"LaneZoneCounter initialized with {num_zones} zone(s)")
            
        else:
            raise ValueError(f"Unknown counter type: {counter_type}")
        
    def run_video(self, video_path, counter_type='lane', counter_kwargs=None, vid_stride=1, display=True):
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Failed to read video: {video_path}")
            return
        
        if counter_kwargs is None:
            counter_kwargs = {}
        counter_kwargs['frame_shape'] = frame.shape
        self.initialize_counter(counter_type, **counter_kwargs)
        
        frame_id = 0
        detections = []

        while True:
            frame_id += 1
            
            if frame_id == 1 or frame_id % vid_stride == 0:
                detections = self.detector.detect(frame)
                
                if self.speed_estimator:
                    detections = self.speed_estimator.update(detections)
                
                if self.config['saver'].get('save_images', False) or self.config['saver'].get('save_detections', False) or self.config['saver'].get('save_crops', False):
                    self.saver.save(frame, detections, frame_id)

            if self.counter:
                self.counter.update(detections)
                
            frame = draw_boxes(frame, detections, self.detector.model.names)
            
            if self.counter:
                frame = self.counter.draw(frame)
            
            if display:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) == 27:
                    logger.info("Video processing interrupted by user")
                    break
            
            ret, frame = cap.read()
            if not ret:
                break

        cap.release()
        if display:
            cv2.destroyAllWindows()
            
        logger.info("Video processing completed")
        if self.counter:
            logger.info(f"Final count: {self.counter.count}")