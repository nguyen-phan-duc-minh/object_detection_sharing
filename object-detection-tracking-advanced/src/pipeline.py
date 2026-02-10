"""
Detection Pipeline Module
Main pipeline orchestrating detection, tracking, and counting
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from core.logging_setup import get_logger
from core.settings_loader import SettingsLoader
from src.detection.detector import ObjectDetector
from src.tracking.tracker import ObjectTracker
from src.counting.line_counter import MultiLineCounter
from src.counting.zone_counter import MultiZoneCounter
from src.visualization.draw import Visualizer

logger = get_logger("pipeline")


class DetectionPipeline:
    """Main detection pipeline for video/image processing"""
    
    def __init__(self, settings: SettingsLoader):
        """
        Initialize detection pipeline
        
        Args:
            settings: Settings loader instance
        """
        self.settings = settings
        
        # Initialize components
        logger.info("Initializing detection pipeline...")
        self._initialize_components()
        
        # Performance tracking
        self.fps = 0.0
        self.frame_count = 0
        self.processing_times = []
        
        logger.info("Detection pipeline ready")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components"""
        # Detector
        model_config = self.settings.get_section('model')
        self.detector = ObjectDetector(
            weights_path=model_config.get('weights_path'),
            confidence_threshold=model_config.get('confidence_threshold', 0.3),
            iou_threshold=model_config.get('iou_threshold', 0.5),
            device=model_config.get('device', 'cpu'),
            classes=model_config.get('classes'),
            half_precision=self.settings.get('performance.use_half_precision', False)
        )
        
        # Tracker
        tracking_config = self.settings.get_section('tracking')
        self.tracker = ObjectTracker(
            tracker_type=tracking_config.get('tracker_type', 'bytetrack'),
            track_activation_threshold=tracking_config.get('track_activation_threshold', 0.25),
            lost_track_buffer=tracking_config.get('lost_track_buffer', 30),
            minimum_matching_threshold=tracking_config.get('minimum_matching_threshold', 0.8),
            minimum_consecutive_frames=tracking_config.get('minimum_consecutive_frames', 1)
        )
        
        # Counters
        counting_config = self.settings.get_section('counting')
        
        # Line counters
        if counting_config.get('line_counting', {}).get('enabled', False):
            line_configs = counting_config['line_counting'].get('lines', [])
            self.line_counter = MultiLineCounter(line_configs)
            logger.info(f"Initialized {len(self.line_counter)} line counters")
        else:
            self.line_counter = None
        
        # Zone counters
        if counting_config.get('zone_counting', {}).get('enabled', False):
            zone_configs = counting_config['zone_counting'].get('zones', [])
            self.zone_counter = MultiZoneCounter(zone_configs)
            logger.info(f"Initialized {len(self.zone_counter)} zone counters")
        else:
            self.zone_counter = None
        
        # Visualizer
        viz_config = self.settings.get_section('visualization')
        self.visualizer = Visualizer(
            thickness=viz_config.get('thickness', 2),
            text_scale=viz_config.get('text_scale', 0.6),
            text_thickness=viz_config.get('text_thickness', 2),
            colors=viz_config.get('colors')
        )
        
        # Get class names
        self.class_names = {
            class_id: self.detector.get_class_name(class_id)
            for class_id in (self.settings.get('model.classes') or [])
        }
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, Dict]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        start_time = time.time()
        
        # Resize frame if configured
        if self.settings.get('performance.resize_frame', False):
            target_w = self.settings.get('performance.target_width', 1280)
            target_h = self.settings.get('performance.target_height', 720)
            frame = cv2.resize(frame, (target_w, target_h))
        
        # Detection
        detections = self.detector.detect(frame)
        
        # Tracking
        tracked_detections = self.tracker.update(detections)
        
        # Counting
        line_counts = {}
        zone_counts = {}
        
        if self.line_counter:
            line_counts = self.line_counter.update(tracked_detections)
        
        if self.zone_counter:
            zone_counts = self.zone_counter.update(tracked_detections)
        
        # Visualization
        annotated_frame = frame.copy()
        
        # Draw detections
        annotated_frame = self.visualizer.draw_detections(
            annotated_frame,
            tracked_detections,
            class_names=self.class_names,
            show_confidence=True,
            show_trace=True
        )
        
        # Draw counting zones/lines
        if self.line_counter:
            for line in self.line_counter:
                start, end = line.get_line_coordinates()
                annotated_frame = self.visualizer.draw_line(
                    annotated_frame, start, end, label=line.name
                )
        
        if self.zone_counter:
            for zone in self.zone_counter:
                annotated_frame = self.visualizer.draw_zone(
                    annotated_frame,
                    zone.get_polygon(),
                    label=zone.name,
                    fill=True
                )
        
        # Draw count info
        all_counts = {**line_counts, **zone_counts}
        if all_counts and self.settings.get('visualization.display_count', True):
            annotated_frame = self.visualizer.draw_count_info(
                annotated_frame, all_counts
            )
        
        # Draw FPS
        if self.settings.get('visualization.display_fps', True):
            annotated_frame = self.visualizer.draw_fps(annotated_frame, self.fps)
        
        # Update FPS
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        self.fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        self.frame_count += 1
        
        # Prepare results
        results = {
            'frame_number': self.frame_count,
            'fps': round(self.fps, 2),
            'detections': len(tracked_detections),
            'line_counts': line_counts,
            'zone_counts': zone_counts,
            'processing_time_ms': round(processing_time * 1000, 2)
        }
        
        return annotated_frame, results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process video file
        
        Args:
            video_path: Input video path
            output_path: Output video path (optional)
            
        Returns:
            Processing results summary
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path and self.settings.get('output.save_video', True):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*self.settings.get('output.output_video_codec', 'mp4v'))
            output_fps = self.settings.get('output.output_video_fps', fps)
            
            # Adjust size if resizing
            if self.settings.get('performance.resize_frame', False):
                width = self.settings.get('performance.target_width', width)
                height = self.settings.get('performance.target_height', height)
            
            writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
            logger.info(f"Output video: {output_path}")
        
        # Process frames
        frame_results = []
        skip_frames = self.settings.get('performance.skip_frames', 0)
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if configured
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue
                
                # Process frame
                annotated_frame, results = self.process_frame(frame)
                frame_results.append(results)
                
                # Write output
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                if self.settings.get('visualization.show_display', True):
                    cv2.imshow('Detection Pipeline', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User interrupted processing")
                        break
                
                frame_idx += 1
                
                # Progress log
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames ({self.fps:.1f} FPS)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Generate summary
        summary = self._generate_summary(frame_results)
        
        # Save reports
        if self.settings.get('output.save_reports', True):
            self._save_reports(summary)
        
        logger.info(f"Processing complete: {frame_idx} frames processed")
        return summary
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Process single image
        
        Args:
            image_path: Input image path
            output_path: Output image path (optional)
            
        Returns:
            Processing results
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Process
        annotated_frame, results = self.process_frame(frame)
        
        # Save output
        if output_path and self.settings.get('output.save_images', True):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_frame)
            logger.info(f"Saved output: {output_path}")
        
        # Display
        if self.settings.get('visualization.show_display', True):
            cv2.imshow('Detection Result', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def _generate_summary(self, frame_results: list) -> Dict:
        """Generate processing summary"""
        if not frame_results:
            return {}
        
        total_detections = sum(r['detections'] for r in frame_results)
        avg_fps = sum(r['fps'] for r in frame_results) / len(frame_results)
        
        # Get final counts
        final_line_counts = self.line_counter.get_all_counts() if self.line_counter else {}
        final_zone_counts = self.zone_counter.get_all_counts() if self.zone_counter else {}
        
        summary = {
            'total_frames': len(frame_results),
            'total_detections': total_detections,
            'average_fps': round(avg_fps, 2),
            'line_counts': final_line_counts,
            'zone_counts': final_zone_counts,
            'detector_stats': self.detector.get_statistics(),
            'tracker_stats': self.tracker.get_statistics()
        }
        
        return summary
    
    def _save_reports(self, summary: Dict) -> None:
        """Save processing reports"""
        reports_dir = Path("outputs/logs")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = reports_dir / f"report_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Report saved: {json_path}")
    
    def reset(self) -> None:
        """Reset pipeline state"""
        self.detector.reset_statistics()
        self.tracker.reset()
        if self.line_counter:
            self.line_counter.reset_all()
        if self.zone_counter:
            self.zone_counter.reset_all()
        
        self.frame_count = 0
        self.processing_times.clear()
        logger.info("Pipeline reset")
