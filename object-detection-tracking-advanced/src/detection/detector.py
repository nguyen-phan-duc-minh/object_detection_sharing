"""
Object Detector Module
Implements YOLO-based object detection with optimizations
"""
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

from core.logging_setup import get_logger

logger = get_logger("detector")


class ObjectDetector:
    """YOLO-based object detector with supervision integration"""
    
    def __init__(
        self,
        weights_path: str = "weights/yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        half_precision: bool = False
    ):
        """
        Initialize object detector
        
        Args:
            weights_path: Path to YOLO weights file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cpu' or 'cuda')
            classes: List of class IDs to detect (None = all classes)
            half_precision: Use FP16 for faster inference (GPU only)
        """
        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        self.half_precision = half_precision and device != "cpu"
        
        # Load model
        self._load_model()
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        
        logger.info(f"Detector initialized: {weights_path} on {device}")
        logger.info(f"Confidence: {confidence_threshold}, IOU: {iou_threshold}")
        if classes:
            logger.info(f"Filtering classes: {classes}")
    
    def _load_model(self) -> None:
        """Load YOLO model with error handling"""
        try:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
            
            self.model = YOLO(str(self.weights_path))
            
            # Warm up model
            logger.info("Warming up model...")
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False, device=self.device)
            
            logger.info("Model loaded and warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            supervision Detections object
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                device=self.device,
                half=self.half_precision,
                verbose=False
            )[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Update statistics
            self.frame_count += 1
            self.total_detections += len(detections)
            
            logger.debug(f"Frame {self.frame_count}: {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return sv.Detections.empty()
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """
        Detect objects in multiple frames (batch processing)
        
        Args:
            frames: List of input frames
            
        Returns:
            List of Detections objects
        """
        try:
            # Run batch inference
            results = self.model(
                frames,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                device=self.device,
                half=self.half_precision,
                verbose=False,
                stream=True
            )
            
            # Convert all results
            detections_list = []
            for result in results:
                detections = sv.Detections.from_ultralytics(result)
                detections_list.append(detections)
                self.total_detections += len(detections)
            
            self.frame_count += len(frames)
            logger.debug(f"Batch processed: {len(frames)} frames")
            
            return detections_list
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [sv.Detections.empty() for _ in frames]
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get class name from class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            Class name
        """
        return self.model.names.get(class_id, "unknown")
    
    def get_statistics(self) -> dict:
        """
        Get detection statistics
        
        Returns:
            Dictionary with statistics
        """
        avg_detections = self.total_detections / self.frame_count if self.frame_count > 0 else 0
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': round(avg_detections, 2)
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics"""
        self.frame_count = 0
        self.total_detections = 0
        logger.info("Detector statistics reset")
    
    def update_settings(
        self,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> None:
        """
        Update detector settings
        
        Args:
            confidence_threshold: New confidence threshold
            iou_threshold: New IOU threshold
            classes: New class filter
        """
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            logger.info(f"Confidence threshold updated: {confidence_threshold}")
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            logger.info(f"IOU threshold updated: {iou_threshold}")
        
        if classes is not None:
            self.classes = classes
            logger.info(f"Class filter updated: {classes}")
    
    def __repr__(self) -> str:
        return (f"ObjectDetector(model={self.weights_path.name}, "
                f"device={self.device}, conf={self.confidence_threshold})")
