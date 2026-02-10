"""
Visualization Module
Draw detections, tracks, and counting information on frames
"""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from core.logging_setup import get_logger

logger = get_logger("visualization")


class Visualizer:
    """Visualize detections, tracking, and counting results"""
    
    def __init__(
        self,
        thickness: int = 2,
        text_scale: float = 0.6,
        text_thickness: int = 2,
        colors: Optional[Dict[str, List[int]]] = None
    ):
        """
        Initialize visualizer
        
        Args:
            thickness: Line thickness for boxes
            text_scale: Text scale factor
            text_thickness: Text thickness
            colors: Dictionary mapping object types to BGR colors
        """
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        
        # Default colors (BGR format)
        self.colors = colors or {
            'person': [0, 255, 0],      # Green
            'car': [255, 0, 0],         # Blue
            'line': [0, 255, 255],      # Yellow
            'zone': [255, 0, 255],      # Magenta
            'default': [0, 255, 0]      # Green
        }
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=thickness,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_position=sv.Position.TOP_LEFT
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=30,
            position=sv.Position.CENTER
        )
        
        logger.info("Visualizer initialized")
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        class_names: Optional[Dict[int, str]] = None,
        show_confidence: bool = True,
        show_trace: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: Detection results
            class_names: Mapping of class IDs to names
            show_confidence: Show confidence scores
            show_trace: Show tracking traces
            
        Returns:
            Annotated frame
        """
        if detections is None or len(detections) == 0:
            return frame
        
        # Draw bounding boxes
        frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        
        # Prepare labels
        labels = []
        for idx in range(len(detections)):
            class_id = detections.class_id[idx] if detections.class_id is not None else None
            confidence = detections.confidence[idx] if detections.confidence is not None else None
            tracker_id = detections.tracker_id[idx] if detections.tracker_id is not None else None
            
            # Build label
            label_parts = []
            
            if class_names and class_id is not None:
                label_parts.append(class_names.get(class_id, f"Class {class_id}"))
            
            if tracker_id is not None:
                label_parts.append(f"ID:{tracker_id}")
            
            if show_confidence and confidence is not None:
                label_parts.append(f"{confidence:.2f}")
            
            labels.append(" ".join(label_parts))
        
        # Draw labels
        frame = self.label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        
        # Draw traces
        if show_trace and detections.tracker_id is not None:
            frame = self.trace_annotator.annotate(
                scene=frame,
                detections=detections
            )
        
        return frame
    
    def draw_line(
        self,
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        label: Optional[str] = None,
        color: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Draw a counting line
        
        Args:
            frame: Input frame
            start: Line start point (x, y)
            end: Line end point (x, y)
            label: Line label
            color: Line color (BGR)
            
        Returns:
            Annotated frame
        """
        color = color or self.colors.get('line', [0, 255, 255])
        
        # Draw line
        cv2.line(frame, start, end, color, self.thickness)
        
        # Draw endpoints
        cv2.circle(frame, start, 5, color, -1)
        cv2.circle(frame, end, 5, color, -1)
        
        # Draw label
        if label:
            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            self._draw_text(frame, label, mid_point, color)
        
        return frame
    
    def draw_zone(
        self,
        frame: np.ndarray,
        polygon: np.ndarray,
        label: Optional[str] = None,
        color: Optional[List[int]] = None,
        fill: bool = False,
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Draw a counting zone
        
        Args:
            frame: Input frame
            polygon: Zone polygon vertices
            label: Zone label
            color: Zone color (BGR)
            fill: Fill the polygon
            alpha: Fill transparency
            
        Returns:
            Annotated frame
        """
        color = color or self.colors.get('zone', [255, 0, 255])
        
        if fill:
            # Create overlay for transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw polygon outline
        cv2.polylines(frame, [polygon], True, color, self.thickness)
        
        # Draw label
        if label:
            centroid = polygon.mean(axis=0).astype(int)
            self._draw_text(frame, label, tuple(centroid), color)
        
        return frame
    
    def draw_count_info(
        self,
        frame: np.ndarray,
        counts: Dict[str, Dict[str, int]],
        position: str = "top_left"
    ) -> np.ndarray:
        """
        Draw counting information on frame
        
        Args:
            frame: Input frame
            counts: Dictionary with counting results
            position: Position on frame ('top_left', 'top_right', etc.)
            
        Returns:
            Annotated frame
        """
        if not counts:
            return frame
        
        # Calculate position
        h, w = frame.shape[:2]
        margin = 10
        line_height = 30
        
        if position == "top_left":
            x, y = margin, margin + 30
        elif position == "top_right":
            x, y = w - 300, margin + 30
        elif position == "bottom_left":
            x, y = margin, h - (len(counts) * line_height) - margin
        else:  # bottom_right
            x, y = w - 300, h - (len(counts) * line_height) - margin
        
        # Draw semi-transparent background
        overlay = frame.copy()
        box_h = len(counts) * line_height + 40
        cv2.rectangle(overlay, (x - 5, y - 25), (x + 290, y + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw count information
        for name, count_data in counts.items():
            text = f"{name}:"
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                       self.text_scale, (255, 255, 255), self.text_thickness)
            y += line_height
            
            for key, value in count_data.items():
                text = f"  {key}: {value}"
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                           self.text_scale * 0.8, (200, 200, 200), self.text_thickness - 1)
                y += line_height - 5
        
        return frame
    
    def draw_fps(
        self,
        frame: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw FPS counter on frame
        
        Args:
            frame: Input frame
            fps: Current FPS
            position: Text position
            
        Returns:
            Annotated frame
        """
        text = f"FPS: {fps:.1f}"
        
        # Draw background
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_h - 5),
                     (position[0] + text_w + 5, position[1] + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   self.text_scale, (0, 255, 0), self.text_thickness)
        
        return frame
    
    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: List[int]
    ) -> None:
        """
        Draw text with background
        
        Args:
            frame: Input frame
            text: Text to draw
            position: Text position
            color: Text color
        """
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        
        # Draw background
        cv2.rectangle(frame,
                     (position[0] - 5, position[1] - text_h - 5),
                     (position[0] + text_w + 5, position[1] + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   self.text_scale, color, self.text_thickness)
    
    def __repr__(self) -> str:
        return f"Visualizer(thickness={self.thickness})"
