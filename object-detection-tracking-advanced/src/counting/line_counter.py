"""
Line Counter Module
Implements line-based object counting with direction detection
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
from supervision.geometry.core import Point

from core.logging_setup import get_logger
from src.counting.direction import Direction

logger = get_logger("counter.line")


class LineCounter:
    """Count objects crossing a line with direction detection"""
    
    def __init__(
        self,
        name: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        count_in: bool = True,
        count_out: bool = True
    ):
        """
        Initialize line counter
        
        Args:
            name: Line identifier
            start: Line start point (x, y)
            end: Line end point (x, y)
            count_in: Count objects crossing in
            count_out: Count objects crossing out
        """
        self.name = name
        self.start = Point(start[0], start[1])
        self.end = Point(end[0], end[1])
        self.count_in = count_in
        self.count_out = count_out
        
        # Initialize supervision LineZone
        self.line_zone = sv.LineZone(
            start=self.start,
            end=self.end
        )
        
        # Tracking states
        self.counts_in = 0
        self.counts_out = 0
        self.tracked_objects: Dict[int, Dict] = {}
        
        logger.info(f"Line counter '{name}' initialized: {start} -> {end}")
    
    def update(self, detections: sv.Detections) -> Dict[str, int]:
        """
        Update counter with new detections
        
        Args:
            detections: Detection results with tracking IDs
            
        Returns:
            Dictionary with current counts
        """
        if detections is None or len(detections) == 0:
            return self.get_counts()
        
        # Trigger line zone counting
        crossed_in, crossed_out = self.line_zone.trigger(detections)
        
        # Update counts
        if self.count_in:
            self.counts_in += len([x for x in crossed_in if x])
        
        if self.count_out:
            self.counts_out += len([x for x in crossed_out if x])
        
        # Track individual objects
        if detections.tracker_id is not None:
            for idx, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is None:
                    continue
                
                # Get center point
                x1, y1, x2, y2 = detections.xyxy[idx]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Update tracked object
                if tracker_id not in self.tracked_objects:
                    self.tracked_objects[tracker_id] = {
                        'positions': [center],
                        'crossed': False
                    }
                else:
                    self.tracked_objects[tracker_id]['positions'].append(center)
                    # Keep only last 10 positions
                    if len(self.tracked_objects[tracker_id]['positions']) > 10:
                        self.tracked_objects[tracker_id]['positions'].pop(0)
        
        return self.get_counts()
    
    def get_counts(self) -> Dict[str, int]:
        """
        Get current counts
        
        Returns:
            Dictionary with in/out counts
        """
        return {
            'in': self.counts_in,
            'out': self.counts_out,
            'total': self.counts_in + self.counts_out
        }
    
    def reset(self) -> None:
        """Reset counter"""
        self.counts_in = 0
        self.counts_out = 0
        self.tracked_objects.clear()
        logger.info(f"Line counter '{self.name}' reset")
    
    def get_line_coordinates(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get line coordinates
        
        Returns:
            Tuple of (start, end) points
        """
        return ((int(self.start.x), int(self.start.y)), 
                (int(self.end.x), int(self.end.y)))
    
    def __repr__(self) -> str:
        return (f"LineCounter(name='{self.name}', "
                f"in={self.counts_in}, out={self.counts_out})")


class MultiLineCounter:
    """Manage multiple line counters"""
    
    def __init__(self, line_configs: Optional[List[Dict]] = None):
        """
        Initialize multi-line counter
        
        Args:
            line_configs: List of line configuration dictionaries
        """
        self.counters: Dict[str, LineCounter] = {}
        
        if line_configs:
            for config in line_configs:
                self.add_line(
                    name=config['name'],
                    start=config['start'],
                    end=config['end'],
                    count_in=config.get('count_in', True),
                    count_out=config.get('count_out', True)
                )
        
        logger.info(f"Multi-line counter initialized with {len(self.counters)} lines")
    
    def add_line(
        self,
        name: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        count_in: bool = True,
        count_out: bool = True
    ) -> None:
        """
        Add a new line counter
        
        Args:
            name: Line identifier
            start: Line start point
            end: Line end point
            count_in: Count objects crossing in
            count_out: Count objects crossing out
        """
        self.counters[name] = LineCounter(name, start, end, count_in, count_out)
    
    def update(self, detections: sv.Detections) -> Dict[str, Dict[str, int]]:
        """
        Update all line counters
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary mapping line names to counts
        """
        results = {}
        for name, counter in self.counters.items():
            results[name] = counter.update(detections)
        return results
    
    def get_all_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get counts from all lines
        
        Returns:
            Dictionary mapping line names to counts
        """
        return {name: counter.get_counts() for name, counter in self.counters.items()}
    
    def reset_all(self) -> None:
        """Reset all counters"""
        for counter in self.counters.values():
            counter.reset()
        logger.info("All line counters reset")
    
    def __len__(self) -> int:
        return len(self.counters)
    
    def __getitem__(self, name: str) -> LineCounter:
        return self.counters[name]
    
    def __iter__(self):
        return iter(self.counters.values())
