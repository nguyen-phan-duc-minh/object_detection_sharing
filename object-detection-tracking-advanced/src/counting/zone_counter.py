"""
Zone Counter Module
Implements zone-based object counting (entering/exiting polygonal zones)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

from core.logging_setup import get_logger
from src.counting.direction import Direction

logger = get_logger("counter.zone")


class ZoneCounter:
    """Count objects entering/exiting a polygonal zone"""
    
    def __init__(
        self,
        name: str,
        polygon: List[Tuple[int, int]],
        count_in: bool = True,
        count_out: bool = True
    ):
        """
        Initialize zone counter
        
        Args:
            name: Zone identifier
            polygon: List of polygon vertices [(x1, y1), (x2, y2), ...]
            count_in: Count objects entering zone
            count_out: Count objects exiting zone
        """
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.count_in = count_in
        self.count_out = count_out
        
        # Initialize supervision PolygonZone
        self.polygon_zone = sv.PolygonZone(
            polygon=self.polygon,
            frame_resolution_wh=(1920, 1080)  # Will be updated on first frame
        )
        
        # Tracking states
        self.counts_in = 0
        self.counts_out = 0
        self.current_count = 0
        self.tracked_objects: Dict[int, bool] = {}  # tracker_id: is_inside
        
        logger.info(f"Zone counter '{name}' initialized with {len(polygon)} vertices")
    
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
        
        # Trigger zone to get mask of objects inside
        mask = self.polygon_zone.trigger(detections)
        self.current_count = int(np.sum(mask))
        
        # Track individual objects for in/out counting
        if detections.tracker_id is not None:
            current_ids = set()
            
            for idx, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is None:
                    continue
                
                current_ids.add(tracker_id)
                is_inside = bool(mask[idx])
                
                # Check if this is a new object
                if tracker_id not in self.tracked_objects:
                    self.tracked_objects[tracker_id] = is_inside
                    # Don't count initial state
                else:
                    # Check for state change
                    was_inside = self.tracked_objects[tracker_id]
                    
                    if not was_inside and is_inside:
                        # Object entered zone
                        if self.count_in:
                            self.counts_in += 1
                            logger.debug(f"Object {tracker_id} entered zone '{self.name}'")
                    
                    elif was_inside and not is_inside:
                        # Object exited zone
                        if self.count_out:
                            self.counts_out += 1
                            logger.debug(f"Object {tracker_id} exited zone '{self.name}'")
                    
                    self.tracked_objects[tracker_id] = is_inside
            
            # Clean up old tracked objects
            old_ids = set(self.tracked_objects.keys()) - current_ids
            for old_id in old_ids:
                if old_id in self.tracked_objects:
                    del self.tracked_objects[old_id]
        
        return self.get_counts()
    
    def get_counts(self) -> Dict[str, int]:
        """
        Get current counts
        
        Returns:
            Dictionary with in/out/current counts
        """
        return {
            'in': self.counts_in,
            'out': self.counts_out,
            'current': self.current_count,
            'total': self.counts_in
        }
    
    def reset(self) -> None:
        """Reset counter"""
        self.counts_in = 0
        self.counts_out = 0
        self.current_count = 0
        self.tracked_objects.clear()
        logger.info(f"Zone counter '{self.name}' reset")
    
    def get_polygon(self) -> np.ndarray:
        """
        Get zone polygon
        
        Returns:
            Polygon vertices as numpy array
        """
        return self.polygon
    
    def __repr__(self) -> str:
        return (f"ZoneCounter(name='{self.name}', "
                f"in={self.counts_in}, out={self.counts_out}, current={self.current_count})")


class MultiZoneCounter:
    """Manage multiple zone counters"""
    
    def __init__(self, zone_configs: Optional[List[Dict]] = None):
        """
        Initialize multi-zone counter
        
        Args:
            zone_configs: List of zone configuration dictionaries
        """
        self.counters: Dict[str, ZoneCounter] = {}
        
        if zone_configs:
            for config in zone_configs:
                self.add_zone(
                    name=config['name'],
                    polygon=config['polygon'],
                    count_in=config.get('count_in', True),
                    count_out=config.get('count_out', True)
                )
        
        logger.info(f"Multi-zone counter initialized with {len(self.counters)} zones")
    
    def add_zone(
        self,
        name: str,
        polygon: List[Tuple[int, int]],
        count_in: bool = True,
        count_out: bool = True
    ) -> None:
        """
        Add a new zone counter
        
        Args:
            name: Zone identifier
            polygon: Zone polygon vertices
            count_in: Count objects entering
            count_out: Count objects exiting
        """
        self.counters[name] = ZoneCounter(name, polygon, count_in, count_out)
    
    def update(self, detections: sv.Detections) -> Dict[str, Dict[str, int]]:
        """
        Update all zone counters
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary mapping zone names to counts
        """
        results = {}
        for name, counter in self.counters.items():
            results[name] = counter.update(detections)
        return results
    
    def get_all_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get counts from all zones
        
        Returns:
            Dictionary mapping zone names to counts
        """
        return {name: counter.get_counts() for name, counter in self.counters.items()}
    
    def reset_all(self) -> None:
        """Reset all counters"""
        for counter in self.counters.values():
            counter.reset()
        logger.info("All zone counters reset")
    
    def __len__(self) -> int:
        return len(self.counters)
    
    def __getitem__(self, name: str) -> ZoneCounter:
        return self.counters[name]
    
    def __iter__(self):
        return iter(self.counters.values())
