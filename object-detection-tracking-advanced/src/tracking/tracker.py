"""
Object Tracker Module
Implements ByteTrack/BoTSORT tracking with supervision
"""
from typing import Optional

import numpy as np
import supervision as sv

from core.logging_setup import get_logger

logger = get_logger("tracker")


class ObjectTracker:
    """Multi-object tracker using ByteTrack or BoTSORT"""
    
    def __init__(
        self,
        tracker_type: str = "bytetrack",
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1
    ):
        """
        Initialize object tracker
        
        Args:
            tracker_type: Type of tracker ('bytetrack' or 'botsort')
            track_activation_threshold: Detection confidence threshold for track activation
            lost_track_buffer: Number of frames to keep lost tracks
            minimum_matching_threshold: Minimum IOU for matching
            minimum_consecutive_frames: Minimum consecutive detections to start a track
        """
        self.tracker_type = tracker_type.lower()
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames
        
        # Initialize tracker
        self._initialize_tracker()
        
        # Statistics
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = 0
        
        logger.info(f"Tracker initialized: {tracker_type}")
        logger.info(f"Activation threshold: {track_activation_threshold}")
        logger.info(f"Lost buffer: {lost_track_buffer} frames")
    
    def _initialize_tracker(self) -> None:
        """Initialize the appropriate tracker"""
        try:
            if self.tracker_type == "bytetrack":
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_activation_threshold,
                    lost_track_buffer=self.lost_track_buffer,
                    minimum_matching_threshold=self.minimum_matching_threshold,
                    minimum_consecutive_frames=self.minimum_consecutive_frames
                )
            elif self.tracker_type == "botsort":
                # BoTSORT requires additional dependencies
                try:
                    from supervision.tracker.byte_tracker.bot_sort import BotSORT
                    self.tracker = sv.ByteTrack(  # Fallback to ByteTrack
                        track_activation_threshold=self.track_activation_threshold,
                        lost_track_buffer=self.lost_track_buffer,
                        minimum_matching_threshold=self.minimum_matching_threshold,
                        minimum_consecutive_frames=self.minimum_consecutive_frames
                    )
                    logger.warning("BoTSORT not available, using ByteTrack instead")
                except ImportError:
                    logger.warning("BoTSORT dependencies not found, falling back to ByteTrack")
                    self.tracker_type = "bytetrack"
                    self.tracker = sv.ByteTrack(
                        track_activation_threshold=self.track_activation_threshold,
                        lost_track_buffer=self.lost_track_buffer,
                        minimum_matching_threshold=self.minimum_matching_threshold,
                        minimum_consecutive_frames=self.minimum_consecutive_frames
                    )
            else:
                raise ValueError(f"Unknown tracker type: {self.tracker_type}")
            
            logger.info(f"{self.tracker_type.upper()} tracker ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            raise
    
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections
        
        Args:
            detections: Detection results from detector
            
        Returns:
            Detections with tracker IDs assigned
        """
        try:
            # Update tracker
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Update statistics
            self.frame_count += 1
            if tracked_detections.tracker_id is not None:
                unique_ids = set(tracked_detections.tracker_id)
                self.active_tracks = len(unique_ids)
                
                # Update total unique tracks
                max_id = max(tracked_detections.tracker_id) if len(tracked_detections.tracker_id) > 0 else 0
                if max_id > self.total_tracks:
                    self.total_tracks = max_id
            
            logger.debug(f"Frame {self.frame_count}: {self.active_tracks} active tracks")
            
            return tracked_detections
            
        except Exception as e:
            logger.error(f"Tracking update failed: {e}")
            return detections
    
    def reset(self) -> None:
        """Reset tracker state"""
        self._initialize_tracker()
        self.frame_count = 0
        self.active_tracks = 0
        logger.info("Tracker reset")
    
    def get_statistics(self) -> dict:
        """
        Get tracking statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'frames_processed': self.frame_count,
            'total_unique_tracks': self.total_tracks,
            'active_tracks': self.active_tracks
        }
    
    def get_track_history(self, detections: sv.Detections) -> dict:
        """
        Get track history for visualization
        
        Args:
            detections: Tracked detections
            
        Returns:
            Dictionary mapping tracker_id to positions
        """
        history = {}
        if detections.tracker_id is not None:
            for idx, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is not None:
                    x1, y1, x2, y2 = detections.xyxy[idx]
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    if tracker_id not in history:
                        history[tracker_id] = []
                    history[tracker_id].append(center)
        
        return history
    
    def __repr__(self) -> str:
        return (f"ObjectTracker(type={self.tracker_type}, "
                f"active_tracks={self.active_tracks})")
