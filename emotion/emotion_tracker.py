import time
import logging
from typing import Dict, Optional, List, Tuple
from collections import deque
from statistics import mean

logger = logging.getLogger(__name__)

class EmotionTracker:
    """
    Production-ready emotion tracker with temporal smoothing and advanced tracking features.
    Tracks emotion changes over time with confidence-based smoothing.
    """

    def __init__(
        self,
        history_size: int = 10,
        smoothing_window: int = 3,
        min_confidence: float = 0.3,
        track_timeout: float = 5.0,
        enable_logging: bool = True
    ):
        """
        Initialize emotion tracker.
        
        Args:
            history_size: Number of recent emotion detections to keep per track
            smoothing_window: Window size for temporal smoothing
            min_confidence: Minimum confidence to accept emotion detection
            track_timeout: Seconds before a track is considered stale
            enable_logging: Whether to enable detailed logging
        """
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        self.min_confidence = min_confidence
        self.track_timeout = track_timeout
        self.enable_logging = enable_logging
        
        self.tracks: Dict[int, Dict] = {}
        self.emotion_history: Dict[int, deque] = {}
        self.confidence_history: Dict[int, deque] = {}
        self.last_update: Dict[int, float] = {}
        
        if self.enable_logging:
            logger.info(f"EmotionTracker initialized with history_size={history_size}, smoothing_window={smoothing_window}")

    def emoton_tracker(self, person_id: int, emotion: str, confidence: float) -> Optional[Dict]:
        """
        Emotion track with new detection.
        
        Args:
            person_id: Unique identifier for the tracked person
            emotion: Detected emotion label
            confidence: Confidence score for the detection
            
        Returns:
            Updated track data or None if confidence is too low
        """
        if confidence < self.min_confidence:
            if self.enable_logging:
                logger.debug(f"Track {person_id}: confidence {confidence:.3f} below threshold {self.min_confidence}")
            return None
        
        current_time = time.time()
        
        # Initialize track if new
        if person_id not in self.tracks:
            self.tracks[person_id] = {
                "emotion": emotion,
                "confidence": confidence,
                "updated_at": current_time,
                "detection_count": 1
            }
            self.emotion_history[person_id] = deque(maxlen=self.history_size)
            self.confidence_history[person_id] = deque(maxlen=self.history_size)
            self.last_update[person_id] = current_time
        else:
            # Check if track is stale
            if current_time - self.last_update[person_id] > self.track_timeout:
                if self.enable_logging:
                    logger.info(f"Track {person_id} was stale, resetting history")
                self.emotion_history[person_id].clear()
                self.confidence_history[person_id].clear()
            
            # Update track
            self.tracks[person_id]["emotion"] = emotion
            self.tracks[person_id]["confidence"] = confidence
            self.tracks[person_id]["updated_at"] = current_time
            self.tracks[person_id]["detection_count"] += 1
            self.last_update[person_id] = current_time
        
        # Update history
        self.emotion_history[person_id].append(emotion)
        self.confidence_history[person_id].append(confidence)
        
        # Apply temporal smoothing
        smoothed_emotion = self._get_smoothed_emotion(person_id)
        if smoothed_emotion:
            self.tracks[person_id]["smoothed_emotion"] = smoothed_emotion
        
        if self.enable_logging:
            logger.debug(f"Track {person_id}: emotion={emotion}, confidence={confidence:.3f}, smoothed={smoothed_emotion}")
        
        return self.tracks[person_id]

    def _get_smoothed_emotion(self, person_id: int) -> Optional[str]:
        """
        Get temporally smoothed emotion using majority voting over recent history.
        
        Args:
            person_id: Track identifier
            
        Returns:
            Smoothed emotion or None if insufficient history
        """
        if person_id not in self.emotion_history:
            return None
        
        history = list(self.emotion_history[person_id])
        if len(history) < self.smoothing_window:
            return history[-1] if history else None
        
        # Use only recent history for smoothing
        recent_history = history[-self.smoothing_window:]
        
        # Majority voting
        from collections import Counter
        emotion_counts = Counter(recent_history)
        most_common = emotion_counts.most_common(1)
        
        return most_common[0][0] if most_common else None

    def get(self, person_id: int) -> Optional[Dict]:
        """
        Get current track data.
        
        Args:
            person_id: Track identifier
            
        Returns:
            Track data dictionary or None if track doesn't exist
        """
        return self.tracks.get(person_id)

    def get_smoothed(self, person_id: int) -> Optional[str]:
        """
        Get smoothed emotion for a track.
        
        Args:
            person_id: Track identifier
            
        Returns:
            Smoothed emotion or None
        """
        track = self.get(person_id)
        return track.get("smoothed_emotion") if track else None

    def get_confidence_trend(self, person_id: int) -> Optional[List[float]]:
        """
        Get confidence trend for a track.
        
        Args:
            person_id: Track identifier
            
        Returns:
            List of recent confidence values or None
        """
        if person_id not in self.confidence_history:
            return None
        return list(self.confidence_history[person_id])

    def get_average_confidence(self, person_id: int) -> Optional[float]:
        """
        Get average confidence for a track.
        
        Args:
            person_id: Track identifier
            
        Returns:
            Average confidence or None
        """
        trend = self.get_confidence_trend(person_id)
        return mean(trend) if trend else None

    def remove(self, person_id: int):
        """
        Remove a track.
        
        Args:
            person_id: Track identifier
        """
        if person_id in self.tracks:
            del self.tracks[person_id]
        if person_id in self.emotion_history:
            del self.emotion_history[person_id]
        if person_id in self.confidence_history:
            del self.confidence_history[person_id]
        if person_id in self.last_update:
            del self.last_update[person_id]
        if self.enable_logging:
            logger.debug(f"Removed track {person_id}")

    def clear(self):
        """Clear all tracks."""
        self.tracks.clear()
        self.emotion_history.clear()
        self.confidence_history.clear()
        self.last_update.clear()
        if self.enable_logging:
            logger.info("Cleared all tracks")
    
    def get_all(self) -> Dict:
        """Get all tracks."""
        return self.tracks

    def cleanup_stale_tracks(self) -> int:
        """
        Remove stale tracks that haven't been updated recently.
        
        Returns:
            Number of tracks removed
        """
        current_time = time.time()
        stale_tracks = [
            person_id for person_id, last_update in self.last_update.items()
            if current_time - last_update > self.track_timeout
        ]
        
        for person_id in stale_tracks:
            self.remove(person_id)
        
        if stale_tracks and self.enable_logging:
            logger.info(f"Cleaned up {len(stale_tracks)} stale tracks")
        
        return len(stale_tracks)

    def get_statistics(self) -> Dict:
        """Get tracker statistics."""
        return {
            "total_tracks": len(self.tracks),
            "history_size": self.history_size,
            "smoothing_window": self.smoothing_window,
            "min_confidence": self.min_confidence,
            "track_timeout": self.track_timeout
        }
