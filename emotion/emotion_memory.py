import logging
from collections import Counter, deque
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from statistics import mean
from database.unified_db import UnifiedDatabase

logger = logging.getLogger(__name__)

class EmotionMemory:
    """
    Production-ready emotion memory with persistence, statistics, and advanced analysis.
    Stores emotion history with temporal context and provides analysis capabilities.
    """

    def __init__(
        self,
        db: UnifiedDatabase,
        max_history: int = 50,
        enable_logging: bool = True
    ):
        """
        Initialize emotion memory with unified database.
        
        Args:
            db: UnifiedDatabase instance
            max_history: Maximum number of emotion entries to keep per track
            enable_logging: Whether to enable detailed logging
        """
        self.db = db
        self.max_history = max_history
        self.enable_logging = enable_logging
        
        # In-memory caches for fast access
        self.emotion_history: Dict[int, deque] = {}
        self.emotion_timestamps: Dict[int, deque] = {}
        self.emotion_confidences: Dict[int, deque] = {}
        self.track_metadata: Dict[int, Dict] = {}
        
        # Load from database
        self._load_from_database()
        
        if self.enable_logging:
            logger.info(f"EmotionMemory initialized with max_history={max_history}")

    def emotion_memory(self, session_id: int, person_id: int, emotion: str, confidence: float = 1.0, timestamp: Optional[float] = None):
        """
        Emotion history for a person.
        
        Args:
            session_id: Session ID
            person_id: Unique identifier for the tracked person
            emotion: Detected emotion label
            confidence: Confidence score for the detection
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        # Save to unified database
        self.db.add_emotion(session_id, person_id, emotion, confidence, timestamp)
        
        # Update in-memory cache
        if person_id not in self.emotion_history:
            self.emotion_history[person_id] = deque(maxlen=self.max_history)
            self.emotion_timestamps[person_id] = deque(maxlen=self.max_history)
            self.emotion_confidences[person_id] = deque(maxlen=self.max_history)
            self.track_metadata[person_id] = {
                "first_seen": timestamp,
                "last_seen": timestamp,
                "total_detections": 0
            }
        
        # Update history
        self.emotion_history[person_id].append(emotion)
        self.emotion_timestamps[person_id].append(timestamp)
        self.emotion_confidences[person_id].append(confidence)
        
        # Update metadata
        self.track_metadata[person_id]["last_seen"] = timestamp
        self.track_metadata[person_id]["total_detections"] += 1
        
        if self.enable_logging:
            logger.debug(f"Updated emotion memory for person {person_id}: {emotion} (confidence: {confidence:.3f})")

    def _load_from_database(self):
        """Load emotion data from unified database into memory."""
        try:
            # Get all persons from database
            persons = self.db.get_all_persons()
            
            for person in persons:
                person_id = person['person_id']
                self.track_metadata[person_id] = {
                    'first_seen': person['first_seen'],
                    'last_seen': person['last_seen'],
                    'total_detections': person['total_detections']
                }
                self.emotion_history[person_id] = deque(maxlen=self.max_history)
                self.emotion_timestamps[person_id] = deque(maxlen=self.max_history)
                self.emotion_confidences[person_id] = deque(maxlen=self.max_history)
            
            # Load emotion entries for each person
            for person_id in self.track_metadata.keys():
                emotions = self.db.get_emotions(person_id, limit=self.max_history)
                for emo in reversed(emotions):  # Reverse to maintain chronological order
                    self.emotion_history[person_id].append(emo['emotion'])
                    self.emotion_timestamps[person_id].append(emo['timestamp'])
                    self.emotion_confidences[person_id].append(emo['confidence'])
            
            if self.enable_logging:
                logger.info(f"Loaded {len(self.track_metadata)} persons from database")
                
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to load from database: {e}")

    def get(self, person_id: int) -> Optional[str]:
        """
        Get most frequent emotion for a person using majority voting.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Most common emotion or None if no history
        """
        if person_id not in self.emotion_history:
            return None
        
        history = list(self.emotion_history[person_id])
        
        if not history:
            return None
        
        counter = Counter(history)
        return counter.most_common(1)[0][0]

    def get_history(self, person_id: int) -> Optional[List[Tuple[str, float, float]]]:
        """
        Get full emotion history for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            List of (emotion, confidence, timestamp) tuples or None
        """
        if person_id not in self.emotion_history:
            return None
        
        return list(zip(
            self.emotion_history[person_id],
            self.emotion_confidences[person_id],
            self.emotion_timestamps[person_id]
        ))

    def get_emotion_distribution(self, person_id: int) -> Optional[Dict[str, int]]:
        """
        Get emotion distribution for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dictionary mapping emotions to counts or None
        """
        if person_id not in self.emotion_history:
            return None
        
        history = list(self.emotion_history[person_id])
        return dict(Counter(history))

    def get_emotion_percentage(self, person_id: int) -> Optional[Dict[str, float]]:
        """
        Get emotion distribution as percentages for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dictionary mapping emotions to percentages or None
        """
        distribution = self.get_emotion_distribution(person_id)
        if not distribution:
            return None
        
        total = sum(distribution.values())
        return {emotion: (count / total) * 100 for emotion, count in distribution.items()}

    def get_average_confidence(self, person_id: int) -> Optional[float]:
        """
        Get average confidence for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Average confidence or None
        """
        if person_id not in self.emotion_confidences:
            return None
        
        confidences = list(self.emotion_confidences[person_id])
        return mean(confidences) if confidences else None

    def get_emotion_duration(self, person_id: int, emotion: str) -> Optional[float]:
        """
        Get total duration of a specific emotion for a person.
        
        Args:
            person_id: Person identifier
            emotion: Emotion label to analyze
            
        Returns:
            Total duration in seconds or None
        """
        if person_id not in self.emotion_history:
            return None
        
        history = list(self.emotion_history[person_id])
        timestamps = list(self.emotion_timestamps[person_id])
        
        total_duration = 0.0
        for i, (emo, ts) in enumerate(zip(history, timestamps)):
            if emo == emotion and i < len(timestamps) - 1:
                total_duration += timestamps[i + 1] - ts
        
        return total_duration if total_duration > 0 else None

    def get_track_statistics(self, person_id: int) -> Optional[Dict]:
        """
        Get comprehensive statistics for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Dictionary with person statistics or None
        """
        if person_id not in self.track_metadata:
            return None
        
        history = self.get_history(person_id)
        if not history:
            return None
        
        metadata = self.track_metadata[person_id]
        duration = metadata["last_seen"] - metadata["first_seen"]
        
        return {
            "person_id": person_id,
            "total_detections": metadata["total_detections"],
            "duration_seconds": duration,
            "most_common_emotion": self.get(person_id),
            "emotion_distribution": self.get_emotion_distribution(person_id),
            "emotion_percentages": self.get_emotion_percentage(person_id),
            "average_confidence": self.get_average_confidence(person_id),
            "first_seen": datetime.fromtimestamp(metadata["first_seen"]).isoformat(),
            "last_seen": datetime.fromtimestamp(metadata["last_seen"]).isoformat()
        }

    def get_global_statistics(self) -> Dict:
        """
        Get global statistics across all persons.
        
        Returns:
            Dictionary with global statistics
        """
        total_detections = sum(
            metadata["total_detections"] for metadata in self.track_metadata.values()
        )
        
        all_emotions = []
        for history in self.emotion_history.values():
            all_emotions.extend(list(history))
        
        global_distribution = dict(Counter(all_emotions)) if all_emotions else {}
        
        return {
            "total_persons": len(self.track_metadata),
            "total_detections": total_detections,
            "global_emotion_distribution": global_distribution,
            "average_detections_per_person": total_detections / len(self.track_metadata) if self.track_metadata else 0
        }

    def clear(self, person_id: int):
        """
        Clear emotion history for a specific person.
        
        Args:
            person_id: Person identifier
        """
        if person_id in self.emotion_history:
            del self.emotion_history[person_id]
        if person_id in self.emotion_timestamps:
            del self.emotion_timestamps[person_id]
        if person_id in self.emotion_confidences:
            del self.emotion_confidences[person_id]
        if person_id in self.track_metadata:
            del self.track_metadata[person_id]
        
        if self.enable_logging:
            logger.debug(f"Cleared emotion memory for person {person_id}")
        
        self.db.clear_person_data(person_id)

    def clear_all(self):
        """Clear all emotion history."""
        self.emotion_history.clear()
        self.emotion_timestamps.clear()
        self.emotion_confidences.clear()
        self.track_metadata.clear()
        
        if self.enable_logging:
            logger.info("Cleared all emotion memory")
        
        self.db.clear_all_data()

    def save_to_disk(self):
        """Save to unified database (auto-saved on each update)."""
        # Data is automatically saved to database on each update
        # This method is kept for compatibility
        if self.enable_logging:
            logger.info("Emotion data auto-saved to unified database")

    def load_from_disk(self):
        """Load from unified database (auto-loaded on initialization)."""
        # Data is automatically loaded from database on initialization
        # This method is kept for compatibility
        if self.enable_logging:
            logger.info("Emotion data auto-loaded from unified database")

    def export_report(self, person_id: int) -> Optional[str]:
        """
        Generate a human-readable report for a person.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Formatted report string or None
        """
        stats = self.get_track_statistics(person_id)
        if not stats:
            return None
        
        report = f"Emotion Report for Person {person_id}\n"
        report += f"{'='*40}\n"
        report += f"Total Detections: {stats['total_detections']}\n"
        report += f"Duration: {stats['duration_seconds']:.1f} seconds\n"
        report += f"Most Common Emotion: {stats['most_common_emotion']}\n\n"
        report += "Emotion Distribution:\n"
        
        for emotion, percentage in stats['emotion_percentages'].items():
            report += f"  {emotion}: {percentage:.1f}%\n"
        
        report += f"\nAverage Confidence: {stats['average_confidence']:.3f}\n"
        report += f"First Seen: {stats['first_seen']}\n"
        report += f"Last Seen: {stats['last_seen']}\n"
        
        return report
    