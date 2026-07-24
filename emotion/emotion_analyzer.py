import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timedelta
from database.unified_db import UnifiedDatabase

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Analyze emotion data to compute statistics and trends."""
    
    def __init__(self, db: UnifiedDatabase, enable_logging: bool = True):
        """
        Initialize emotion analyzer.
        
        Args:
            db: UnifiedDatabase instance
            enable_logging: Whether to enable logging
        """
        self.db = db
        self.enable_logging = enable_logging
    
    def analyze_last_minutes(self, session_id: int, minutes: int = 2) -> Dict:
        """
        Analyze emotions from the last N minutes.
        
        Args:
            session_id: Session ID to analyze
            minutes: Number of minutes to look back (default: 2)
            
        Returns:
            Dictionary containing:
                - dominant_emotion: Most common emotion
                - dominance_percentage: Percentage of dominant emotion
                - average_confidence: Average confidence across all detections
                - emotion_distribution: Count of each emotion
                - trends: Emotion trend analysis
                - total_detections: Total number of emotion detections
        """
        try:
            time_window_seconds = minutes * 60
            emotions = self.db.get_emotions_by_session(session_id, time_window_seconds)
            
            if not emotions:
                return {
                    "dominant_emotion": None,
                    "dominance_percentage": 0.0,
                    "average_confidence": 0.0,
                    "emotion_distribution": {},
                    "trends": {"direction": "stable", "change": 0.0},
                    "total_detections": 0
                }
            
            # Calculate emotion distribution
            emotion_counts = Counter([e['emotion'] for e in emotions])
            total_detections = len(emotions)
            
            # Calculate dominant emotion and percentage
            dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]
            dominance_percentage = (dominant_count / total_detections) * 100
            
            # Calculate average confidence
            average_confidence = sum([e['confidence'] for e in emotions]) / total_detections
            
            # Analyze trends
            trends = self._analyze_trends(emotions)
            
            result = {
                "dominant_emotion": dominant_emotion,
                "dominance_percentage": dominance_percentage,
                "average_confidence": average_confidence,
                "emotion_distribution": dict(emotion_counts),
                "trends": trends,
                "total_detections": total_detections
            }
            
            if self.enable_logging:
                logger.info(f"Emotion analysis completed: {result}")
            
            return result
            
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to analyze emotions: {e}")
            return {
                "dominant_emotion": None,
                "dominance_percentage": 0.0,
                "average_confidence": 0.0,
                "emotion_distribution": {},
                "trends": {"direction": "error", "change": 0.0},
                "total_detections": 0
            }
    
    def _analyze_trends(self, emotions: List[Dict]) -> Dict:
        """
        Analyze emotion trends over time.
        
        Args:
            emotions: List of emotion entries with timestamps
            
        Returns:
            Dictionary with trend direction and change percentage
        """
        if len(emotions) < 2:
            return {"direction": "insufficient_data", "change": 0.0}
        
        # Split emotions into first half and second half
        mid_point = len(emotions) // 2
        first_half = emotions[:mid_point]
        second_half = emotions[mid_point:]
        
        # Count emotions in each half
        first_half_counts = Counter([e['emotion'] for e in first_half])
        second_half_counts = Counter([e['emotion'] for e in second_half])
        
        # Find dominant emotion in each half
        first_dominant = first_half_counts.most_common(1)[0] if first_half_counts else (None, 0)
        second_dominant = second_half_counts.most_common(1)[0] if second_half_counts else (None, 0)
        
        # Calculate change
        if first_dominant[0] == second_dominant[0]:
            # Same dominant emotion
            if first_half_counts and second_half_counts:
                first_percentage = (first_dominant[1] / len(first_half)) * 100
                second_percentage = (second_dominant[1] / len(second_half)) * 100
                change = second_percentage - first_percentage
                
                if change > 5:
                    direction = "increasing"
                elif change < -5:
                    direction = "decreasing"
                else:
                    direction = "stable"
            else:
                direction = "stable"
                change = 0.0
        else:
            # Different dominant emotion
            direction = "changing"
            change = 0.0
        
        return {
            "direction": direction,
            "change": change,
            "first_half_dominant": first_dominant[0],
            "second_half_dominant": second_dominant[0]
        }
    
    def analyze_by_person(self, session_id: int, person_id: int, minutes: int = 2) -> Dict:
        """
        Analyze emotions for a specific person in the last N minutes.
        
        Args:
            session_id: Session ID
            person_id: Person ID to analyze
            minutes: Number of minutes to look back
            
        Returns:
            Dictionary with emotion analysis for the specific person
        """
        try:
            time_window_seconds = minutes * 60
            all_emotions = self.db.get_emotions_by_session(session_id, time_window_seconds)
            
            # Filter emotions for the specific person
            person_emotions = [e for e in all_emotions if e['person_id'] == person_id]
            
            if not person_emotions:
                return {
                    "dominant_emotion": None,
                    "dominance_percentage": 0.0,
                    "average_confidence": 0.0,
                    "emotion_distribution": {},
                    "trends": {"direction": "no_data", "change": 0.0},
                    "total_detections": 0
                }
            
            # Calculate emotion distribution
            emotion_counts = Counter([e['emotion'] for e in person_emotions])
            total_detections = len(person_emotions)
            
            # Calculate dominant emotion and percentage
            dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]
            dominance_percentage = (dominant_count / total_detections) * 100
            
            # Calculate average confidence
            average_confidence = sum([e['confidence'] for e in person_emotions]) / total_detections
            
            # Analyze trends
            trends = self._analyze_trends(person_emotions)
            
            return {
                "dominant_emotion": dominant_emotion,
                "dominance_percentage": dominance_percentage,
                "average_confidence": average_confidence,
                "emotion_distribution": dict(emotion_counts),
                "trends": trends,
                "total_detections": total_detections
            }
            
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to analyze emotions for person {person_id}: {e}")
            return {
                "dominant_emotion": None,
                "dominance_percentage": 0.0,
                "average_confidence": 0.0,
                "emotion_distribution": {},
                "trends": {"direction": "error", "change": 0.0},
                "total_detections": 0
            }
