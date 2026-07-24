import cv2
import torch
import logging
import time
from typing import Dict, Optional, List, Tuple

# Monkey-patch torch.load to use weights_only=False for hsemotion compatibility
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, map_location=map_location, *args, **kwargs)
torch.load = _patched_torch_load

from hsemotion.facial_emotions import HSEmotionRecognizer

logger = logging.getLogger(__name__)

class EmotionDetector:
    """Production-ready emotion detection with error handling and performance monitoring."""

    def __init__(
        self,
        model_name: str = "enet_b0_8_best_afew",
        confidence_threshold: float = 0.3,
        enable_logging: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize emotion detector.
        
        Args:
            model_name: Model name to use (enet_b2_8, enet_b0_8_best_vgaf, etc.)
            confidence_threshold: Minimum confidence to accept emotion detection
            enable_logging: Whether to enable detailed logging
            device: Device to use (cuda/cpu). If None, auto-detect.
        """
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Performance metrics
        self.detection_count = 0
        self.total_detection_time = 0.0
        self.failed_detections = 0
        
        try:
            self.model = HSEmotionRecognizer(
                model_name=model_name,
                device=self.device
            )
            if self.enable_logging:
                logger.info(f"EmotionDetector initialized with model: {model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize emotion detector: {e}")
            raise

    def detect(self, face_image) -> Optional[Dict]:
        """
        Detect emotion from face image.
        
        Args:
            face_image: BGR face image (numpy array)
            
        Returns:
            Dictionary with 'emotion' and 'confidence' keys, or None if detection fails
        """
        start_time = time.time()
        
        try:
            if face_image is None:
                if self.enable_logging:
                    logger.warning("Received None face image")
                self.failed_detections += 1
                return None

            if face_image.size == 0:
                if self.enable_logging:
                    logger.warning("Received empty face image")
                self.failed_detections += 1
                return None

            # Convert BGR to RGB
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            emotion, scores = self.model.predict_emotions(
                rgb,
                logits=True
            )

            confidence = torch.softmax(
                torch.tensor(scores),
                dim=0
            )
            print("con:", confidence)
            confidence = float(torch.max(confidence))

            print("confidence:", confidence)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                if self.enable_logging:
                    logger.debug(f"Emotion confidence {confidence:.3f} below threshold {self.confidence_threshold}")
                self.failed_detections += 1
                return None
            
            # Update metrics
            detection_time = time.time() - start_time
            self.detection_count += 1
            self.total_detection_time += detection_time
            
            if self.enable_logging:
                logger.debug(f"Detected emotion: {emotion} with confidence: {confidence:.3f} in {detection_time:.3f}s")
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "all_scores": scores.tolist() if self.enable_logging else None
            }
            
        except cv2.error as e:
            if self.enable_logging:
                logger.error(f"OpenCV error in emotion detection: {e}")
            self.failed_detections += 1
            return None
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Unexpected error in emotion detection: {e}")
            self.failed_detections += 1
            return None

    def detect_batch(self, face_images: List) -> List[Optional[Dict]]:
        """
        Detect emotions from multiple face images.
        
        Args:
            face_images: List of BGR face images
            
        Returns:
            List of emotion detection results
        """
        return [self.detect(img) for img in face_images]
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        avg_time = (self.total_detection_time / self.detection_count) if self.detection_count > 0 else 0
        return {
            "total_detections": self.detection_count,
            "failed_detections": self.failed_detections,
            "success_rate": (self.detection_count / (self.detection_count + self.failed_detections)) if (self.detection_count + self.failed_detections) > 0 else 0,
            "average_detection_time": avg_time,
            "model_name": self.model_name,
            "device": self.device
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.detection_count = 0
        self.total_detection_time = 0.0
        self.failed_detections = 0
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotion labels."""
        return list(self.model.idx_to_class.values())

