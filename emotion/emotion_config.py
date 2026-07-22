"""
Configuration management for emotion detection system.
Centralizes all emotion-related parameters for easy tuning and maintenance.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EmotionDetectorConfig:
    """Configuration for EmotionDetector."""
    model_name: str = "enet_b2_8"
    confidence_threshold: float = 0.3
    enable_logging: bool = True
    device: Optional[str] = None  # None = auto-detect


@dataclass
class EmotionTrackerConfig:
    """Configuration for EmotionTracker."""
    history_size: int = 10
    smoothing_window: int = 3
    min_confidence: float = 0.3
    track_timeout: float = 5.0
    enable_logging: bool = True


@dataclass
class EmotionMemoryConfig:
    """Configuration for EmotionMemory."""
    max_history: int = 50
    db_path: Optional[str] = None  # Path to SQLite database file
    auto_save: bool = True
    enable_logging: bool = True


@dataclass
class EmotionSystemConfig:
    """Complete configuration for the emotion detection system."""
    detector: EmotionDetectorConfig
    tracker: EmotionTrackerConfig
    memory: EmotionMemoryConfig
    
    def __post_init__(self):
        """Set default database path if not provided."""
        if self.memory.db_path is None:
            # Default to data/emotion.db in the project directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.memory.db_path = os.path.join(base_dir, "data", "emotion.db")


class EmotionConfigManager:
    """
    Manages emotion system configuration with file persistence.
    Supports loading from JSON files and environment variables.
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "emotion_config.json"
    )
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration JSON file (uses default if not provided)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> EmotionSystemConfig:
        """Load configuration from file or create default configuration."""
        if os.path.exists(self.config_path):
            try:
                return self.load_from_file(self.config_path)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}. Using defaults.")
        
        # Create default configuration
        return self.create_default_config()
    
    def create_default_config(self) -> EmotionSystemConfig:
        """Create default configuration."""
        return EmotionSystemConfig(
            detector=EmotionDetectorConfig(),
            tracker=EmotionTrackerConfig(),
            memory=EmotionMemoryConfig()
        )
    
    def load_from_file(self, path: str) -> EmotionSystemConfig:
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            EmotionSystemConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return EmotionSystemConfig(
            detector=EmotionDetectorConfig(**data.get("detector", {})),
            tracker=EmotionTrackerConfig(**data.get("tracker", {})),
            memory=EmotionMemoryConfig(**data.get("memory", {}))
        )
    
    def save_to_file(self, path: Optional[str] = None):
        """
        Save current configuration to JSON file.
        
        Args:
            path: Optional path to save to (uses default if not provided)
        """
        save_path = path or self.config_path
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        data = {
            "detector": asdict(self.config.detector),
            "tracker": asdict(self.config.tracker),
            "memory": asdict(self.config.memory)
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved configuration to {save_path}")
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary with configuration updates
        """
        if "detector" in updates:
            for key, value in updates["detector"].items():
                if hasattr(self.config.detector, key):
                    setattr(self.config.detector, key, value)
        
        if "tracker" in updates:
            for key, value in updates["tracker"].items():
                if hasattr(self.config.tracker, key):
                    setattr(self.config.tracker, key, value)
        
        if "memory" in updates:
            for key, value in updates["memory"].items():
                if hasattr(self.config.memory, key):
                    setattr(self.config.memory, key, value)
        
        logger.info("Configuration updated from dictionary")
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        # Detector settings
        if "EMOTION_MODEL_NAME" in os.environ:
            self.config.detector.model_name = os.environ["EMOTION_MODEL_NAME"]
        if "EMOTION_CONFIDENCE_THRESHOLD" in os.environ:
            self.config.detector.confidence_threshold = float(os.environ["EMOTION_CONFIDENCE_THRESHOLD"])
        
        # Tracker settings
        if "EMOTION_HISTORY_SIZE" in os.environ:
            self.config.tracker.history_size = int(os.environ["EMOTION_HISTORY_SIZE"])
        if "EMOTION_SMOOTHING_WINDOW" in os.environ:
            self.config.tracker.smoothing_window = int(os.environ["EMOTION_SMOOTHING_WINDOW"])
        if "EMOTION_TRACK_TIMEOUT" in os.environ:
            self.config.tracker.track_timeout = float(os.environ["EMOTION_TRACK_TIMEOUT"])
        
        # Memory settings
        if "EMOTION_MAX_HISTORY" in os.environ:
            self.config.memory.max_history = int(os.environ["EMOTION_MAX_HISTORY"])
        if "EMOTION_PERSISTENCE_PATH" in os.environ:
            self.config.memory.persistence_path = os.environ["EMOTION_PERSISTENCE_PATH"]
        
        logger.info("Configuration updated from environment variables")
    
    def get_config(self) -> EmotionSystemConfig:
        """Get current configuration."""
        return self.config
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.create_default_config()
        logger.info("Configuration reset to defaults")
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate detector config
            assert 0 <= self.config.detector.confidence_threshold <= 1, \
                "Detector confidence_threshold must be between 0 and 1"
            
            # Validate tracker config
            assert self.config.tracker.history_size > 0, \
                "Tracker history_size must be positive"
            assert self.config.tracker.smoothing_window > 0, \
                "Tracker smoothing_window must be positive"
            assert 0 <= self.config.tracker.min_confidence <= 1, \
                "Tracker min_confidence must be between 0 and 1"
            assert self.config.tracker.track_timeout > 0, \
                "Tracker track_timeout must be positive"
            
            # Validate memory config
            assert self.config.memory.max_history > 0, \
                "Memory max_history must be positive"
            
            logger.info("Configuration validation passed")
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[EmotionConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> EmotionConfigManager:
    """
    Get global configuration manager instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        EmotionConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = EmotionConfigManager(config_path)
    return _config_manager


def reset_config_manager():
    """Reset global configuration manager instance."""
    global _config_manager
    _config_manager = None
