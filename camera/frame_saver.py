import logging
from database.unified_db import UnifiedDatabase

logger = logging.getLogger(__name__)


class FrameSaver:

    def __init__(self, db: UnifiedDatabase, enable_logging: bool = True):
        """
        Initialize frame saver with unified database.
        
        Args:
            db: UnifiedDatabase instance
            enable_logging: Whether to enable detailed logging
        """
        self.db = db
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            logger.info("FrameSaver initialized with unified database")

    def save(self, frame, notes: str = None):
        """
        Save frame to database.
        
        Args:
            frame: Frame to save (numpy array)
            notes: Optional notes about the frame
        
        Returns:
            Frame ID if successful, -1 otherwise
        """
        frame_id = self.db.save_frame(frame, notes)
        
        if frame_id > 0 and self.enable_logging:
            logger.info(f"Saved frame with ID: {frame_id}")
        elif self.enable_logging:
            logger.warning("Failed to save frame")
        
        return frame_id
    
    def get_recent_frames(self, limit: int = 10):
        """
        Get recent saved frames metadata.
        
        Args:
            limit: Maximum number of frames to return
            
        Returns:
            List of frame metadata dictionaries
        """
        return self.db.get_recent_frames(limit)
    
    def get_frame(self, frame_id: int):
        """
        Get a saved frame by ID.
        
        Args:
            frame_id: Frame ID
            
        Returns:
            Frame as numpy array or None if not found
        """
        return self.db.get_frame(frame_id)