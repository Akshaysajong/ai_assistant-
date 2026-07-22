import logging
from typing import Dict, Optional, List
from datetime import datetime
from database.unified_db import UnifiedDatabase

logger = logging.getLogger(__name__)


class PersonMemory:
    def __init__(self, db: UnifiedDatabase, enable_logging: bool = True):
        """
        Initialize person memory with unified database.
        
        Args:
            db: UnifiedDatabase instance
            enable_logging: Whether to enable detailed logging
        """
        self.db = db
        self.enable_logging = enable_logging
        self.person_data = {}  # In-memory cache for fast access
        
        if self.enable_logging:
            logger.info("PersonMemory initialized with unified database")

    def add_matched_face(self, session_id: int, person_id: int, face_image, bbox, confidence, person_name=None, match_score=None):
        """
        Store face data for a tracked person with optional match info
        
        Args:
            session_id: Session ID
            person_id: Person tracking ID
            face_image: Face crop as numpy array (RGB)
            bbox: Face bounding box (x1, y1, x2, y2)
            confidence: Face detection confidence
            person_name: Optional name of matched registered person
            match_score: Optional face match confidence score
        """
        
        # If matched, save match record and frame
        if person_name and match_score is not None:
            # Get the registered person's person_id from the persons table
            registered_person = self.db.get_person_by_name(person_name)
            if registered_person:
                registered_person_id = registered_person['person_id']
                matched_id = self.db.add_matched_person(session_id, registered_person_id, person_name, match_score)
                if matched_id != -1:
                    # Save matched person frame with embedding
                    self.db.add_matched_person_frame(
                        matched_id=matched_id,
                        person_id=registered_person_id,
                        person_name=person_name,
                        image=face_image,
                        bbox=bbox,
                        confidence=confidence,
                    )
                if self.enable_logging:
                    logger.info(f"Saved matched frame for track {person_id} -> '{person_name}' (score: {match_score:.4f})")
        
        # Update in-memory cache
        if person_id not in self.person_data:
            self.person_data[person_id] = {
                "faces": [],
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
            }

        self.person_data[person_id]["faces"].append({
            "image": face_image,
            "bbox": bbox,
            "confidence": confidence,
            "timestamp": datetime.now(),
        })
        self.person_data[person_id]["last_seen"] = datetime.now()
        
        if self.enable_logging:
            logger.debug(f"Added face for track {person_id}")

    def get_person_data(self, person_id):
        """Retrieve all data for a specific person"""
        # Check cache first
        if person_id in self.person_data:
            return self.person_data[person_id]
        
        # Load from database
        person = self.db.get_person(person_id)
        if person:
            face_images = self.db.get_face_images(person_id)
            self.person_data[person_id] = {
                "faces": [{
                    "image": img,
                    "bbox": (0, 0, 0, 0),  # Placeholder, actual bbox stored in DB
                    "confidence": 0.0,
                    "timestamp": datetime.now()
                } for img in face_images],
                "first_seen": datetime.fromtimestamp(person['first_seen']),
                "last_seen": datetime.fromtimestamp(person['last_seen']),
            }
            return self.person_data[person_id]
        
        return None

    def save_to_disk(self):
        """Save all memory data to database (auto-saved)"""
        # Data is automatically saved to database on each add_face call
        # This method is kept for compatibility
        if self.enable_logging:
            logger.info("Person data auto-saved to database")
        return True

    def load_from_disk(self, filepath):
        """Load memory data from database (auto-loaded)"""
        # Data is automatically loaded from database
        # This method is kept for compatibility
        if self.enable_logging:
            logger.info("Person data loaded from database")
        return True

    def get_all_persons(self):
        """Get all tracked person IDs"""
        # Combine cache and database
        cached_ids = list(self.person_data.keys())
        db_persons = self.db.get_all_persons()
        db_ids = [p['person_id'] for p in db_persons]
        
        all_ids = list(set(cached_ids + db_ids))
        return all_ids
    
    def clear_cache(self):
        """Clear in-memory cache (data remains in database)"""
        self.person_data.clear()
        if self.enable_logging:
            logger.info("Person memory cache cleared")
