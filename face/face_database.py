import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from database.unified_db import UnifiedDatabase

logger = logging.getLogger(__name__)


class FaceDatabase:
    def __init__(self, db: UnifiedDatabase, enable_logging: bool = True):
        """
        Initialize face database with unified database.
        
        Args:
            db: UnifiedDatabase instance
            enable_logging: Whether to enable detailed logging
        """
        self.db = db
        self.enable_logging = enable_logging
        self.database: Dict[str, List[np.ndarray]] = {}  # In-memory cache
        self._load_from_database()
        
        if self.enable_logging:
            logger.info("FaceDatabase initialized with unified database")
    
    def _load_from_database(self):
        """Load embeddings from database into cache."""
        self.database = self.db.get_all_embeddings()
        if self.enable_logging:
            logger.info(f"Loaded {len(self.database)} persons from database")
    
    def add_person(self, person_name: str, registered_count: int = 0) -> bool:
        """
        Add a person to the database (for face registration).
        
        Args:
            person_name: Name/ID of the person
            registered_count: Number of registered faces (for total_detections)
            
        Returns:
            True if person was added, False if already exists or error
        """
        return self.db.add_person(person_name, registered_count)
    
    def add_person_embedding(self, person_name: str, embedding: np.ndarray):
        """
        Add a person embedding to the database.
        
        Args:
            person_name: Name/ID of the person
            embedding: Person embedding vector
        """
        # Add to database
        self.db.add_person_embedding(person_name, embedding)
        
        # Update cache
        if person_name not in self.database:
            self.database[person_name] = []
        self.database[person_name].append(embedding)
        
        if self.enable_logging:
            logger.debug(f"Added embedding for {person_name}")
    
    def add_faces(self, person_name: str, embeddings: List[np.ndarray]):
        """
        Add multiple face embeddings for a person.
        
        Args:
            person_name: Name/ID of the person
            embeddings: List of face embedding vectors
        """
        for embedding in embeddings:
            self.add_face(person_name, embedding)
    
    def get_embeddings(self, person_name: str) -> Optional[List[np.ndarray]]:
        """
        Get all embeddings for a person.
        
        Args:
            person_name: Name/ID of the person
            
        Returns:
            List of embeddings or None if person not found
        """
        return self.database.get(person_name)
    
    def get_average_embedding(self, person_name: str) -> Optional[np.ndarray]:
        """
        Get the average embedding for a person.
        
        Args:
            person_name: Name/ID of the person
            
        Returns:
            Average embedding vector or None if person not found
        """
        embeddings = self.get_embeddings(person_name)
        if embeddings and len(embeddings) > 0:
            return np.mean(embeddings, axis=0)
        return None
    
    def recognize(self, embedding: np.ndarray, threshold: float = 0.2, use_cosine=False) -> Tuple[Optional[str], float]:
        """
        Recognize a face by comparing with database.
        
        Args:
            embedding: Face embedding to recognize
            threshold: Distance threshold for match
            use_cosine: Use cosine similarity instead of Euclidean distance
            
        Returns:
            Tuple of (person_name, distance) or (None, distance) if no match
        """
        best_match = None
        best_distance = float('inf')
        
        for person_name, embeddings in self.database.items():
            if len(embeddings) == 0:
                continue
            
            # Compare with average embedding
            avg_embedding = np.mean(embeddings, axis=0)
            
            if use_cosine:
                # Use cosine similarity (1 - cosine = distance)
                distance = 1 - np.dot(embedding, avg_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(avg_embedding) + 1e-7
                )
            else:
                # Use Euclidean distance
                distance = np.linalg.norm(embedding - avg_embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = person_name
        
        if best_distance < threshold:
            return best_match, best_distance
        
        return None, best_distance
    
    def recognize_with_details(self, embedding: np.ndarray, threshold: float = 0.6) -> Dict:
        """
        Recognize a face with detailed comparison results.
        
        Args:
            embedding: Face embedding to recognize
            threshold: Distance threshold for match
            
        Returns:
            Dictionary with recognition details
        """
        results = []
        
        for person_name, embeddings in self.database.items():
            if len(embeddings) == 0:
                continue
            
            # Compare with each stored embedding
            distances = [np.linalg.norm(embedding - emb) for emb in embeddings]
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            results.append({
                'person_name': person_name,
                'avg_distance': avg_distance,
                'min_distance': min_distance,
                'num_embeddings': len(embeddings)
            })
        
        # Sort by average distance
        results.sort(key=lambda x: x['avg_distance'])
        
        if results and results[0]['avg_distance'] < threshold:
            return {
                'matched': True,
                'person_name': results[0]['person_name'],
                'distance': results[0]['avg_distance'],
                'all_results': results
            }
        
        return {
            'matched': False,
            'person_name': None,
            'distance': results[0]['avg_distance'] if results else float('inf'),
            'all_results': results
        }
    
    def remove_person(self, person_name: str) -> bool:
        """
        Remove a person from the database.
        
        Args:
            person_name: Name/ID of the person to remove
            
        Returns:
            True if person was removed, False if not found
        """
        # Remove from database
        success = self.db.delete_person_by_name(person_name)
        
        # Remove from cache
        if person_name in self.database:
            del self.database[person_name]
        
        if success and self.enable_logging:
            logger.info(f"Removed {person_name} from database")
        
        return success
    
    def update_person(self, old_name: str, new_name: str) -> bool:
        """
        Update a person's name.
        
        Args:
            old_name: Current name of the person
            new_name: New name for the person
            
        Returns:
            True if person was updated, False if not found
        """
        # Update in database
        success = self.db.update_person_name(old_name, new_name)
        
        # Update in cache
        if old_name in self.database:
            self.database[new_name] = self.database.pop(old_name)
        
        if success and self.enable_logging:
            logger.info(f"Updated person name from {old_name} to {new_name}")
        
        return success
    
    def get_all_persons(self) -> List[str]:
        """
        Get list of all persons in database.
        
        Returns:
            List of person names
        """
        return list(self.database.keys())
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_persons': len(self.database),
            'total_embeddings': sum(len(emb) for emb in self.database.values()),
            'persons': {}
        }
        
        for person_name, embeddings in self.database.items():
            stats['persons'][person_name] = len(embeddings)
        
        return stats
    
    def save(self):
        """Save database to unified database (auto-saved)."""
        # Data is automatically saved to database on each add_face call
        # This method is kept for compatibility
        if self.enable_logging:
            logger.info("Face database auto-saved to unified database")
    
    def load(self):
        """Load database from unified database (auto-loaded)."""
        # Reload from database
        self._load_from_database()
        if self.enable_logging:
            logger.info("Face database reloaded from unified database")
    
    def clear(self):
        """Clear all data from database."""
        # Clear from database
        self.db.clear_all_data()
        # Clear cache
        self.database.clear()
        if self.enable_logging:
            logger.info("Face database cleared")
