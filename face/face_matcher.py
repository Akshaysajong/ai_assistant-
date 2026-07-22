from .face_encoder import FaceEncoder
from .face_database import FaceDatabase
from database.unified_db import UnifiedDatabase


class FaceMatcher:
    def __init__(self, threshold=0.5, model_type="deepface", db: UnifiedDatabase = None):
        """
        Initialize face matcher.
        
        Args:
            threshold: Distance threshold for face matching (lower = stricter)
            model_type: Type of encoder model ('deepface')
            db: UnifiedDatabase instance for face database
        """
        self.threshold = threshold
        self.encoder = FaceEncoder(model_type=model_type)
        self.database = FaceDatabase(db) if db else FaceDatabase()
    
    def match_face(self, face_image):
        """
        Match a face image against the database.
        
        Args:
            face_image: RGB face image (numpy array)
            
        Returns:
            dict: Matching result with keys:
                - 'matched': bool
                - 'person_name': str or None
                - 'distance': float
                - 'confidence': float (1 - distance, normalized)
        """
        # Encode the face
        embedding = self.encoder.encode(face_image)
        
        # Use cosine similarity for DeepFace embeddings
        use_cosine = True
        
        # Try to recognize
        person_name, distance = self.database.recognize(embedding, self.threshold, use_cosine=use_cosine)
        
        # Calculate confidence (inverse of distance)
        confidence = max(0, 1 - distance)
        
        return {
            'matched': person_name is not None,
            'person_name': person_name,
            'distance': distance,
            'confidence': confidence
        }
    
    # def match_face_with_details(self, face_image):
    #     """
    #     Match a face with detailed comparison results.
        
    #     Args:
    #         face_image: RGB face image (numpy array)
            
    #     Returns:
    #         dict: Detailed matching result
    #     """
    #     embedding = self.encoder.encode(face_image)
    #     result = self.database.recognize_with_details(embedding, self.threshold)
    #     return result
    
    def set_threshold(self, threshold):
        """Update the matching threshold."""
        self.threshold = threshold
    
    def get_database_stats(self):
        """Get database statistics."""
        return self.database.get_stats()
    
    def is_database_empty(self):
        """Check if database has any registered faces."""
        stats = self.database.get_stats()
        return stats['total_persons'] == 0
