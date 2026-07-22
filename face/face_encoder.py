import numpy as np
import cv2


class FaceEncoder:
    def __init__(self, model_type="deepface"):
        """
        Initialize face encoder.
        Args:
            model_type: Type of model to use ('deepface')
        """
        self.model_type = model_type
        
        if model_type == "deepface":
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
            except ImportError:
                print("Warning: deepface not installed. Please install it with: pip install deepface")
                raise ImportError("deepface is required")
    
    def encode(self, face_image):
        """
        Encode a face image into an embedding vector using DeepFace.
        
        Args:
            face_image: RGB face image (numpy array)
            
        Returns:
            numpy array: Face embedding vector
        """
        if self.model_type == "deepface":
            return self._encode_deepface(face_image)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
    
    def _encode_deepface(self, face_image):
        """Encode using DeepFace library."""
        # Convert RGB to BGR for DeepFace
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        
        try:
            # Get face embedding using DeepFace
            # Use 'skip' detector to bypass detection entirely and avoid Haar cascade
            # Use ArcFace model for best multi-angle accuracy
            # Enable alignment for better accuracy on different angles
            embedding = self.DeepFace.represent(
                face_bgr,
                enforce_detection=False,
                detector_backend='skip',  # Skip detection to avoid Haar cascade
                model_name="ArcFace",  # ArcFace provides best multi-angle accuracy
                align=True  # Enable alignment for better accuracy on different angles
            )
            # print("embedding:", embedding)
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            else:
                raise ValueError("No embedding generated")
        except Exception as e:
            print(f"Error generating embedding with DeepFace: {e}")
            # Return zero embedding as fallback
            return np.zeros(512)  # ArcFace produces 512-dimensional embeddings
    
    def encode_batch(self, face_images):
        """
        Encode multiple face images.
        
        Args:
            face_images: List of RGB face images
            
        Returns:
            numpy array: Stack of face embedding vectors
        """
        embeddings = []
        for face_image in face_images:
            embedding = self.encode(face_image)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def compare(self, embedding1, embedding2, threshold=0.4):
        """
        Compare two face embeddings using cosine distance.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Distance threshold for match (lower is stricter)
            
        Returns:
            tuple: (is_match, distance)
        """
        # Use cosine distance for DeepFace embeddings
        distance = 1 - np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-7
        )
        
        is_match = distance < threshold
        return is_match, distance
