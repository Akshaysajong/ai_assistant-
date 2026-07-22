"""
Unified SQLite database manager for AI Assistant project.
Stores all data including images, embeddings, and metadata in a single database.
"""

import sqlite3
import os
import logging
import pickle
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedDatabase:
    """
    Unified database manager for all AI Assistant data.
    Handles person tracking, face recognition, emotion detection, and frame storage.
    """
    
    def __init__(self, db_path: str = "data/ai_assistant.db", enable_logging: bool = True):
        """
        Initialize unified database.
        
        Args:
            db_path: Path to SQLite database file
            enable_logging: Whether to enable detailed logging
        """
        self.db_path = db_path
        self.enable_logging = enable_logging
        
        # Create database directory
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        # Initialize database schema
        self._init_database()
        
        if self.enable_logging:
            logger.info(f"Unified database initialized at {db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with error handling."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            if self.enable_logging:
                logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize complete database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # A table for store session
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at REAL NOT NULL,
                    ended_at REAL
                )
            ''')
            
            # Persons table - stores tracked person metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    total_detections INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    UNIQUE(person_id)
                    )
            ''')
            
            # Face embeddings table - stores face recognition data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
                )
            ''')
            
            # Face images table - stores face crops with metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    person_name TEXT,
                    image BLOB NOT NULL,
                    embedding BLOB,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    confidence REAL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE,
                    FOREIGN KEY (person_name) REFERENCES persons (person_name) ON DELETE CASCADE
                )
            ''')
            
            # Matched persons table - stores face match results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matched_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    matched_at REAL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE,
                    FOREIGN KEY (person_name) REFERENCES persons (person_name) ON DELETE CASCADE
                )
            ''')
            
            # Matched person frames table - stores matched person images
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matched_person_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matched_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    image BLOB NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    confidence REAL,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (matched_id) REFERENCES matched_persons (id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE,
                    FOREIGN KEY (person_name) REFERENCES persons (person_name) ON DELETE CASCADE
                )
            ''')
            
            # Emotion entries table - stores emotion detection data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotion_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
                )
            ''')
            
            # Saved frames table - stores complete frames
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS saved_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame BLOB NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    channels INTEGER,
                    timestamp REAL NOT NULL,
                    notes TEXT
                )
            ''')
            
            # Pose data table - stores pose detection results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pose_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    person_id INTEGER,
                    keypoints BLOB,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
                )
            ''')
            
            # Activity data table - stores activity recognition results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    person_id INTEGER,
                    activity TEXT NOT NULL,
                    confidence REAL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions (started_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_persons_name ON persons (person_name)')
            
            # Migration: Add person_name column to face_embeddings if it doesn't exist
            cursor.execute("PRAGMA table_info(face_embeddings)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'person_name' not in columns:
                cursor.execute('ALTER TABLE face_embeddings ADD COLUMN person_name TEXT')
                conn.commit()  # Commit migration before creating index
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_embeddings_name ON face_embeddings (person_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_images_person ON face_images (person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_images_timestamp ON face_images (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_persons_session ON matched_persons (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_persons_person ON matched_persons (person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_persons_name ON matched_persons (person_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_person_frames_matched_id ON matched_person_frames (matched_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_person_frames_person ON matched_person_frames (person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_person_frames_name ON matched_person_frames (person_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_matched_person_frames_timestamp ON matched_person_frames (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_session ON emotion_entries (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_person ON emotion_entries (person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_timestamp ON emotion_entries (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_saved_frames_timestamp ON saved_frames (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pose_session ON pose_data (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pose_person ON pose_data (person_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_data (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_activity_person ON activity_data (person_id)')
            
            conn.commit()
        
        if self.enable_logging:
            logger.info("Database schema initialized successfully")
    
    # ============ Session Management ============
    
    def create_session(self) -> int:
        """Create a new session and return its ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (started_at)
                    VALUES (strftime('%s', 'now'))
                ''')
                session_id = cursor.lastrowid
                if self.enable_logging:
                    logger.info(f"Created new session with ID: {session_id}")
                return session_id
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to create session: {e}")
            return -1
    
    def end_session(self, session_id: int) -> bool:
        """End a session by setting the ended_at timestamp."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions SET ended_at = strftime('%s', 'now')
                    WHERE id = ?
                ''', (session_id,))
                if self.enable_logging:
                    logger.info(f"Ended session with ID: {session_id}")
                return cursor.rowcount > 0
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to end session: {e}")
            return False
    
    def get_current_session(self) -> Optional[Dict]:
        """Get the most recent active session (without ended_at)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM sessions 
                    WHERE ended_at IS NULL 
                    ORDER BY started_at DESC 
                    LIMIT 1
                ''')
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get current session: {e}")
            return None
    
    # ============ Person Management ============
    
    def add_person(self, person_name: str, registered_count: int) -> bool:
        """Add a person by name (for face registration)."""
        total_detections = registered_count
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Check if person already exists
                cursor.execute('SELECT person_name FROM persons WHERE person_name = ?', (person_name,))
                if cursor.fetchone():
                    return True  # Person already exists
                
                # Insert new person with a person_id (negative to avoid conflicts)
                cursor.execute('''
                    INSERT INTO persons (person_id, person_name, first_seen, last_seen, total_detections)
                    VALUES (?, ?, strftime('%s', 'now'), strftime('%s', 'now'), ?)
                ''', (-(abs(hash(person_name)) % 1000000), person_name, total_detections))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add person by name: {e}")
            return False
    
    def update_person_name(self, person_id: int, person_name: str) -> bool:
        """Update person name for a track."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE persons SET person_name = ? WHERE person_id = ?', (person_name, person_id))
                return cursor.rowcount > 0
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to update person name: {e}")
            return False
    
    def add_matched_person(self, session_id: int, person_id: int, person_name: str, match_score: float) -> int:
        """
        Add a face match record linking person_id to registered person.
        
        Args:
            session_id: Current session ID
            person_id: Person ID
            person_name: Name of the registered person matched
            match_score: Face matching confidence score
            
        Returns:
            ID of the inserted matched person record, or -1 on failure
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO matched_persons (session_id, person_id, person_name, match_score)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, person_id, person_name, match_score))
                return cursor.lastrowid
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add matched person: {e}")
            return -1
    
    def add_matched_person_frame(self, matched_id: int, person_id: int, person_name: str, 
                                image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                confidence: float) -> bool:
        """
        Add a matched person frame to the database.
        
        Args:
            matched_id: ID from matched_persons table
            person_id: Person ID
            person_name: Name of the registered person
            image: Face/frame image as numpy array
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', image)
            image_blob = buffer.tobytes()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO matched_person_frames 
                    (matched_id, person_id, person_name, image, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (matched_id, person_id, person_name, image_blob, 
                      bbox[0], bbox[1], bbox[2], bbox[3], confidence))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add matched person frame: {e}")
            return False
    
    def get_person(self, person_id: int) -> Optional[Dict]:
        """Get person data by person ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM persons WHERE person_id = ?', (person_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get person: {e}")
            return None
    
    def get_person_by_name(self, person_name: str) -> Optional[Dict]:
        """Get person data by person name."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM persons WHERE person_name = ?', (person_name,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get person by name: {e}")
            return None
    
    def get_all_persons(self) -> List[Dict]:
        """Get all persons."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM persons ORDER BY last_seen DESC')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get all persons: {e}")
            return []
    
    def delete_person_by_name(self, person_name: str) -> bool:
        """Delete a person by name (for face registration)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Delete face embeddings (cascade will handle this)
                cursor.execute('DELETE FROM face_embeddings WHERE person_name = ?', (person_name,))
                # Delete person record
                cursor.execute('DELETE FROM persons WHERE person_name = ?', (person_name,))
                return cursor.rowcount > 0
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to delete person by name: {e}")
            return False
    
    def update_person_name(self, old_name: str, new_name: str) -> bool:
        """Update a person's name."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Update person record
                cursor.execute('UPDATE persons SET person_name = ? WHERE person_name = ?', (new_name, old_name))
                # Update face embeddings
                cursor.execute('UPDATE face_embeddings SET person_name = ? WHERE person_name = ?', (new_name, old_name))
                return cursor.rowcount > 0
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to update person name: {e}")
            return False
    
    # ============ Face Embedding Management ============
    
    def add_person_embedding(self, person_name: str, embedding: np.ndarray) -> bool:
        """Add person embedding for a person."""
        try:
            embedding_blob = pickle.dumps(embedding)
            # Use person_id based on person name hash
            person_id = -(abs(hash(person_name)) % 1000000)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO face_embeddings (person_id, person_name, embedding, embedding_dim)
                    VALUES (?, ?, ?, ?)
                ''', (person_id, person_name, embedding_blob, len(embedding)))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add face embedding: {e}")
            return False

    
    
    def get_face_embeddings(self, person_name: str) -> List[np.ndarray]:
        """Get all embeddings for a person."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT embedding FROM face_embeddings WHERE person_name = ?', (person_name,))
                return [pickle.loads(row['embedding']) for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get face embeddings: {e}")
            return []
    
    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Get all embeddings organized by person."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT person_name, embedding FROM face_embeddings')
                result = {}
                for row in cursor.fetchall():
                    person_name = row['person_name']
                    if person_name not in result:
                        result[person_name] = []
                    result[person_name].append(pickle.loads(row['embedding']))
                return result
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get all embeddings: {e}")
            return {}
    
    # ============ Face Image Management ============
    
    def add_face_image(self, person_id: int, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                      confidence: float, timestamp: Optional[float] = None,
                      person_name: Optional[str] = None, embedding: Optional[np.ndarray] = None) -> bool:
        """Add face image for a person with optional person name and embedding."""
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', image)
            image_blob = buffer.tobytes()
            
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            
            embedding_blob = pickle.dumps(embedding) if embedding is not None else None
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO face_images (person_id, person_name, image, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (person_id, person_name, image_blob, embedding_blob, bbox[0], bbox[1], bbox[2], bbox[3], confidence, timestamp))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add face image: {e}")
            return False
    
    def save_person_faces_with_embeddings(self, person_name: str, face_data: List[Dict]) -> int:
        """
        Save all detected faces for a person with their embeddings.
        
        Args:
            person_name: Name of the person
            face_data: List of dictionaries containing:
                - 'image': numpy array of face image
                - 'embedding': numpy array of face embedding
                - 'bbox': tuple of (x1, y1, x2, y2)
                - 'confidence': float
                - 'timestamp': float (optional)
        
        Returns:
            Number of faces saved
        """
        saved_count = 0
        for face in face_data:
            try:
                success = self.add_face_image(
                    person_id=-(abs(hash(person_name)) % 1000000),  # Use person_id
                    image=face['image'],
                    bbox=face['bbox'],
                    confidence=face['confidence'],
                    timestamp=face.get('timestamp'),
                    person_name=person_name,
                    embedding=face['embedding']
                )
                if success:
                    saved_count += 1
            except Exception as e:
                if self.enable_logging:
                    logger.error(f"Failed to save face for {person_name}: {e}")
        
        if self.enable_logging:
            logger.info(f"Saved {saved_count}/{len(face_data)} faces for {person_name}")
        
        return saved_count
    
    def get_face_images(self, person_id: int, limit: int = 10) -> List[np.ndarray]:
        """Get recent face images for a person."""
        try:
            import cv2
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT image FROM face_images 
                    WHERE person_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (person_id, limit))
                return [cv2.imdecode(np.frombuffer(row['image'], dtype=np.uint8), cv2.IMREAD_COLOR) 
                        for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get face images: {e}")
            return []
    
    # ============ Emotion Management ============
    
    def add_emotion(self, session_id: int, person_id: int, emotion: str, confidence: float, 
                   timestamp: Optional[float] = None) -> bool:
        """Add emotion entry for a person."""
        try:
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO emotion_entries (session_id, person_id, emotion, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, person_id, emotion, confidence, timestamp))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add emotion: {e}")
            return False
    
    def get_emotions(self, person_id: int, limit: int = 50) -> List[Dict]:
        """Get emotion entries for a person."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT emotion, confidence, timestamp 
                    FROM emotion_entries 
                    WHERE person_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (person_id, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get emotions: {e}")
            return []
    
    def get_emotions_by_session(self, session_id: int, time_window_seconds: Optional[int] = None) -> List[Dict]:
        """Get all emotion entries for a session, optionally within a time window."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if time_window_seconds:
                    import time
                    start_time = time.time() - time_window_seconds
                    cursor.execute('''
                        SELECT person_id, emotion, confidence, timestamp
                        FROM emotion_entries 
                        WHERE session_id = ? AND timestamp >= ?
                        ORDER BY timestamp ASC
                    ''', (session_id, start_time))
                else:
                    cursor.execute('''
                        SELECT person_id, emotion, confidence, timestamp
                        FROM emotion_entries 
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    ''', (session_id,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get emotions by session: {e}")
            return []
    
    # ============ Frame Management ============
    
    def save_frame(self, frame: np.ndarray, notes: Optional[str] = None) -> int:
        """Save a complete frame to database."""
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', frame)
            frame_blob = buffer.tobytes()
            
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) > 2 else 1
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO saved_frames (frame, width, height, channels, timestamp, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (frame_blob, width, height, channels, datetime.now().timestamp(), notes))
                return cursor.lastrowid
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to save frame: {e}")
            return -1
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get a saved frame by ID."""
        try:
            import cv2
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT frame FROM saved_frames WHERE id = ?', (frame_id,))
                row = cursor.fetchone()
                if row:
                    return cv2.imdecode(np.frombuffer(row['frame'], dtype=np.uint8), cv2.IMREAD_COLOR)
                return None
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get frame: {e}")
            return None
    
    def get_recent_frames(self, limit: int = 10) -> List[Dict]:
        """Get recent saved frames metadata."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, width, height, timestamp, notes 
                    FROM saved_frames 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get recent frames: {e}")
            return []
    
    # ============ Pose Management ============
    
    def add_pose(self, session_id: int, person_id: Optional[int], keypoints: np.ndarray, bbox: Tuple[int, int, int, int],
                timestamp: Optional[float] = None) -> bool:
        """Add pose data."""
        try:
            keypoints_blob = pickle.dumps(keypoints)
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO pose_data (session_id, person_id, keypoints, bbox_x1, bbox_y1, bbox_x2, bbox_y2, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, person_id, keypoints_blob, bbox[0], bbox[1], bbox[2], bbox[3], timestamp))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add pose: {e}")
            return False
    
    # ============ Activity Management ============
    
    def add_activity(self, session_id: int, person_id: Optional[int], activity: str, confidence: float,
                    timestamp: Optional[float] = None) -> bool:
        """Add activity data."""
        try:
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO activity_data (session_id, person_id, activity, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, person_id, activity, confidence, timestamp))
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to add activity: {e}")
            return False
    
    def get_activities(self, person_id: int, limit: int = 10) -> List[Dict]:
        """Get recent activities for a person."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM activity_data 
                    WHERE person_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (person_id, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get activities: {e}")
            return []
    
    # ============ Statistics ============
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                cursor.execute('SELECT COUNT(*) FROM persons')
                stats['total_persons'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM face_embeddings')
                stats['total_embeddings'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM face_images')
                stats['total_face_images'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM emotion_entries')
                stats['total_emotions'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM saved_frames')
                stats['total_frames'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM pose_data')
                stats['total_poses'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM activity_data')
                stats['total_activities'] = cursor.fetchone()[0]
                
                # Database size
                stats['db_size_bytes'] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return stats
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to get database stats: {e}")
            return {}
    
    # ============ Cleanup ============
    
    def clear_person_data(self, person_id: int) -> bool:
        """Clear all data for a specific person."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM persons WHERE person_id = ?', (person_id,))
                # Cascading deletes will handle related tables
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to clear person data: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """Clear all data from database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                tables = ['persons', 'face_embeddings', 'face_images', 'emotion_entries', 
                         'saved_frames', 'pose_data', 'activity_data']
                for table in tables:
                    cursor.execute(f'DELETE FROM {table}')
                return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Failed to clear all data: {e}")
            return False
