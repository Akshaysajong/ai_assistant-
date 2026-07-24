import os
import cv2
import time
import numpy as np
from camera.camera_img import ImageLoader
from face.face_cropper import FaceCropper
from face.face_encoder import FaceEncoder
from face.face_database import FaceDatabase
from face.face_detector import FaceDetector
from database.unified_db import UnifiedDatabase

class PersonRegister:
    def __init__(self):
        self.camera = ImageLoader(r"C:\project\python\ai assistant\images\nikhil")
        self.face_detector = FaceDetector()
        self.face_cropper = FaceCropper(
            face_size=(160, 160),
            margin=20,
            detector=self.face_detector,
        )
        self.face_encoder = FaceEncoder(model_type="deepface")
        
        # Initialize unified database
        db_path = "data/ai_assistant.db"
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db = UnifiedDatabase(db_path=db_path, enable_logging=True)
        self.face_database = FaceDatabase(self.db, enable_logging=True)
    
    def register(self, person_name: str, max_images=50):
        """
        Register a person by automatically capturing face images while moving head.
        Like iPhone Face ID setup - captures automatically as you move your face.
        
        Args:
            person_name: Name/ID of the person to register
            max_images: Maximum number of face images to capture
            
        Returns:
            int: Number of successfully registered faces
        """
        save_dir = f"data/register/{person_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Store all embeddings for averaging and individual face data
        all_embeddings = []
        all_face_data = []  # Store individual face data with embeddings
        registered_count = 0
        last_capture_time = 0
        capture_interval = 0.3  # Capture every 0.3 seconds (faster for more angles)
        last_face_center = None
        min_movement = 15  # Minimum pixel movement to trigger capture (lower for more diverse angles)
        
        print(f"Registering {person_name} from images.")
        print(f"Processing images from folder: {self.camera.folder}")
        print(f"Found {len(self.camera.images)} images to process.")
        print(f"Capturing up to {max_images} faces. Press 'q' to cancel.")
        
        cv2.namedWindow("Face Registration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Registration", 720, 720)
        
        while registered_count < max_images:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                print(f"No more images to process. Completed {registered_count} registrations.")
                break
            
            print(f"Processing image {self.camera.index}/{len(self.camera.images)}")
            
            # Detect and crop faces
            try:
                faces = self.face_cropper.crop(frame)
                print(f"Detected {len(faces)} faces in image")
            except Exception as e:
                print(f"Error in face detection: {e}")
                continue
            
            # Draw face detection boxes on frame
            display_frame = frame.copy()
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                confidence = face["confidence"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{confidence:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display progress
            cv2.putText(display_frame, f"Progress: {registered_count}/{max_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Move head slowly in all directions", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to cancel", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Face Registration", display_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Registration cancelled by user.")
                cv2.destroyAllWindows()
                return registered_count
            
            if not faces:
                print(f"No face detected. Progress: {registered_count}/{max_images}")
                time.sleep(0.1)
                continue
            
            # Get the best face
            best_face = max(faces, key=lambda x: x["confidence"])
            
            # Calculate face center for movement detection
            x1, y1, x2, y2 = best_face["bbox"]
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Check if enough time has passed and face has moved
            current_time = time.time()
            should_capture = False
            
            if last_face_center is None:
                should_capture = True
            else:
                # Calculate movement
                movement = abs(face_center[0] - last_face_center[0]) + abs(face_center[1] - last_face_center[1])
                
                if movement > min_movement and (current_time - last_capture_time) > capture_interval:
                    should_capture = True
            
            if should_capture:
                try:
                    # Save face image
                    face_bgr = cv2.cvtColor(best_face["image"], cv2.COLOR_RGB2BGR)
                    face_path = f"{save_dir}/{registered_count}.jpg"
                    cv2.imwrite(face_path, face_bgr)
                    print(f"Saved face image to {face_path}")
                    
                    # Encode face and store for averaging
                    print("Encoding face...")
                    face_embedding = self.face_encoder.encode(best_face["image"])
                    print(f"Face embedding shape: {face_embedding.shape}")
                    print(f"Embedding sample (first 5 values): {face_embedding[:5]}")
                    print(f"Embedding min/max: {face_embedding.min():.4f} / {face_embedding.max():.4f}")
                    print(f"Embedding norm: {np.linalg.norm(face_embedding):.4f}")
                    
                    # Store embedding for later averaging
                    all_embeddings.append(face_embedding)
                    
                    # Store individual face data with embedding for database
                    all_face_data.append({
                        'image': best_face["image"],
                        'embedding': face_embedding,
                        'bbox': best_face["bbox"],
                        'confidence': best_face["confidence"],
                        'timestamp': time.time()
                    })
                    
                    registered_count += 1
                    last_capture_time = current_time
                    last_face_center = face_center
                    
                    print(f"Captured face {registered_count}/{max_images}")
                    
                except Exception as e:
                    print(f"Error during face capture: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.05)
        
        # Cleanup
        cv2.destroyAllWindows()
        
        # Average embeddings and save to database
        if all_embeddings:
            try:
                print(f"Averaging {len(all_embeddings)} embeddings...")
                
                # # Compare embeddings to see if they're different
                # if len(all_embeddings) > 1:
                #     for i in range(len(all_embeddings)):
                #         for j in range(i+1, len(all_embeddings)):
                #             diff = np.linalg.norm(all_embeddings[i] - all_embeddings[j])
                #             print(f"Distance between embedding {i} and {j}: {diff:.4f}")
                
                avg_embedding = np.mean(all_embeddings, axis=0)
                print(f"Average embedding shape: {avg_embedding.shape}")
                print(f"Avg embedding sample (first 5 values): {avg_embedding[:5]}")
                print(f"Avg embedding min/max: {avg_embedding.min():.4f} / {avg_embedding.max():.4f}")
                
                # Normalize the averaged embedding
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-7)
                print(f"Normalized embedding norm: {np.linalg.norm(avg_embedding):.4f}")
                
                # Add person to database first
                print(f"Adding person {person_name} to database...")
                self.face_database.add_person(person_name, registered_count)
                print(f"Successfully added person {person_name} to database")
                
                # Save all individual faces with embeddings to database
                print(f"Saving {len(all_face_data)} individual faces with embeddings...")
                saved_faces = self.db.save_person_faces_with_embeddings(person_name, all_face_data)
                print(f"Successfully saved {saved_faces} faces with embeddings to database")
                
                # Add averaged embedding for recognition
                print(f"Adding averaged embedding to database for {person_name}...")
                self.face_database.add_person_embedding(person_name, avg_embedding)
                print("Successfully added averaged embedding to database")
                
                # Save database
                print("Saving database...")
                self.face_database.save()
                print("Database saved successfully")
            except Exception as e:
                print(f"Error averaging/saving embeddings: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No embeddings captured to save.")
        
        # Print statistics
        try:
            stats = self.face_database.get_stats()
            print(f"Registration complete. Captured {registered_count} faces.")
            print(f"Database stats: {stats['total_persons']} persons, {stats['total_embeddings']} total embeddings")
        except Exception as e:
            print(f"Error getting stats: {e}")
            print(f"Registration complete. Captured {registered_count} faces.")
        
        return registered_count
        