import os
import time
import cv2
from ultralytics.engine.results import Boxes
from camera.camera_org import Camera
from camera.frame_saver import FrameSaver
from tracking.person_tracking import PersonTracker
from tracking.person_memory import PersonMemory
from face.face_cropper import FaceCropper
from face.face_detector import FaceDetector
from face.face_matcher import FaceMatcher
from database.unified_db import UnifiedDatabase

from pose.pose_detector import PoseDetector
from pose.pose_analyzer import PoseAnalyzer
from pose.activity_recognition import ActivityRecognizer
from emotion.emotion_detector import EmotionDetector
from emotion.emotion_tracker import EmotionTracker
from emotion.emotion_memory import EmotionMemory



# Initialize unified database
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "ai_assistant.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
db = UnifiedDatabase(db_path=DB_PATH, enable_logging=True)

# Create a new session for this run
session_id = db.create_session()
print(f"Created session with ID: {session_id}")

camera = Camera()
saver = FrameSaver(db=db, enable_logging=True)
tracker = PersonTracker()
memory = PersonMemory(db=db, enable_logging=True)
face_detector = FaceDetector()
face_cropper = FaceCropper(
    face_size=(160, 160),
    margin=20,
    detector=face_detector,
)
face_matcher = FaceMatcher(threshold=0.5, model_type="deepface", db=db)
pose_detector = PoseDetector()
pose_analyzer = PoseAnalyzer()
activity_recognizer = ActivityRecognizer(window_size=30)
# Emotion system configuration
emotion_detector = EmotionDetector(
    model_name="enet_b2_8",
    confidence_threshold=0.3,
    enable_logging=True
)
emotion_tracker = EmotionTracker(
    history_size=10,
    smoothing_window=3,
    min_confidence=0.3,
    track_timeout=5.0,
    enable_logging=True
)
emotion_memory = EmotionMemory(
    db=db,
    max_history=50,
    enable_logging=True
)

#FPS optimization settings
frame_skip = 2  # Process every Nth frame for face recognition (skip frames in between)
frame_counter = 0
max_faces_per_person = 1  # Only process the best face per person to save time

# Track face matches for temporal consistency
face_match_history = {}  # person_id -> {person_name: match_count}
required_consecutive_matches = 3  # Require 3 consecutive matches to confirm


print("Press:")
print("S = Save Image")
print("Q = Quit")

# Check if database has registered faces
db_stats = face_matcher.get_database_stats()
if db_stats['total_persons'] == 0:
    print("Warning: No faces registered in database. Run face registration first.")
else:
    print(f"Loaded {db_stats['total_persons']} persons with {db_stats['total_embeddings']} face embeddings.")

prev = time.time()

while True:

    ret, frame = camera.read()
    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not ret:
        break
    
    results = tracker.track(frame)
    person_count = len(results)

    # # Increment frame counter
    # frame_counter += 1
    # process_face_recognition = (frame_counter % frame_skip == 0)

    for person in results:
        x1, y1, x2, y2 = person["bbox"]
        person_id = person["id"]
        confidence = person["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{person_id} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop face from person bounding box
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size > 0:
            faces = face_cropper.crop(person_crop)
            # Only process the best face (highest confidence) to save time
            if faces:
                faces = sorted(faces, key=lambda x: x["confidence"], reverse=True)[:max_faces_per_person]
            for face in faces:
                face_bbox = face["bbox"]
                fx1, fy1, fx2, fy2 = face_bbox
                # Convert face bbox coordinates back to frame coordinates
                abs_fx1, abs_fy1 = x1 + fx1, y1 + fy1
                abs_fx2, abs_fy2 = x1 + fx2, y1 + fy2
                
                # Match face against database
                match_result = face_matcher.match_face(face["image"])

                # Initialize match history for this track
                if person_id not in face_match_history:
                    face_match_history[person_id] = {}

                # Update match history
                if match_result['matched']:
                    person_name = match_result['person_name']
                    if person_name not in face_match_history[person_id]:
                        face_match_history[person_id][person_name] = 0
                    face_match_history[person_id][person_name] += 1

                    # Check if we have enough consecutive matches
                    if face_match_history[person_id][person_name] >= required_consecutive_matches:
                        label = f"{person_name} (Matched)"
                        color = (0, 255, 0)  # Green for known
                        confidence_percent = match_result['confidence'] * 100
                        confidence_text = f"Confidence: {confidence_percent:.0f}%"
                        status_text = "Person Present"
                        
                    else:
                        # Still building confidence
                        label = "Analyzing..."
                        color = (255, 255, 0)  # Yellow for analyzing
                        confidence_text = f"Building confidence: {face_match_history[person_id][person_name]}/{required_consecutive_matches}"
                        status_text = "Verifying"

                    # Save matched face data to memory
                    memory.add_matched_face(
                        session_id=session_id,
                        person_id=person_id,
                        face_image=face["image"],
                        bbox=face_bbox,
                        confidence=face["confidence"],
                        person_name=person_name if match_result['matched'] else None,
                        match_score=match_result['confidence'] if match_result['matched'] else None
                    )
                    
                    
                    # Detect emotion from face
                    try:
                        emotion_result = emotion_detector.detect(face["image"])
                        print("emotion_result:", emotion_result)
                        if emotion_result:
                            # Update emotion tracker and unified memory
                            emotion_tracker.emoton_tracker(person_id, emotion_result['emotion'], emotion_result['confidence'])
                            
                            # Only store emotion in database if person is matched/registered
                            if match_result['matched'] and person_name:
                                # Get registered person's person_id
                                registered_person = db.get_person_by_name(person_name)
                                if registered_person:
                                    registered_person_id = registered_person['person_id']
                                    emotion_memory.emotion_memory(session_id, registered_person_id, emotion_result['emotion'], emotion_result['confidence'])
                            
                            # Get smoothed emotion for display
                            smoothed_emotion = emotion_tracker.get_smoothed(person_id)
                            display_emotion = smoothed_emotion if smoothed_emotion else emotion_result['emotion']
                            
                            emotion_text = f"Emotion: {display_emotion} ({emotion_result['confidence']:.2f})"
                            cv2.putText(frame, emotion_text, (abs_fx1, abs_fy2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            # Also display emotion above the person box for better visibility
                            emotion_label = f"{display_emotion}"
                            cv2.putText(frame, emotion_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    except Exception as e:
                        print(f"Error detecting emotion: {e}")

                    # Detect pose
                    pose_results = pose_detector.detect(frame)
                    # Draw pose on frame
                    for result in pose_results:
                        frame = result.plot()
                    # Analyze pose
                    pose_analysis = pose_analyzer.analyze(pose_results)
                    
                    # Store pose data in database for decision making
                    if pose_results:
                        import time
                        timestamp = time.time()
                        for pose_result in pose_results:
                            # Extract keypoints and bbox from pose result
                            if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
                                keypoints = pose_result.keypoints.data.cpu().numpy() if hasattr(pose_result.keypoints.data, 'cpu') else pose_result.keypoints.data
                            else:
                                keypoints = None
                            
                            if hasattr(pose_result, 'boxes') and pose_result.boxes is not None and len(pose_result.boxes.xyxy) > 0:
                                bbox = pose_result.boxes.xyxy[0].cpu().numpy() if hasattr(pose_result.boxes.xyxy[0], 'cpu') else pose_result.boxes.xyxy[0]
                                bbox = tuple(map(int, bbox))
                            else:
                                bbox = (0, 0, 0, 0)
                            
                            # Store each pose with person_id (only for matched persons)
                            if keypoints is not None and match_result['matched'] and person_name:
                                registered_person = db.get_person_by_name(person_name)
                                if registered_person:
                                    registered_person_id = registered_person['person_id']
                                    db.add_pose(session_id, registered_person_id, keypoints, bbox, timestamp)
                    
                    # Recognize activity
                    activity_result = activity_recognizer.update(pose_analysis)
                    
                    # Store activity data in database for decision making
                    if activity_result:
                        import time
                        timestamp = time.time()
                        # Store each activity with person_id (only for matched persons)
                        if match_result['matched'] and person_name:
                            registered_person = db.get_person_by_name(person_name)
                            if registered_person:
                                registered_person_id = registered_person['person_id']
                                db.add_activity(session_id, registered_person_id, activity_result['activity'], activity_result['confidence'], timestamp)

                    # Display pose analysis on frame
                    if pose_analysis and pose_analysis["poses"]:
                        y_offset = 90
                        for i, pose_info in enumerate(pose_analysis["poses"]):
                            actions = pose_info.get("actions", [])
                            if actions:
                                action_text = f"Person {i+1}: {', '.join(actions)}"
                                cv2.putText(frame, action_text, (10, y_offset), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                y_offset += 25

                    # Display recognized activity
                    if activity_result:
                        activity_text = f"Activity: {activity_result['activity']} ({activity_result['confidence']:.2f})"
                        cv2.putText(frame, activity_text, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                else:
                    # Reset match history for this track
                    face_match_history[person_id] = {}
                    label = "Unknown Person"
                    color = (0, 0, 255)  # Red for unknown
                    confidence_percent = match_result['confidence'] * 100
                    confidence_text = f"Confidence: {confidence_percent:.0f}%"
                    status_text = "Not Registered"

                
                cv2.rectangle(frame, (abs_fx1, abs_fy1), (abs_fx2, abs_fy2), color, 2)
                cv2.putText(frame, label, (abs_fx1, abs_fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, confidence_text, (abs_fx1, abs_fy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, status_text, (abs_fx1, abs_fy2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(frame, f"People: {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Person Tracker", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        saver.save(frame)

    elif key == ord("q"):
        memory.save_to_disk()
        break

camera.release()
cv2.destroyAllWindows()

# Display emotion report for the session
print("\n" + "=" * 60)
print("EMOTION REPORT FOR SESSION")
print("=" * 60)

try:
    emotions = db.get_emotions_by_session(session_id, time_window_seconds=120)
    if emotions:
        from collections import Counter
        emotion_counts = Counter([e['emotion'] for e in emotions])
        total_emotions = len(emotions)
        
        print(f"\nSession ID: {session_id}")
        print(f"Total emotion detections: {total_emotions}")
        print(f"\nEmotion distribution:")
        for emotion, count in emotion_counts.most_common():
            percentage = (count / total_emotions) * 100
            print(f"  {emotion}: {count} ({percentage:.1f}%)")
        
        # Group by person
        person_emotions = {}
        for e in emotions:
            person_id = e['person_id']
            if person_id not in person_emotions:
                person_emotions[person_id] = []
            person_emotions[person_id].append((e['emotion'], e['confidence']))
        
        print(f"\nEmotions by person:")
        for person_id, person_data in person_emotions.items():
            person_emotion_counts = Counter([e[0] for e in person_data])
            most_common = person_emotion_counts.most_common(1)[0] if person_emotion_counts else ("N/A", 0)
            avg_confidence = sum([e[1] for e in person_data]) / len(person_data)
            print(f"  Person ID {person_id}:")
            print(f"    Total detections: {len(person_data)}")
            print(f"    Most common emotion: {most_common[0]} ({most_common[1]} times)")
            print(f"    Average confidence: {avg_confidence:.3f}")
    else:
        print("No emotions recorded for this session.")
    
    print("\n" + "=" * 60)
except Exception as e:
    print(f"Error generating emotion report: {e}")
