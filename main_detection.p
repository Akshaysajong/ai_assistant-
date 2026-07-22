import time
import cv2
from camera.camera import Camera
from camera.frame_saver import FrameSaver
from detection.person_detector import PersonDetector

camera = Camera()
saver = FrameSaver()
detector = PersonDetector()

print("Press:")
print("S = Save Image")
print("Q = Quit")

prev = time.time()

while True:

    ret, frame = camera.read()

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not ret:
        break
    
    people = detector.detect(frame)
    for person in people:
        x1, y1, x2, y2 = person["bbox"]
        confidence = person["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"People: {len(people)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Person Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        saver.save(frame)

    elif key == ord("q"):
        break

camera.release()