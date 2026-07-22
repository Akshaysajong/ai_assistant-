from ultralytics import YOLO


class PersonDetector:

    PERSON_CLASS = 0

    def __init__(self):
        self.model = YOLO("yolo11n.pt")
    
    def detect(self, frame):
        results= self.model(frame, verbose=False)

        people = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != self.PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                people.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })
        return people