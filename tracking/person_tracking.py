from ultralytics import YOLO


class PersonTracker:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
    
    def track(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

        tracked_people = []

        for result in results:
            if result.boxes.id is None:
                continue
            
            for box, person_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])
                tracked_people.append({
                    "id": int(person_id),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })


        return tracked_people
