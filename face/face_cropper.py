import cv2
from .face_detector import FaceDetector


class FaceCropper:

    def __init__(self, face_size=(160, 160), margin=20, detector=None,):
        self.detector = detector if detector is not None else FaceDetector()
        self.face_size = face_size
        self.margin = margin

    def crop(self, frame):
        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)

        faces = []
        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(w, x2 + self.margin)
            y2 = min(h, y2 + self.margin)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, self.face_size, interpolation=cv2.INTER_LINEAR)

            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            faces.append(
                {
                    "image": rgb_crop,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": det["confidence"]
                }
            )

        return faces



