from ultralytics import YOLO


class FaceDetector:
    def __init__(self, conf=0.45, iou=0.5, device=None,):
        self.model = YOLO("yolov11n-face.pt")
        self.conf = conf  # Lower confidence threshold for better detection of profile faces
        self.iou = iou
        self.device = device

    def detect(self, frame):
        """
        Returns:
        [
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.98
            },
            ...
        ]
        """

        h, w = frame.shape[:2]

        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False,)
        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(box.conf.item())
                    }
                )

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        return detections