# from ultralytics import YOLO


# class PoseDetector:

#     def __init__(self):
#         self.model = YOLO("yolo11n-pose.pt")

#     def detect(self,frame):
#         results = self.model(
#             frame,
#             verbose = False
#         )
        
#         return results


from ultralytics import YOLO

class PoseDetector:

    def __init__(self, model="yolo11n-pose.pt", conf=0.5, device=None):
        self.model = YOLO(model)
        self.conf = conf
        self.device = device

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )
        return results