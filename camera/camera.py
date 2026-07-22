import cv2


class Camera:

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture("video/video2.mp4")

        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
