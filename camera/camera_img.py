import os
import cv2


# class Camera:

#     def __init__(self, camera_index=0):
#         self.cap = cv2.imread(r"C:\Users\sajon\Downloads\akshay.jpeg")

#         if self.cap is None:
#             raise Exception("Cannot open image")

#     def read(self):
#         return self.cap

#     def release(self):
#         pass

class ImageLoader:

    def __init__(self, folder):
        self.folder = folder
        self.images = [
            os.path.join(self.folder, file)
            for file in os.listdir(self.folder)
            if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg') or file.lower().endswith('.png')
        ]
        
        if not self.images:
            raise Exception("No images found in the specified folder")
        
        self.index = 0

    def read(self):

        if self.index >= len(self.images):
            return False, None

        image = cv2.imread(self.images[self.index])
        self.index += 1
        return True, image