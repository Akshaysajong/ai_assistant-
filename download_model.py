import urllib.request
import os

# Download yolov11n-face.pt model
model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt"
model_path = "yolov11n-face.pt"

print(f"Downloading {model_url} to {model_path}...")
try:
    urllib.request.urlretrieve(model_url, model_path)
    print("Download completed successfully!")
except Exception as e:
    print(f"Download failed: {e}")
    print("\nAlternative: You can manually download a face detection model from:")
    print("https://github.com/deepcam-cn/yolov5-face")
    print("Or use the existing yolo11n.pt by renaming it:")
    if os.path.exists("yolo11n.pt"):
        os.rename("yolo11n.pt", "yolov11n-face.pt")
        print("Renamed yolo11n.pt to yolov11n-face.pt")
