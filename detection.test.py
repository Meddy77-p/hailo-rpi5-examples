import hailo
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO("yolo11n.pt")
runner = HailoRunner(model)

results = runner.predict(source=0, show=True)

print(results)
