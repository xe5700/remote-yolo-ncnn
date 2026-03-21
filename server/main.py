
from ultralytics import YOLO
model = YOLO("models/yolo26n_ncnn_model_384", task="detect")
out=model.predict("models/bus.jpg", save=False)
for item in out:
    print(item)
    print(item.boxes)