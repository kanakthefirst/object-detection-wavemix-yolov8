from ultralytics import YOLO
model = YOLO(model='ultralytics/cfg/models/v8/yolov8.yaml', task='detect')
result = model.train(data='VOC.yaml', epochs=2, imgsz=640)