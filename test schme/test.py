from ultralytics import YOLOv10

model = YOLOv10("test schme/yolov10m.pt")

results = model("assets/zidane.jpg", show = True)