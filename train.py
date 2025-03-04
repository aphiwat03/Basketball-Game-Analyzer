from ultralytics import YOLO

# โหลดโมเดล YOLO ที่มี pretrained weights
model = YOLO("yolov8n.pt")  

# ฝึกโมเดลด้วยชุดข้อมูลของคุณ
model.train(data="data.yaml", epochs=100, imgsz=640)
