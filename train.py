from ultralytics import YOLO

def main():
    # โหลดโมเดล YOLOv11m ที่มี pretrained weights
    model = YOLO(r"C:\Project CV\bassket ball\yolov8m.pt")

    # เริ่มการฝึก
    model.train(
    data=r"datasets/data.yaml",
    epochs=200,
    imgsz=640,         # ลดจาก 720 → เร็วขึ้นชัดเจน
    batch=16,          # เบาลง → ไม่ overload GPU
    device=0,
    workers=6,
    cache=True,        # ถ้ามี RAM ว่าง ≥ 12GB
    patience=20,   
fliplr=0.5,         # พลิกซ้ายขวา
mosaic=0.2,         # ผสมภาพเบา ๆ
auto_augment=None,  # ปิดของหนัก
erasing=0.0,         # ปิดลบวัตถุ
    save=True,       # ✅ สำคัญที่สุดในงานจริง
    plots=True,      # 📈 แนะนำให้เปิดไว้ดูกราฟ
    val=True      # ← เปิดการตรวจสอบ mAP
    )

if __name__ == "__main__":
    main()
