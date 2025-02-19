from ultralytics import YOLO
import cv2
import os

# โหลดโมเดล YOLO
model_path = r"D:\Project CV\Basketball-Game-Analyzer\runs\detect\train\weights\best.pt"
model = YOLO(model_path)  # ใช้โมเดลที่เทรนมาแล้ว

# เปิดไฟล์วิดีโอ
video_path = "videoplayback.mp4"  # เปลี่ยนเป็นพาธของไฟล์วิดีโอที่ต้องการใช้
if not os.path.exists(video_path):
    print("ไม่พบไฟล์วิดีโอที่ระบุ")
    exit()

cap = cv2.VideoCapture(video_path)  # เปิดไฟล์วิดีโอ

if not cap.isOpened():
    print("ไม่สามารถเปิดไฟล์วิดีโอได้")
    exit()

# ชื่อของคลาสที่คาดหวัง
class_names = ['Basketball', 'Basketball Hoop']

while True:
    ret, frame = cap.read()
    if not ret:
        break  # ถ้าไม่สามารถอ่านวิดีโอได้ ให้หยุด

    # ทำการตรวจจับวัตถุจากภาพในวิดีโอ
    results = model(frame)

    # วาดกรอบสีเขียวรอบๆ ลูกบาสและห่วง
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # พิกัดกรอบ (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # แปลงเป็นจำนวนเต็มและแปลงเป็น list
            conf = box.conf[0]  # ความมั่นใจ

            # ตรวจสอบว่ามีการตรวจจับลูกบาสหรือห่วงบาส
            class_id = int(box.cls[0])  # รหัสคลาส

            # ตรวจสอบว่า class_id อยู่ในขอบเขตของ class_names
            if class_id < len(class_names):
                current_class = class_names[class_id]
            else:
                current_class = "Unknown"  # หากไม่พบคลาส

            if current_class == 'Basketball' or current_class == 'Basketball Hoop':
                color = (0, 255, 0)  # กรอบสีเขียว
                # วาดกรอบรอบวัตถุ
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # แสดงชื่อของวัตถุที่ตรวจจับ
                cv2.putText(frame, current_class, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # แสดงผลลัพธ์
    cv2.imshow('Basketball Detection', frame)

    # ถ้ากด 'q' จะปิดหน้าต่าง
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดไฟล์วิดีโอและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()
