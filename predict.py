from ultralytics import YOLO
import cv2

# Load a custom model
model = YOLO(r"D:\Project CV\Basketball-Game-Analyzer\runs\detect\train\weights\best.pt")  # load a custom model

# Predict with the model
results = model("Screenshot 2025-02-19 161232.png")  # predict on an image

# Access the results
for result in results:
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
    
    # แสดงกรอบบนภาพ
    image = cv2.imread("Screenshot 2025-02-19 161232.png")  # โหลดภาพต้นฉบับ
    for box, name, conf in zip(xyxy, names, confs):
        x1, y1, x2, y2 = map(int, box)  # แปลงค่าพิกัดเป็นจำนวนเต็ม
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # วาดกรอบสีเขียว
        cv2.putText(image, f"{name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # แสดงผลลัพธ์
    cv2.imshow("Detection Result", image)
    cv2.waitKey(0)  # กดปุ่มใดๆ เพื่อปิดหน้าต่างแสดงผล

    # บันทึกภาพที่มีกรอบ
    output_path = "detected_output.png"
    cv2.imwrite(output_path, image)

# ปิดหน้าต่าง
cv2.destroyAllWindows()
