import cv2
import math
import time
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO

# กำหนดค่าขนาด bounding box ขั้นต่ำสำหรับ person (ปรับได้ตามความเหมาะสม)
MIN_WIDTH = 50
MIN_HEIGHT = 50

# สำหรับการคำนวณค่า intensity (เฉพาะในกรณีที่มี 2 คน)
def compute_average_intensity(frame, box):
    """คำนวณค่าเฉลี่ยความสว่างใน bounding box ที่ระบุ (ใน grayscale)"""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def load_model():
    model_path = r"best8s_new.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    model = YOLO(model_path)
    model.to(device)
    return model, device

def select_video_source():
    source = input("กด 1 สำหรับ Webcam, กด 2 สำหรับคลิปวิดีโอ: ")
    if source == '1':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # สำหรับ webcam เราจะใช้การคำนวณแบบ single player
        player_count = 1
    elif source == '2':
        video_path = "2player.MOV"  # หรือเปลี่ยนเป็น video file ที่ต้องการ
        cap = cv2.VideoCapture(video_path)
        # ถ้าเลือกวีดีโอ ให้ถามว่ามีคนกี่คน (1 หรือ 2)
        while True:
            player_count = input("ระบุจำนวนผู้เล่นในวีดีโอ (1 หรือ 2): ")
            if player_count in ['1', '2']:
                player_count = int(player_count)
                break
            else:
                print("❌ กรุณากรอก 1 หรือ 2 เท่านั้น")
    else:
        print("❌ เลือกไม่ถูกต้อง ใช้ Webcam แทน")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        player_count = 1
    return cap, player_count

def point_to_line_distance(px, py, x1, y1, x2, y2):
    if (x1, y1) == (x2, y2):
        return math.hypot(px - x1, py - y1)
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denom = math.hypot(x2 - x1, y2 - y1)
    return num / denom

def main():
    cap, player_count = select_video_source()
    model, device = load_model()

    # กำหนด class names ตามที่โมเดลฝึกไว้
    class_names = ['Basketball-courts', 'ball', 'made', 'person', 'rim', 'shoot']

    ball_positions = deque(maxlen=25)
    cooldown = 0
    score1 = 0
    score2 = 0  # จะใช้เฉพาะเมื่อมี 2 คน
    meter_per_pixel = 0.05  # ปรับค่านี้ตามการสอบเทียบระยะจริง

    last_removal_time = time.time()

    # กำหนดสีสำหรับผู้เล่น
    player1_color = (0, 255, 0)   # สีเขียว สำหรับ player1
    player2_color = (255, 0, 0)   # สีน้ำเงิน สำหรับ player2 (เฉพาะกรณี 2 คน)

    # Persistent variables สำหรับเก็บข้อมูลผู้เล่น
    persistent_player1 = None  # dict {'box': (x1,y1,x2,y2), 'center': (cx,cy)}
    persistent_player2 = None  # เฉพาะกรณี 2 คน

    # สำหรับแสดงข้อความเมื่อมีการทำคะแนน
    shot_message = ""
    message_start_time = None
    message_duration = 2  # แสดงข้อความ 2 วินาที

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        
        # ปรับ conf ให้สูงขึ้นเพื่อลด false positive
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5)
        boxes = results[0].boxes

        rim_center = None
        ball_center = None
        detected_persons = []  # เก็บข้อมูล person ที่ตรวจจับได้ในเฟรมนี้

        # วนลูปตรวจจับ object
        for box in boxes:
            cls = int(box.cls[0])
            label = class_names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if label == 'ball':
                ball_center = (cx, cy)
                color = (0, 140, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            elif label == 'rim':
                rim_center = (cx, cy)
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            elif label == 'person':
                # กรอง bounding box ที่มีขนาดเล็กเกินไป
                w = x2 - x1
                h = y2 - y1
                if w < MIN_WIDTH or h < MIN_HEIGHT:
                    continue
                # สำหรับกรณี 2 คน ให้คำนวณค่า intensity เพื่อแยกสี
                if player_count == 2:
                    intensity = compute_average_intensity(frame, (x1, y1, x2, y2))
                    detected_persons.append({'box': (x1, y1, x2, y2), 'center': (cx, cy), 'intensity': intensity})
                else:
                    detected_persons.append({'box': (x1, y1, x2, y2), 'center': (cx, cy)})
            elif label == 'shoot':
                color = (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # อัปเดต persistent players ตามจำนวนที่เลือก
        if player_count == 1:
            if detected_persons:
                if persistent_player1 is None:
                    # กำหนดให้ผู้ที่อยู่ซ้ายสุดในเฟรมแรกเป็น player1
                    detected_persons_sorted = sorted(detected_persons, key=lambda p: p['center'][0])
                    persistent_player1 = detected_persons_sorted[0]
                else:
                    new_p1 = min(detected_persons, key=lambda p: math.hypot(p['center'][0] - persistent_player1['center'][0],
                                                                             p['center'][1] - persistent_player1['center'][1]))
                    persistent_player1 = new_p1
        elif player_count == 2:
            if len(detected_persons) >= 2:
                # เลือก player1 เป็นคนที่มี intensity ต่ำ (สีดำมาก)
                p1_candidate = min(detected_persons, key=lambda p: p['intensity'])
                # เลือก player2 เป็นคนที่มี intensity สูง (สีขาวมาก)
                p2_candidate = max(detected_persons, key=lambda p: p['intensity'])
                persistent_player1 = p1_candidate
                persistent_player2 = p2_candidate
            elif len(detected_persons) == 1:
                persistent_player1 = detected_persons[0]
                persistent_player2 = None

        # วาด bounding box และ label
        if player_count == 1:
            if persistent_player1 is not None:
                (x1, y1, x2, y2) = persistent_player1['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), player1_color, 2)
                cv2.putText(frame, "player1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, player1_color, 2)
        elif player_count == 2:
            if persistent_player1 is not None:
                (x1, y1, x2, y2) = persistent_player1['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), player1_color, 2)
                cv2.putText(frame, "player1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, player1_color, 2)
            if persistent_player2 is not None:
                (x1, y1, x2, y2) = persistent_player2['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), player2_color, 2)
                cv2.putText(frame, "player2", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, player2_color, 2)

        # บันทึกตำแหน่งลูกบาส
        if ball_center:
            ball_positions.append(ball_center)
        if len(ball_positions) > 1:
            pts = np.array(ball_positions, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

        # ตรวจจับการยิง: ลูกผ่าน rim จากบนลงล่าง
        if cooldown == 0 and rim_center and len(ball_positions) >= 2:
            for i in range(1, len(ball_positions)):
                bx1, by1 = ball_positions[i - 1]
                bx2, by2 = ball_positions[i]
                if by1 < by2:  # ลูกเคลื่อนลง
                    dist = point_to_line_distance(rim_center[0], rim_center[1], bx1, by1, bx2, by2)
                    if dist < 20 and by1 < rim_center[1] < by2:
                        # ใช้จุดที่ลูกบาสอยู่สูงสุด (จุดเริ่มยิง)
                        shoot_start = min(ball_positions, key=lambda p: p[1])
                        dx = rim_center[0] - shoot_start[0]
                        dy = rim_center[1] - shoot_start[1]
                        pixel_distance = math.sqrt(dx**2 + dy**2)
                        meter_distance = pixel_distance * meter_per_pixel

                        # กำหนดผู้ยิง: ถ้า player_count==1 ให้ถือว่าเป็น player1 เสมอ
                        # ถ้า player_count==2 ให้เลือกผู้ที่ใกล้ shoot_start มากกว่า
                        if player_count == 1:
                            shooter = 1
                        else:
                            if persistent_player1 is not None and persistent_player2 is not None:
                                dist_to_p1 = math.hypot(shoot_start[0] - persistent_player1['center'][0],
                                                        shoot_start[1] - persistent_player1['center'][1])
                                dist_to_p2 = math.hypot(shoot_start[0] - persistent_player2['center'][0],
                                                        shoot_start[1] - persistent_player2['center'][1])
                                shooter = 1 if dist_to_p1 < dist_to_p2 else 2
                            elif persistent_player1 is not None:
                                shooter = 1
                            else:
                                shooter = 1

                        # กำหนดคะแนน: 3 คะแนนถ้าระยะ >= 6.3 เมตร, 2 คะแนนถ้าน้อยกว่า
                        if meter_distance >= 6.3:
                            if shooter == 1:
                                score1 += 3
                                shot_message = "player2 got 3 point"
                            elif shooter == 2:
                                score2 += 3
                                shot_message = "player1 got 3 point"
                        else:
                            if shooter == 1:
                                score1 += 2
                                shot_message = "player2 got 2 point"
                            elif shooter == 2:
                                score2 += 2
                                shot_message = "player1 got 2 point"

                        message_start_time = time.time()
                        cooldown = 10
                        ball_positions.clear()
                        break

        if cooldown > 0:
            cooldown -= 1

        if rim_center:
            cv2.circle(frame, rim_center, 6, (0, 0, 255), -1)

        curr_time = time.time()
        if curr_time - last_removal_time >= 0.15 and ball_positions:
            ball_positions.popleft()
            last_removal_time = curr_time

        # แสดงคะแนน
        if player_count == 1:
            cv2.putText(frame, f"P1: {score1}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, player1_color, 3)
        else:
            cv2.putText(frame, f"P1: {score2}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, player1_color, 3)
            cv2.putText(frame, f"P2: {score1}", (frame_width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, player2_color, 3)

        # แสดงข้อความ shot message
        if message_start_time is not None and (time.time() - message_start_time) < message_duration:
            cv2.putText(frame, shot_message, (int(frame_width/2)-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        else:
            shot_message = ""
            message_start_time = None

        cv2.imshow("Basketball Tracker", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
