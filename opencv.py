import cv2
import time
import os
import logging
import numpy as np
import psutil

# --- 1. CONFIGURATION & SETUP ---
mode = input("Run in Headless Mode? (y/n): ").lower()
headless = True if mode == 'y' else False

folders = ["captured_motion", "captured_faces"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

logging.basicConfig(
    filename='activity_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_xml)

# Camera Setup
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- WINDOW RESIZE SETUP ---
if not headless:
    cv2.namedWindow("Smart Monitor", cv2.WINDOW_NORMAL) # Allows manual resizing
    cv2.resizeWindow("Smart Monitor", 960, 540)        # Initial "comfortable" view size

print("Initializing camera...")
time.sleep(2)

avg_background = None
last_motion_save = 0
last_face_save = 0
last_hw_check = 0
hw_check_interval = 2.0
cooldown = 3
font = cv2.FONT_HERSHEY_SIMPLEX
cpu_str, temp_str = "CPU: --%", "TEMP: --C"

def get_hw_stats():
    try:
        cpu_usage = psutil.cpu_percent()
        temp = "N/A"
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                temp = f"{entries[0].current}C"
                break
        return f"CPU: {cpu_usage}%", f"TEMP: {temp}"
    except:
        return f"CPU: {psutil.cpu_percent()}%", "TEMP: N/A"

def draw_styled_text(img, text, pos, color=(0, 255, 0)):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, 0.45, color, 1, cv2.LINE_AA)

try:
    while True:
        check, frame = video.read()
        if not check: break

        current_time = time.time()

        if current_time - last_hw_check > hw_check_interval:
            cpu_str, temp_str = get_hw_stats()
            last_hw_check = current_time

        # TIMESTAMP (on high-res frame)
        ts_text = time.strftime("%Y-%m-%d %H:%M:%S")
        ts_pos = (15, frame.shape[0] - 15)
        draw_styled_text(frame, ts_text, ts_pos, color=(0, 255, 255))

        # 2. PROCESSING
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg_background is None:
            avg_background = gray_blurred.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray_blurred, avg_background, 0.1)
        background_delta = cv2.absdiff(gray_blurred, cv2.convertScaleAbs(avg_background))
        thresh = cv2.threshold(background_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. DETECTION
        motion_this_frame = False
        motion_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 5000: continue
            motion_this_frame = True
            motion_boxes.append(cv2.boundingRect(contour))

        faces = []
        if motion_this_frame:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 4. SAVING
        file_ts = time.strftime('%H%M%S')
        if motion_this_frame and (current_time - last_motion_save > cooldown):
            cv2.imwrite(os.path.join("captured_motion", f"motion_{file_ts}.jpg"), frame)
            last_motion_save = current_time

        if len(faces) > 0 and (current_time - last_face_save > cooldown):
            cv2.imwrite(os.path.join("captured_faces", f"face_{file_ts}.jpg"), frame)
            last_face_save = current_time

        # 5. SHOW FEED
        if not headless:
            # Overlays
            stats_rows = [cpu_str, temp_str, f"RES: {frame.shape[1]}x{frame.shape[0]}"]
            for i, text in enumerate(stats_rows):
                y_pos = 25 + (i * 20)
                draw_styled_text(display_frame, text, (15, y_pos), color=(0, 255, 0))

            for (x, y, w, h) in motion_boxes:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Displaying the frame in the named window
            cv2.imshow("Smart Monitor", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        else:
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    video.release()
    cv2.destroyAllWindows()
