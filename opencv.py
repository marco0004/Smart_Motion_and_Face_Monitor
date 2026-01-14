import cv2
import time
import os
import logging
import numpy as np
import psutil
from threading import Thread

# --- 1. CONFIGURATION ---
CONFIG = {
    "headless": input("Run in Headless Mode? (y/n): ").lower() == 'y',
    "min_contour_area": 5000,
    "motion_cooldown": 2.0,
    "learning_rate": 0.05,
    "resolution": (640, 480),
    "folders": ["captured_motion", "captured_faces"]
}

# Setup Folders
for folder in CONFIG["folders"]:
    os.makedirs(folder, exist_ok=True)

# Logging Setup
logging.basicConfig(
    filename='activity_log.txt', level=logging.INFO,
    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 2. MULTI-THREADED CAMERA CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["resolution"][0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["resolution"][1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- 3. UTILITY FUNCTIONS ---
def get_hw_stats():
    try:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        cpu = f"CPU: {cpu_usage}%"

        temps = psutil.sensors_temperatures()
        temp_val = "N/A"
        if temps:
            for key in temps:
                if temps[key]:
                    temp_val = f"{temps[key][0].current}°C"
                    break

        return cpu, f"TEMP: {temp_val}"
    except Exception:
        return "CPU: --%", "TEMP: N/A"

def draw_ui_element(img, text, pos, color=(0, 255, 0)):
    """Draws a text element with a black background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Ensure coordinates are integers for OpenCV
    cv2.rectangle(img, (int(x - 5), int(y - h - 5)), (int(x + w + 5), int(y + 5)), (0, 0, 0), -1)
    cv2.putText(img, text, (int(x), int(y)), font, font_scale, color, thickness, cv2.LINE_AA)

# --- 4. MAIN EXECUTION ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vs = VideoStream().start()

psutil.cpu_percent(interval=None)
time.sleep(2.0)

avg_background = None
last_save = {"motion": 0, "face": 0}
last_hw_check = 0
cpu_str, temp_str = "CPU: --%", "TEMP: --C"

print(f"Monitoring started. Headless: {CONFIG['headless']}")

try:
    while True:
        frame = vs.read()
        if frame is None:
            continue

        # Get frame dimensions for dynamic UI placement
        # H is Height (Y-axis), W is Width (X-axis)
        (H, W) = frame.shape[:2]

        current_time = time.time()

        if current_time - last_hw_check > 2.0:
            cpu_str, temp_str = get_hw_stats()
            last_hw_check = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg_background is None:
            avg_background = gray_blurred.copy().astype("float")
            continue

        cv2.accumulateWeighted(gray_blurred, avg_background, CONFIG["learning_rate"])
        diff = cv2.absdiff(gray_blurred, cv2.convertScaleAbs(avg_background))
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        active_boxes = []
        for c in contours:
            if cv2.contourArea(c) > CONFIG["min_contour_area"]:
                motion_detected = True
                active_boxes.append(cv2.boundingRect(c))

        faces = []
        if motion_detected:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        file_ts = time.strftime('%Y%m%d_%H%M%S')
        readable_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_msg = "IDLE"

        # --- DYNAMIC UI POSITIONS ---
        # Calculate Y positions based on height (H)
        pos_timestamp = (20, H - 20)  # 20 pixels from bottom
        pos_status = (20, H - 45)     # 45 pixels from bottom
        pos_tag = (20, H - 70)        # 70 pixels from bottom

        if motion_detected or len(faces) > 0:
            save_frame = frame.copy()
            # Draw timestamp on saved files dynamically
            draw_ui_element(save_frame, readable_ts, pos_timestamp, (255, 255, 255))

            if motion_detected:
                status_msg = "MOTION DETECTED"
                if (current_time - last_save["motion"] > CONFIG["motion_cooldown"]):
                    cv2.imwrite(f"captured_motion/motion_{file_ts}.jpg", save_frame)
                    last_save["motion"] = current_time
                    logging.info("Motion detected and saved.")

            if len(faces) > 0:
                status_msg = "FACE DETECTED"
                if (current_time - last_save["face"] > CONFIG["motion_cooldown"]):
                    draw_ui_element(save_frame, "FACE CAPTURE", pos_tag, (0, 255, 255))
                    cv2.imwrite(f"captured_faces/face_{file_ts}.jpg", save_frame)
                    last_save["face"] = current_time
                    logging.info(f"Face detected: {len(faces)} found.")

        # Visuals (Skip if headless)
        if not CONFIG["headless"]:
            display_frame = frame.copy()
            draw_ui_element(display_frame, cpu_str, (20, 30), (0, 255, 0))
            draw_ui_element(display_frame, temp_str, (20, 55), (0, 255, 0))

            status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
            draw_ui_element(display_frame, f"STATUS: {status_msg}", pos_status, status_color)
            draw_ui_element(display_frame, readable_ts, pos_timestamp, (255, 255, 255))

            for (x, y, w, h) in active_boxes:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Smart Monitor", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nManual Exit.")
finally:
    vs.stop()
    cv2.destroyAllWindows()
