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
    "min_contour_area": 5000,      # Minimum size of movement to trigger
    "motion_cooldown": 3.0,        # Seconds between saves
    "learning_rate": 0.05,         # How fast the background "absorbs" changes
    "resolution": (1280, 720),
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
    """Retrieves CPU and Temp data with error handling."""
    try:
        cpu = f"CPU: {psutil.cpu_percent()}%"
        temps = psutil.sensors_temperatures()
        # Search for common temperature keys
        temp_val = "N/A"
        if temps:
            for key in ['coretemp', 'cpu_thermal', 'acpitz']:
                if key in temps:
                    temp_val = f"{temps[key][0].current}°C"
                    break
        return cpu, f"TEMP: {temp_val}"
    except Exception:
        return "CPU: --%", "TEMP: N/A"

def draw_overlay(img, text, pos, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, 0.45, color, 1, cv2.LINE_AA)

# --- 4. MAIN EXECUTION ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vs = VideoStream().start()
time.sleep(2.0) # Warm up

avg_background = None
last_save = {"motion": 0, "face": 0}
last_hw_check = 0
cpu_str, temp_str = "CPU: --%", "TEMP: --C"

print(f"Monitoring started. Headless: {CONFIG['headless']}")

try:
    while True:
        frame = vs.read()
        if frame is None: break

        current_time = time.time()
        display_frame = frame.copy() if not CONFIG["headless"] else None
        
        # HW Stats Update (Every 2 seconds)
        if current_time - last_hw_check > 2.0:
            cpu_str, temp_str = get_hw_stats()
            last_hw_check = current_time

        # Image Processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        if avg_background is None:
            avg_background = gray_blurred.copy().astype("float")
            continue

        # Motion Detection Logic
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

        # Face Detection (Only if motion exists to save CPU)
        faces = []
        if motion_detected:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # File Saving Logic
        file_ts = time.strftime('%Y%m%d_%H%M%S')
        if motion_detected and (current_time - last_save["motion"] > CONFIG["motion_cooldown"]):
            cv2.imwrite(f"captured_motion/motion_{file_ts}.jpg", frame)
            last_save["motion"] = current_time
            logging.info("Motion detected and saved.")

        if len(faces) > 0 and (current_time - last_save["face"] > CONFIG["motion_cooldown"]):
            cv2.imwrite(f"captured_faces/face_{file_ts}.jpg", frame)
            last_save["face"] = current_time
            logging.info(f"Face detected: {len(faces)} found.")

        # Visuals (Skip if headless)
        if not CONFIG["headless"]:
            # Draw UI
            draw_overlay(display_frame, time.strftime("%Y-%m-%d %H:%M:%S"), (15, 700), (0, 255, 255))
            draw_overlay(display_frame, cpu_str, (15, 30), (0, 255, 0))
            draw_overlay(display_frame, temp_str, (15, 50), (0, 255, 0))
            
            # Draw Detectors
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
