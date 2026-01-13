import cv2
import time
import os
import logging
import numpy as np

# --- 1. CONFIGURATION & SETUP ---
# User Choice for Headless Mode
mode = input("Run in Headless Mode? (y/n): ").lower()
headless = True if mode == 'y' else False

# Folder Setup
folders = ["captured_motion", "captured_faces"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Logging Setup
if headless:
    logging.basicConfig(
        filename='activity_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    print("Headless Mode Active. Logging to 'activity_log.txt'. Press Ctrl+C to stop.")

# Load Face Model
face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_xml)

# Initialize Camera
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2) # Allow camera to warm up

# Variables for Motion/Logic
avg_background = None
last_motion_save = 0
last_face_save = 0
cooldown = 3  # Seconds between saves

print("Monitoring started...")

try:
    while True:
        check, frame = video.read()
        if not check:
            print("Failed to grab frame. Exiting...")
            break

        # Create a clean copy for processing and a display copy for drawing
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # --- 2. DYNAMIC BACKGROUND SUBTRACTION ---
        # If this is the first frame, initialize the running average
        if avg_background is None:
            avg_background = gray_blurred.copy().astype("float")
            continue

        # Smoothly update the background model to adapt to light changes
        # Higher values (e.g., 0.5) adapt faster but might "absorb" slow-moving objects
        cv2.accumulateWeighted(gray_blurred, avg_background, 0.1)
        background_delta = cv2.absdiff(gray_blurred, cv2.convertScaleAbs(avg_background))

        # Thresholding to find movement
        thresh = cv2.threshold(background_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        

        # --- 3. DETECTION LOGIC ---
        motion_this_frame = False
        motion_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 5000: # Adjust sensitivity here
                continue
            motion_this_frame = True
            motion_boxes.append(cv2.boundingRect(contour))

        faces = []
        if motion_this_frame:
            # Only run face detection if motion is detected (saves CPU)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # --- 4. DRAWING & OVERLAYS ---
        # Apply Timestamp
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp_str, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw Green Boxes for Motion
        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw Blue Boxes for Faces
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # --- 5. SAVING LOGIC ---
        current_time = time.time()
        file_ts = time.strftime('%H%M%S')

        if motion_this_frame and (current_time - last_motion_save > cooldown):
            filepath = os.path.join("captured_motion", f"motion_{file_ts}.jpg")
            cv2.imwrite(filepath, display_frame)
            last_motion_save = current_time
            if headless: logging.info(f"Motion Saved: {filepath}")

        if len(faces) > 0 and (current_time - last_face_save > cooldown):
            filepath = os.path.join("captured_faces", f"face_{file_ts}.jpg")
            cv2.imwrite(filepath, display_frame)
            last_face_save = current_time
            if headless: logging.info(f"Face Detected: {filepath}")

        # --- 6. DISPLAY / TERMINATION ---
        if not headless:
            cv2.imshow("Smart Monitor (Press 'q' to Quit)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # Short sleep to prevent 100% CPU usage in headless mode
            time.sleep(0.05)

except KeyboardInterrupt:
    print("\nInterrupted by user. Cleaning up...")

finally:
    video.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye.")
