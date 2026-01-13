import cv2
import time
import os
import logging

# 1. User Choice for Headless Mode
mode = input("Run in Headless Mode? (y/n): ").lower()
headless = True if mode == 'y' else False

# 2. Setup Folders & Logging
for folder in ["captured_motion", "captured_faces"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

if headless:
    logging.basicConfig(
        filename='activity_log.txt',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    print("Headless Mode Active. Logging to 'activity_log.txt'. Press Ctrl+C to stop.")

# 3. Load Face Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Initialize Camera
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
background = None
last_motion_save = 0
last_face_save = 0
cooldown = 2

while True:
    check, frame = video.read()
    if not check:
        break

    # Apply Timestamp
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, timestamp_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if background is None:
        background = gray_blurred
        continue

    # --- MOTION DETECTION ---
    delta = cv2.absdiff(background, gray_blurred)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_this_frame = False
    motion_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        motion_this_frame = True
        motion_contours.append(cv2.boundingRect(contour))

    # --- FACE DETECTION ---
    faces = []
    if motion_this_frame:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # --- DRAWING & SAVING ---
    save_frame = frame.copy()

    # Draw Motion (Green)
    for (x, y, w, h) in motion_contours:
        cv2.rectangle(save_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw Faces (Blue)
    for (x, y, w, h) in faces:
        cv2.rectangle(save_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save Logic
    current_time = time.time()
    if motion_this_frame and (current_time - last_motion_save > cooldown):
        filename = f"captured_motion/motion_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, save_frame)
        last_motion_save = current_time
        if headless: logging.info(f"Motion Saved: {filename}")

    if len(faces) > 0 and (current_time - last_face_save > cooldown):
        filename = f"captured_faces/face_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, save_frame)
        last_face_save = current_time
        if headless: logging.info(f"Face Saved: {filename}")

    # 5. Conditional Display
    if not headless:
        cv2.imshow("Smart Monitor", save_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        time.sleep(0.1)

video.release()
cv2.destroyAllWindows()