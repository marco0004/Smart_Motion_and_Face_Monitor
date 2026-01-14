# Smart Motion & Face Monitoring System

A lightweight, multi-threaded Python application for real-time security monitoring. This system detects motion, identifies faces, logs activity, and tracks hardware performance.

## 🚀 Features

* **Multi-Threaded Video Processing**: Uses a dedicated thread for camera capture to minimize frame lag.
* **Smart Motion Detection**: Uses weighted Gaussian background subtraction for high accuracy.
* **Optimized Face Detection**: Triggers Haar Cascade classifiers only when motion is present.
* **Hardware Telemetry**: Real-time monitoring of CPU usage and system temperature.
* **Headless Mode**: Optional CLI-only mode for servers or Raspberry Pi.
* **Automated Logging**: Saves all detection events to a local text file.

## 🛠️ Setup

1. Install the required libraries using your terminal (OpenCV, NumPy, and Psutil).
2. Ensure a webcam is connected to your system.
3. Run the script using the command: `python main.py`

## 📋 Configuration

You can tune the system behavior in the CONFIG dictionary within the script:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| min_contour_area | 5000 | Minimum pixel area to trigger a motion event. |
| motion_cooldown | 3.0 | Seconds to wait between saving images. |
| learning_rate | 0.05 | Speed at which the background model updates. |
| resolution | (1280, 720) | Camera capture resolution. |

## 🏃 Operation

* **Startup**: The script will ask if you want to run in Headless Mode (y/n).
* **Monitoring**: If not headless, a window will appear showing the live feed.
* **Exit**: Press the 'q' key to quit the video window or use Ctrl+C in the terminal.
* **Storage**: Captured images are saved to the captured_motion and captured_faces folders.

## 📂 Project Structure

* **main.py**: The core application script.
* **/captured_motion**: Stores .jpg files when movement is detected.
* **/captured_faces**: Stores .jpg files when a human face is identified.
* **activity_log.txt**: Text logs of every detection event with timestamps.

## ⚙️ How It Works



1. **Background Subtraction**: The script maintains a running average of the video feed to identify the static environment.
2. **Thresholding**: It calculates the difference between the current frame and the average background to isolate movement.
3. **Contour Detection**: It draws bounding boxes around moving objects that exceed the minimum size.
4. **Resource Management**: Face detection only runs if motion is detected, which significantly reduces CPU usage.

---
*Developed for efficient local security monitoring.*
