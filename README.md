# 🛡️ Smart Motion & Face Monitor

A professional-grade Python security script that uses computer vision to detect movement and recognize human faces. Designed for both active monitoring and discreet background operation.

---

## 🚀 Key Features

* **Motion Tracking:** Uses frame differencing to highlight and track moving objects.
* **Face Detection:** Utilizes Haar Cascade classifiers to trigger specific alerts when a face is visible.
* **Headless Mode:** Run the script without a GUI for low-resource background monitoring.
* **Activity Logging:** Generates an `activity_log.txt` in headless mode to keep track of every event.
* **Auto-Storage:** Automatically sorts images into `captured_motion/` and `captured_faces/`.
* **Intelligent Throttling:** Built-in cooldown system to prevent redundant photo captures.

---

## 🛠️ Prerequisites

Before running the script, ensure you have Python installed and your camera connected.

* **Python 3.x**
* **OpenCV Library:** ```bash
    pip install opencv-python
    ```

---

## 💻 How to Use

1. **Clone the Project:**
   ```bash
   git clone [https://github.com/yourusername/smart-monitor.git](https://github.com/yourusername/smart-monitor.git)
   cd smart-monitor
   ```
