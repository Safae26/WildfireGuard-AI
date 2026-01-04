# 🔥 WildfireGuard AI
### Autonomous Satellite Surveillance System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)

> **"Time is the only resource we cannot recover."** > WildfireGuard AI leverages Deep Learning to detect early signs of forest fires from video feeds, enabling rapid response and environmental protection.

---

## 🖥️ Interface Preview
<img width="1278" height="451" alt="image" src="https://github.com/user-attachments/assets/8809940a-359a-4caf-acb0-9a7c123e4d92" />
<img width="1280" height="553" alt="image" src="https://github.com/user-attachments/assets/ab44de0c-1358-44d4-874d-e26dc57ccdc6" />

## 🛠️ Project Overview

**WildfireGuard AI** is a real-time computer vision application designed to analyze video streams for smoke signatures and thermal anomalies indicative of forest fires. Built with **Streamlit** and **TensorFlow**, it features a responsive "Command Center" UI with simulated telemetry and live inference logging.

### Key Features
* **Real-Time Inference:** Frame-by-frame analysis using a custom-trained CNN (Convolutional Neural Network).
* **Dynamic Confidence Control:** Adjustable sensitivity threshold via the sidebar to reduce false positives.
* **Immersive UI:** Custom CSS styling with a futuristic, dark-mode aesthetic (Neon Red/Orange).
* **Live Telemetry:** Simulated dashboard metrics for wind, ping, and area coverage.
* **System Logs:** Scrolling terminal-style logs tracking detection events and timestamps.
* **Safety Mechanisms:** Automatic temporary file cleanup and user-controlled stop functionality.

## 🧰 Tech Stack

* **Core Logic:** Python 3.10
* **Frontend:** Streamlit (Custom CSS injected)
* **Computer Vision:** OpenCV (cv2)
* **Deep Learning:** TensorFlow / Keras
* **Data Manipulation:** NumPy

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Safae26/WildfireGuard-AI.git](https://github.com/Safae26/WildfireGuard-AI.git)
cd WildfireGuard-AI
```
### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
Note: Ensure you have the forest_fire.keras model file in the root directory.
### 4. Run the System
```bash
streamlit run app.py
```

## 📂 Project Structure
```bash
WildfireGuard-AI/
├── app.py                # Main application entry point (Streamlit)
├── =wildfire_detection_model.keras     # Trained Deep Learning Model (Required)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation=
```

## Model Architecture
The system utilizes a Convolutional Neural Network (CNN) trained on a dataset of satellite forest fire imagery. The model performs binary classification:
- Preprocessing: Frames are resized to (350, 350) and normalized.
- Inference: The model outputs a probability score (0.0 - 1.0).
- Thresholding: If Probability > Threshold (default 0.5), the system triggers a CRITICAL ALERT.

## Future Roadmap
- Integration with live satellite API feeds.
- GPS coordinate mapping for detected fires.
- SMS/Email alert notifications via Twilio or SMTP.
- YOLOv8 implementation for bounding-box localization.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.
Fork the Project
Create your Feature Branch (git checkout -b feature/NewFeature)
Commit your Changes (git commit -m 'Add some NewFeature')
Push to the Branch (git push origin feature/NewFeature)
Open a Pull Request

## 👤 Author
Safae Data Science & AI Student | Full-Stack Developer

**Built with ❤️ and ☕ by Safae.**
