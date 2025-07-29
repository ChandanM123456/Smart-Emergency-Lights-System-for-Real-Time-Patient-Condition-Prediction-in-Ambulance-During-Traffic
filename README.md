# Smart-Emergency-Lights-System-for-Real-Time-Patient-Condition-Prediction-in-Ambulance-During-Traffic



# 🚨🚑 Smart Emergency Lights System for Real-Time Ambulance Prioritization

This project implements an advanced **Smart Emergency Lights System** designed to revolutionize traffic management during medical emergencies. By integrating real-time ambulance detection with dynamic traffic signal control and an urgency-based prioritization mechanism, the system aims to provide seamless and safe passage for ambulances, ultimately reducing critical response times.

## ✨ Core System Features & Innovations

* **Real-time Ambulance Detection:** Utilizes **YOLOv5** object detection and **OpenCV** for accurate and efficient identification of ambulances approaching intersections via camera feeds.
* **Ambulance Urgency Light System:** Ambulances are envisioned with a special light indicator that communicates the patient's condition (urgency level) to the traffic control system.
* **Dynamic Traffic Signal Prioritization:** Based on detected ambulances and their urgency levels, the system intelligently controls traffic lights to clear lanes and provide a green corridor.
* **Pedestrian Safety Blue Lights:** Traffic signals incorporate dedicated blue lights to alert pedestrians about an approaching ambulance, ensuring their safety when crossing, even if their signal is green.
* **Multi-Ambulance Prioritization:** If multiple ambulances approach from different directions, the system prioritizes based on their indicated urgency levels. In case of equal urgency, a first-come, first-serve approach is used.
* **Modular Python Implementation:** The system is developed in Python, enabling flexible integration of detection, logic, and control modules.
* **Flexible Detection Sources:** Capable of detecting ambulances from images, video files, and live webcam feeds.

## 🚀 Getting Started

Follow these steps to set up and run the ambulance detection component of the Smart Emergency Lights System on your local machine.

### Prerequisites

* **Python 3.8+:** This project is developed with Python 3.12, but other recent versions should work.
* **Git:** For cloning the YOLOv5 repository.
* **PyTorch:** The underlying deep learning library.
* **YOLOv5 Dependencies:** Other packages required by YOLOv5.
* **Webcam:** Required for live camera detection.
* **Google Drive Access:** To download the custom dataset.

### Installation

1.  **Create Your Project Directory:**
    Create a main directory for your project. Navigate into this directory in your command prompt.

    ```bash
    git clone https://github.com/ChandanM123456/Smart-Emergency-Lights-System-for-Real-Time-Patient-Condition-Prediction-in-Ambulance-During-Traffic.git
    cd Smart-Emergency-Lights-System-for-Real-Time-Patient-Condition-Prediction-in-Ambulance-During-Traffic
    ```

2.  **Clone the Official YOLOv5 Repository:**
    This step is crucial as it provides the `detect.py` script and other necessary YOLOv5 utilities.

    ```bash
    git clone [https://github.com/ultralytics/yolov5.git](https://github.com/ultralytics/yolov5.git)
    ```

3.  **Install YOLOv5 Dependencies:**
    Navigate into the cloned `yolov5` directory and install the necessary Python packages.

    ```bash
    cd yolov5
    pip install -r requirements.txt
    ```
    If you encounter issues, ensure `pip` is updated: `python -m pip install --upgrade pip`.

4.  **Download Custom Dataset and `train.py`:**
    Download the custom ambulance dataset and the `train.py` file from the following Google Drive link:
    [https://drive.google.com/file/d/1MWg--RYTXuYt4mHKkUHUjJFsP00bBrFb/view?usp=sharing](https://drive.google.com/file/d/1MWg--RYTXuYt4mHKkUHUjJFsP00bBrFb/view?usp=sharing)

    * **Place the Dataset:** After downloading and unzipping, place the dataset folder (e.g., `Ambulance.v1-ambulance-last-generate.yolov5pytorch`) directly into your main project directory (`C:\Users\YOUR_USER\Desktop\Smart_Emergency_System`).
    * **Place `train.py`:** Place the `train.py` file you downloaded from the drive into the `yolov5` folder (overwriting the existing one if prompted, or rename it if you want to keep the original). **Alternatively, if this `train.py` is meant to be a standalone script, place it in your main project directory alongside `ambulance_detector.py`.** *For this README, we'll assume you place it inside the `yolov5` folder, replacing the original, as this is typical for custom training.*

5.  **Download PyTorch (with CUDA for GPU - Recommended):**
    YOLOv5 relies on PyTorch. For significantly faster training and inference, it's highly recommended to use a GPU if available.
    * **For CPU-only:** `pip install torch torchvision torchaudio`
    * **For GPU (NVIDIA CUDA):** Visit the official PyTorch website ([pytorch.org](https://pytorch.org/get-started/locally/)) and select the appropriate command based on your CUDA version. Example: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (replace `cu118` with your CUDA version).

6.  **Place `ambulance_detector.py`:**
    Ensure the `ambulance_detector.py` script (provided in our conversation) is placed in your main project folder to detect the ambulance.
    Ensure the `image_traffic_system` script (provided in our conversation) is placed in your main project folder to detect the ambulance.
    Ensure the `integrated_traffic_system` script (provided in our conversation) is placed in your main project folder to detect the ambulance.
    
