import os
import torch
from pathlib import Path
import subprocess
import shutil

# --- Configuration ---
# IMPORTANT: This MUST match the exact path where you unzipped your dataset
DATASET_PATH = r"C:\Users\DELL\Desktop\mini project\Ambulance.v1-ambulance-last-generate.yolov5pytorch"
DATA_YAML_PATH = os.path.join(DATASET_PATH, "data.yaml")

# IMPORTANT: Name of your YOLOv5 repository folder within 'mini project'
# This is typically 'yolov5' if you cloned it.
YOLOV5_REPO_NAME = "yolov5" # <<<--- VERIFY THIS FOLDER NAME

# Check if the data.yaml exists
if not os.path.exists(DATA_YAML_PATH):
    print(f"Error: data.yaml not found at {DATA_YAML_PATH}.")
    print("Please ensure your dataset is unzipped and DATASET_PATH is correctly set in this script.")
    print("Also verify that 'data.yaml' exists directly in that folder and is named exactly 'data.yaml'.")
    exit()

# Model parameters
MODEL_NAME = "yolov5s.pt"  # yolov5s.pt (small), yolov5m.pt (medium), yolov5l.pt (large)
BATCH_SIZE = 16            # Adjust based on your GPU memory. Lower if you get CUDA out of memory.
EPOCHS = 50                # Number of training epochs. Start with 50-100, then adjust.
IMG_SIZE = 640             # Input image size for the model
PROJECT_NAME = "ambulance_detection_runs" # Folder for training results
RUN_NAME = "custom_ambulance_model"      # Specific name for this training run

# --- Training Function ---
def train_model():
    """Trains a YOLOv5 model for ambulance detection."""
    print(f"\n--- Starting YOLOv5 Training for {RUN_NAME} ---")
    print(f"Dataset YAML: {DATA_YAML_PATH}")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Image Size: {IMG_SIZE}")

    # CORRECTED: Point to the train.py inside the YOLOv5 repository folder
    train_script_path = os.path.join(os.getcwd(), YOLOV5_REPO_NAME, 'train.py')

    if not os.path.exists(train_script_path):
        print(f"Error: train.py not found at expected path: {train_script_path}")
        print(f"Please ensure '{YOLOV5_REPO_NAME}' is the correct name for your YOLOv5 directory within '{os.getcwd()}'")
        print("And that train.py is directly inside that YOLOv5 directory.")
        return None

    # MODIFIED: Explicitly use 'py -3.12' to ensure correct Python version
    command = [
        "py", "-3.12", train_script_path,
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", DATA_YAML_PATH,
        "--weights", MODEL_NAME,
        "--project", PROJECT_NAME,
        "--name", RUN_NAME,
        "--exist-ok"
    ]

    try:
        print("\nExecuting training command:")
        print(" ".join(command))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Stdout:\n", result.stdout)
        if result.stderr:
            print("Stderr:\n", result.stderr)
        print("\n--- Training Complete! ---")
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        print(f"Command failed with exit code {e.returncode}")
        print(f"Stderr (if captured): {e.stderr}")
        print(f"Stdout (if captured): {e.stdout}")
        return None
    except FileNotFoundError:
        print(f"\nError: 'py' or '{train_script_path}' command not found.")
        print("Please ensure Python 3.12 is installed and 'py' launcher is in your PATH, and you are running this script from the parent directory of your YOLOv5 folder.")
        return None
    
    trained_weights_path = Path(os.getcwd()) / PROJECT_NAME / RUN_NAME / 'weights' / 'best.pt'
    if trained_weights_path.exists():
        print(f"\nTrained weights saved to: {trained_weights_path}")
        return str(trained_weights_path)
    else:
        print(f"\nError: Trained weights not found at {trained_weights_path}. Training might have failed.")
        return None

# --- Detection Function ---
def detect_ambulances(weights_path, source_path, confidence_threshold=0.25):
    """
    Performs object detection using the trained YOLOv5 model.
    :param weights_path: Path to the trained .pt weights file (e.g., 'runs/train/custom_ambulance_model/weights/best.pt')
    :param source_path: Path to the image file, video file, or '0' for webcam.
    :param confidence_threshold: Minimum confidence to display a detection.
    """
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights not found at {weights_path}.")
        print("Please train the model first or provide a valid path to pre-trained weights.")
        return

    print(f"\n--- Starting YOLOv5 Detection ---")
    print(f"Using Weights: {weights_path}")
    print(f"Source: {source_path}")
    print(f"Confidence Threshold: {confidence_threshold}")

    # CORRECTED: Point to the detect.py inside the YOLOv5 repository folder
    detect_script_path = os.path.join(os.getcwd(), YOLOV5_REPO_NAME, 'detect.py')

    if not os.path.exists(detect_script_path):
        print(f"Error: detect.py not found at expected path: {detect_script_path}")
        print(f"Please ensure '{YOLOV5_REPO_NAME}' is the correct name for your YOLOv5 directory within '{os.getcwd()}'")
        print("And that detect.py is directly inside that YOLOv5 directory.")
        return

    # MODIFIED: Explicitly use 'py -3.12' to ensure correct Python version
    command = [
        "py", "-3.12", detect_script_path,
        "--weights", weights_path,
        "--source", source_path,
        "--conf", str(confidence_threshold),
        "--project", "detect_runs",
        "--name", "ambulance_detections",
        "--exist-ok"
    ]

    try:
        print("\nExecuting detection command:")
        print(" ".join(command))
        subprocess.run(command, check=True)
        print("\n--- Detection Complete! Results saved to detect_runs/ambulance_detections ---")
    except subprocess.CalledProcessError as e:
        print(f"\nError during detection: {e}")
        print(f"Command failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"\nError: 'py' or '{detect_script_path}' command not found.")
        print("Please ensure Python 3.12 is installed and 'py' launcher is in your PATH, and you are running this script from the parent directory of your YOLOv5 folder.")

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Welcome to the YOLOv5 Ambulance Detector!")
    print(f"Ensure your dataset is at: {DATASET_PATH}")
    # Clarified instruction:
    print(f"Ensure you are running this script from the directory that CONTAINS your YOLOv5 folder (e.g., '{os.getcwd()}')")
    print(f"And that your YOLOv5 folder is named '{YOLOV5_REPO_NAME}' and contains 'train.py' and 'detect.py'.")

    # 1. Train the model (or use existing)
    train_choice = input("Do you want to train the model? (y/n): ").lower()
    if train_choice == 'y':
        trained_model_path = train_model()
    else:
        # If not training, assume you have existing weights at the standard output path
        # Verify this path on your system!
        trained_model_path = Path(os.getcwd()) / PROJECT_NAME / RUN_NAME / 'weights' / 'best.pt'
        if not trained_model_path.exists():
            print(f"Warning: No existing trained model found at {trained_model_path}.")
            print("Please run training first (choose 'y' when prompted) or provide a correct path to your 'best.pt' file if it's in a non-standard location.")
            trained_model_path = None # Set to None if no existing model is found

    if trained_model_path:
        print(f"\nUsing trained model from: {trained_model_path}")
        
        # --- Choose your detection source ---
        # The line below is already set to use your webcam by default.
        # Option 1: Live Webcam Feed (requires webcam)
        TEST_SOURCE = "0" 

        # Option 2: An Image File (replace with your actual image path)
        # TEST_SOURCE = r"C:\Users\DELL\Desktop\test_images\my_ambulance_photo.jpg" 

        # Option 3: A Video File (replace with your actual video path)
        # TEST_SOURCE = r"C:\Users\DELL\Desktop\test_videos\traffic_cam.mp4"

        # Option 4: A Folder Containing Images/Videos
        # TEST_SOURCE = r"C:\Users\DELL\Desktop\my_media_folder_for_detection"
        
        print(f"\nReady for Detection. Selected TEST_SOURCE: {TEST_SOURCE}")
        if TEST_SOURCE != "0" and not os.path.exists(TEST_SOURCE):
            print(f"Warning: Test source '{TEST_SOURCE}' not found. Please provide a valid path or use '0' for webcam.")
            input("Press Enter to continue without detection or Ctrl+C to exit and correct the path...")
        else:
            detect_ambulances(str(trained_model_path), TEST_SOURCE)
    else:
        print("\nSkipping detection because no trained model weights were available.")

    print("\nAmbulance Detector script finished.")
