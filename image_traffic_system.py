import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Path to the image you want to analyze
# Make sure this path is correct and the image file exists.
INPUT_IMAGE_PATH = r"C:\Users\DELL\Pictures\Screenshots\Screenshot 2025-07-29 195109.png"# <<<--- UPDATE THIS PATH IF YOUR IMAGE IS ELSEWHERE

# Path to your trained YOLOv5 weights file (e.g., best.pt)
# YOU MUST REPLACE THIS WITH THE ACTUAL PATH TO YOUR 'best.pt' FILE.
# A likely path might be:
# r"C:\Users\DELL\Desktop\ambulance detection\runs\train\your_model_name\weights\best.pt"
# Or if your YOLOv5 repo is still in 'mini project':
# r"C:\Users\DELL\Desktop\mini project\yolov5\ambulance_detection_runs\custom_ambulance_model\weights\best.pt"
YOLOV5_WEIGHTS_PATH = r"C:\Users\DELL\Desktop\mini project\yolov5\yolov5s.pt" # <<<--- UPDATE THIS PATH

# --- Classes for Traffic Management ---
class Road:
    def __init__(self, name):
        self.name = name
        self.has_ambulance = False  # Whether an ambulance is present
        self.blue_light = False  # Blue light for pedestrian alert


class Ambulance:
    def __init__(self, road_name, timestamp, green_light):
        self.road_name = road_name
        self.timestamp = timestamp  # Time of ambulance arrival
        self.green_light = green_light  # Green light to prioritize the ambulance


# --- Model Detection Function (Placeholder for YOLOv5) ---
def detect_ambulance_using_model(image, weights_path):
    """
    Detects ambulances in the given image using a model.
    This function is a placeholder for actual YOLOv5 inference.

    Args:
        image (numpy.ndarray): The input image (frame).
        weights_path (str): Path to the YOLOv5 trained weights (e.g., best.pt).

    Returns:
        tuple: (bool) True if an ambulance is detected, (list) [x, y, w, h] bounding box
               coordinates if detected, otherwise (False, None).
    """
    print("\n--- Running Ambulance Detection (Using Model Conceptually) ---")

    # --- START REAL YOLOv5 INTEGRATION HERE ---
    # To use actual YOLOv5, you would typically:
    # 1. Ensure you have PyTorch and Ultralytics YOLOv5 installed in your environment:
    #    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    #    pip install ultralytics # Or pip install -r yolov5/requirements.txt (from yolov5 root)
    #
    # 2. Load the YOLOv5 model once (e.g., outside the loop if processing multiple images/video,
    #    but fine here if processing single image repeatedly):
    #    import torch
    #    try:
    #        # This assumes your 'weights_path' points to the 'best.pt' file
    #        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    #        model.eval() # Set model to evaluation mode
    #    except Exception as e:
    #        print(f"Error loading YOLOv5 model from {weights_path}: {e}")
    #        print("Falling back to simulated detection.")
    #        # Fallback to simulation if model loading fails
    #        is_ambulance_detected = random.choice([True, False])
    #        ambulance_bbox = None
    #        return is_ambulance_detected, ambulance_bbox
    #
    # 3. Preprocess the image and run inference within this function:
    #    results = model(image) # This runs inference on the image
    #
    # 4. Parse results to get bounding boxes and labels:
    #    detections = results.xyxy[0].cpu().numpy() # Get detections for the first image in batch
    #    is_ambulance_detected = False
    #    ambulance_bbox = None
    #    for *xyxy, conf, cls_id in detections:
    #        # Assuming 'ambulance' is class 0 (as per your data.yaml) or check model.names[int(cls_id)]
    #        if model.names[int(cls_id)] == 'ambulance' and conf > 0.5: # Adjust confidence threshold (0.5 is common)
    #            x1, y1, x2, y2 = map(int, xyxy)
    #            ambulance_bbox = [x1, y1, x2 - x1, y2 - y1] # Convert to [x, y, w, h] format
    #            is_ambulance_detected = True
    #            break # Assuming we only care about the first ambulance found for this logic
    #
    #    return is_ambulance_detected, ambulance_bbox
    # --- END REAL YOLOv5 INTEGRATION ---


    # --- SIMULATED DETECTION FOR DEMONSTRATION ---
    # REPLACE THE FOLLOWING LINES WITH YOUR ACTUAL YOLOv5 INFERENCE CODE ABOVE.
    # This simulation will be used if the YOLOv5 integration is commented out or fails.
    is_ambulance_detected = random.choice([True, False]) # Simulate detection (True for the provided image for demo)
    # To force detection for the provided image:
    # is_ambulance_detected = True 
    ambulance_bbox = None

    if is_ambulance_detected:
        # Simulate a bounding box for the ambulance in the provided image.
        # These coordinates are roughly estimated from your Screenshot 2025-07-29 195109.jpg
        height, width, _ = image.shape
        # Example coordinates for the ambulance in the provided image
        x_sim = int(width * 0.45)  # Approximately center-left
        y_sim = int(height * 0.45) # Approximately middle
        w_sim = int(width * 0.1)   # Reasonable width
        h_sim = int(height * 0.2)  # Reasonable height
        ambulance_bbox = [x_sim, y_sim, w_sim, h_sim]
        print(f"SIMULATED: Ambulance detected. Bounding box: {ambulance_bbox}")
    else:
        print("SIMULATED: No ambulance detected.")

    return is_ambulance_detected, ambulance_bbox


# --- Green Light Detection ---
def detect_green_light(image_roi):
    """
    Detect green light within a given image region (ROI).
    """
    if image_roi is None or image_roi.size == 0:
        print("Warning: Empty ROI provided for green light detection. Skipping.")
        return False, np.zeros((10,10), dtype=np.uint8) # Return empty mask

    hsv_image = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)

    # Define the green color range in HSV
    lower_green = np.array([40, 50, 50])  # Adjust as per the green light shade
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Ignore small contours
            green_detected = True
            break

    return green_detected, green_mask


# --- Road Prioritization Logic ---
def free_road(roads, ambulances):
    """Prioritize roads based on ambulance and green light detection."""
    print("\n--- Current Road Statuses ---")
    for road in roads:
        status = f"Name: {road.name}, Ambulance Present: {road.has_ambulance}, Pedestrian Alert (Blue Light): {road.blue_light}"
        print(status)

    print("\n--- Current Ambulance Queue ---")
    for amb in ambulances:
        print(f"Road: {amb.road_name}, Green Light on Arrival: {amb.green_light}, Arrival Time: {amb.timestamp}")

    if ambulances:
        # Prioritize ambulances that detected a green light first (if applicable), then by timestamp
        ambulances_with_green_light = [amb for amb in ambulances if amb.green_light]
        if ambulances_with_green_light:
            # If multiple ambulances have green light, prioritize the one that arrived first
            first_ambulance = min(ambulances_with_green_light, key=lambda amb: amb.timestamp)
            print(f"\nPrioritizing ambulance with green light on {first_ambulance.road_name}.")
        else:
            # If no ambulance has a green light, prioritize the one that arrived first (FCFS)
            first_ambulance = min(ambulances, key=lambda amb: amb.timestamp)
            print(f"\nPrioritizing ambulance based on arrival time on {first_ambulance.road_name}.")

        for road in roads:
            if road.name == first_ambulance.road_name:
                road.blue_light = True  # Activate blue light for pedestrian alert
            else:
                road.blue_light = False # Ensure other roads don't have blue light

        print(f"Action: Freeing road {first_ambulance.road_name}.")
        print(f"Action: Blue light activated on {first_ambulance.road_name} for pedestrian alert.")
        return first_ambulance.road_name

    print("No ambulances detected. Normal traffic signal operation (no special prioritization).")
    return None


# --- Main Execution ---
if __name__ == "__main__":
    print("Welcome to the Integrated Traffic Signal Management System!")

    # Load the single input image
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image not found at '{INPUT_IMAGE_PATH}'. Please check the path.")
        exit()

    frame = cv2.imread(INPUT_IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not load image from '{INPUT_IMAGE_PATH}'. Check if it's a valid image file.")
        exit()

    # We will simulate one "road" for this single image analysis
    current_road = Road("Main Intersection")
    ambulances_in_queue = [] # Renamed to avoid conflict with class name

    print(f"\n***** Analyzing Image for {current_road.name} *****")

    # --- Step 1: Detect Ambulance using the Model (Conceptual/Simulated) ---
    has_ambulance, ambulance_bbox = detect_ambulance_using_model(frame, YOLOV5_WEIGHTS_PATH)
    current_road.has_ambulance = has_ambulance # Update road status

    green_light_detected = False
    green_mask_display = np.zeros(frame.shape[:2], dtype=np.uint8) # Initialize empty mask for display

    if has_ambulance:
        # --- Step 2: If Ambulance Detected, then Detect Green Light ---
        print("Ambulance detected. Proceeding to check for green light...")

        # Define ROI (Region of Interest) for traffic light detection, typically above the ambulance
        # This is where you'd use the actual 'ambulance_bbox' from YOLOv5 if implemented.
        traffic_light_roi = frame # Default to full frame if no specific ROI logic is implemented
        roi_description = "full frame"

        if ambulance_bbox:
            x, y, w, h = ambulance_bbox
            # Conceptual ROI: A region above the detected ambulance (adjust coordinates as needed)
            # For a traffic light, it's typically a small area.
            # Let's try to target a region 2-3 times the ambulance height above it,
            # and centered horizontally over the ambulance.
            
            roi_height_factor = 2.5 # How many 'h' units above the ambulance to start looking
            roi_light_height = int(h * 0.5) # Assume traffic light is about half the ambulance height
            roi_light_width = int(w * 0.5) # Assume traffic light is about half the ambulance width

            roi_y_end = max(0, y - int(h * 0.5)) # Start just above the ambulance
            roi_y_start = max(0, roi_y_end - int(roi_height_factor * h)) # Go higher up
            
            roi_x_center = x + w // 2
            roi_x_start = max(0, roi_x_center - roi_light_width // 2)
            roi_x_end = min(frame.shape[1], roi_x_center + roi_light_width // 2)
            
            # Ensure ROI coordinates are within image bounds
            roi_y_start = max(0, roi_y_start)
            roi_y_end = min(frame.shape[0], roi_y_end)
            roi_x_start = max(0, roi_x_start)
            roi_x_end = min(frame.shape[1], roi_x_end)

            if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                traffic_light_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                roi_description = f"ROI at [({roi_x_start},{roi_y_start}) to ({roi_x_end},{roi_y_end})]"
            else:
                print("Warning: Calculated ROI for traffic light is invalid. Using full image for detection.")
                traffic_light_roi = frame # Fallback to full frame
                roi_description = "full image (due to invalid ROI calculation)"

        print(f"Checking for green light within {roi_description}...")
        green_light_detected, green_mask_display = detect_green_light(traffic_light_roi)
        print(f"Green light detected: {green_light_detected}")

        # Add ambulance to the list for prioritization
        timestamp = int(time.time()) # Use current timestamp for ambulance arrival
        ambulances_in_queue.append(Ambulance(current_road.name, timestamp, green_light_detected))
        print(f"Ambulance recorded on {current_road.name}. Green light status: {green_light_detected}")
    else:
        print("No ambulance detected. No green light check initiated for this road.")

    # --- Visualization of Current Frame and Detection ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 7)) # Increased figure size
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw mock bounding box if ambulance simulatedly detected
    if has_ambulance and ambulance_bbox:
        x, y, w, h = ambulance_bbox
        # Draw rectangle on the RGB image for visualization
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 255), 2) # Cyan rectangle for ambulance
        cv2.putText(image_rgb, "AMBULANCE (Simulated)", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    axs[0].imshow(image_rgb)
    axs[0].set_title(f"{current_road.name} - Ambulance: {'Detected' if has_ambulance else 'None'}")
    axs[0].axis('off')
    
    # Display the green mask (might be for ROI or full frame)
    axs[1].imshow(green_mask_display, cmap='gray')
    axs[1].set_title(f"Green Light: {'Detected' if green_light_detected else 'Not Detected'}")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

    print("\n--- Image Analysis Complete. Proceeding to Prioritization. ---")

    # --- Step 3: Give Instructions (Process road and ambulance data) ---
    while ambulances_in_queue:
        freed_road = free_road([current_road], ambulances_in_queue) # Pass current_road in a list for consistency
        if freed_road:
            print(f"\nAmbulance on {freed_road} has been prioritized and passed. Resetting road status...")
            current_road.has_ambulance = False
            current_road.blue_light = False  # Turn off blue light once ambulance passes
            # Remove the prioritized ambulance from the queue
            ambulances_in_queue = [amb for amb in ambulances_in_queue if amb.road_name != freed_road]
        else:
            print("\nNo more ambulances requiring prioritization.")
            break

    # Display final pedestrian alert status for all roads
    print("\n--- Final Pedestrian Alert Status ---")
    if current_road.blue_light:
        print(f"Blue light is ON at {current_road.name} for pedestrian alert.")
    else:
        print(f"Blue light is OFF at {current_road.name}.")

    print("\nTraffic Signal Management System finished. Goodbye!")
