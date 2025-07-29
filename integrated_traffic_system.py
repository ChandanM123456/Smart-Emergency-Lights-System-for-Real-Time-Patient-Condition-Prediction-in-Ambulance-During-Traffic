import cv2
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import os

# --- Configuration for YOLOv5 (Conceptual) ---
# Set the path to your trained YOLOv5 weights file (e.g., best.pt)
# You MUST replace this with the ACTUAL path to your 'best.pt' file
# For example: r"C:\Users\DELL\Desktop\mini project\yolov5\ambulance_detection_runs\custom_ambulance_model\weights\best.pt"
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
        image (numpy.ndarray): The input frame from the camera.
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
    #    pip install ultralytics # Or pip install -r yolov5/requirements.txt
    #
    # 2. Load the YOLOv5 model once (e.g., outside the loop for efficiency):
    #    import torch
    #    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    #    model.eval() # Set model to evaluation mode
    #
    # 3. Preprocess the image and run inference within this function:
    #    results = model(image) # This runs inference
    #
    # 4. Parse results to get bounding boxes and labels:
    #    detections = results.xyxy[0].cpu().numpy() # Get detections for the first image in batch
    #    is_ambulance_detected = False
    #    ambulance_bbox = None
    #    for *xyxy, conf, cls_id in detections:
    #        if model.names[int(cls_id)] == 'ambulance' and conf > 0.5: # Adjust confidence threshold
    #            x1, y1, x2, y2 = map(int, xyxy)
    #            ambulance_bbox = [x1, y1, x2 - x1, y2 - y1] # Convert to [x, y, w, h]
    #            is_ambulance_detected = True
    #            break # Assuming only one ambulance needed for logic
    #
    #    return is_ambulance_detected, ambulance_bbox
    # --- END REAL YOLOv5 INTEGRATION ---


    # --- SIMULATED DETECTION FOR DEMONSTRATION ---
    # If the YOLOv5 integration above is NOT implemented, or fails,
    # this simulation will be used.
    is_ambulance_detected = random.choice([True, False])
    ambulance_bbox = None

    if is_ambulance_detected:
        # Simulate a bounding box if an ambulance is "detected"
        height, width, _ = image.shape
        # Example: a box in the center, 20% width/height
        x = int(width * 0.4)
        y = int(height * 0.4)
        w = int(width * 0.2)
        h = int(height * 0.2)
        ambulance_bbox = [x, y, w, h]
        print(f"SIMULATED: Ambulance detected. Bounding box: {ambulance_bbox}")
    else:
        print("SIMULATED: No ambulance detected.")

    return is_ambulance_detected, ambulance_bbox


# --- Green Light Detection ---
def detect_green_light(image_roi):
    """
    Detect green light within a given image region (ROI).
    """
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

    # Initialize camera
    camera = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not camera.isOpened():
        print("Error: Unable to access the camera. Please ensure it's connected and not in use.")
        exit()

    num_roads_to_check = 4 # Number of roads/intersections to simulate checking
    time_gap_between_checks = 2 # Time in seconds between checking each road
    roads = [Road(f"Road {i + 1}") for i in range(num_roads_to_check)]
    ambulances = []

    print("\n--- Starting Traffic Monitoring Loop ---")
    for i in range(num_roads_to_check):
        print(f"\n***** Checking {roads[i].name} *****")
        ret, frame = camera.read()
        if not ret:
            print(f"Error: Unable to capture frame for {roads[i].name}. Skipping.")
            continue

        # --- Step 1: Detect Ambulance using the Model (Conceptual/Simulated) ---
        # Pass the frame and the weights path to the detection function
        has_ambulance, ambulance_bbox = detect_ambulance_using_model(frame, YOLOV5_WEIGHTS_PATH)
        roads[i].has_ambulance = has_ambulance # Update road status

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
                # For example, take a region starting 2*height above the ambulance, extending up
                roi_height_factor = 2 # How many 'h' units above the ambulance
                roi_y_start = max(0, y - int(roi_height_factor * h))
                roi_y_end = y # Or extend further up if needed
                roi_x_start = x
                roi_x_end = x + w # Same width as ambulance, or wider/narrower
                
                # Ensure ROI coordinates are within image bounds
                roi_y_start = max(0, roi_y_start)
                roi_y_end = min(frame.shape[0], roi_y_end)
                roi_x_start = max(0, roi_x_start)
                roi_x_end = min(frame.shape[1], roi_x_end)

                if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                    traffic_light_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                    roi_description = f"ROI at [({roi_x_start},{roi_y_start}) to ({roi_x_end},{roi_y_end})]"
                else:
                    print("Warning: Invalid ROI for traffic light. Using full frame.")

            print(f"Checking for green light within {roi_description}...")
            green_light_detected, green_mask_display = detect_green_light(traffic_light_roi)
            print(f"Green light detected: {green_light_detected}")

            # Add ambulance to the list for prioritization
            timestamp = int(time.time()) # Use current timestamp for ambulance arrival
            ambulances.append(Ambulance(roads[i].name, timestamp, green_light_detected))
            print(f"Ambulance recorded on {roads[i].name}. Green light status: {green_light_detected}")
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
        axs[0].set_title(f"{roads[i].name} - Ambulance: {'Detected' if has_ambulance else 'None'}")
        axs[0].axis('off')
        
        # Display the green mask (might be for ROI or full frame)
        axs[1].imshow(green_mask_display, cmap='gray')
        axs[1].set_title(f"Green Light: {'Detected' if green_light_detected else 'Not Detected'}")
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()

        # Wait for the next check
        if i < num_roads_to_check - 1:
            print(f"Waiting for {time_gap_between_checks} seconds before checking {roads[i+1].name}...")
            time.sleep(time_gap_between_checks)

    camera.release()
    print("\n--- All Roads Checked. Proceeding to Prioritization. ---")

    # --- Step 3: Give Instructions (Process road and ambulance data) ---
    while ambulances:
        freed_road = free_road(roads, ambulances)
        if freed_road:
            print(f"\nAmbulance on {freed_road} has been prioritized and passed. Resetting road status...")
            for road in roads:
                if road.name == freed_road:
                    road.has_ambulance = False
                    road.blue_light = False  # Turn off blue light once ambulance passes
            # Remove the prioritized ambulance from the queue
            ambulances = [amb for amb in ambulances if amb.road_name != freed_road]
        else:
            print("\nNo more ambulances requiring prioritization.")
            break

    # Display final pedestrian alert status for all roads
    print("\n--- Final Pedestrian Alert Status ---")
    for road in roads:
        if road.blue_light:
            print(f"Blue light is ON at {road.name} for pedestrian alert.")
        else:
            print(f"Blue light is OFF at {road.name}.")

    print("\nTraffic Signal Management System finished. Goodbye!")
