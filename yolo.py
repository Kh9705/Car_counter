from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
import numpy as np
from sort import *  # SORT tracker for object tracking

# Check device (CUDA if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 Model
model = YOLO('yolov8l.pt')  # Ensure the model file is in the correct location

# Load Mask Image
mask = cv2.imread("Images/mask-950x480.png")

# Open Video File
cap = cv2.VideoCapture("Videos/cars.mp4")

# Initialize SORT Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# List of class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Define line limits for counting vehicles
limits = [400, 297, 673, 297]  # Line coordinates (x1, y1, x2, y2)
totalCount = []  # Stores unique object IDs that have crossed the line

# Video Processing Loop
while True:
    success, img = cap.read()

    if not success:
        print("End of video or failed to read frame.")
        break  # Stop the loop if video ends or error occurs

    # Apply Mask if Available
    if mask is not None:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Resize mask to match frame
        imgRegion = cv2.bitwise_and(img, mask)  # Apply mask using bitwise AND
    else:
        imgRegion = img  # Use original image if mask is missing

    imgGraphics = cv2.imread("Images/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    # YOLO Detection (Stream Mode for Faster Processing)
    results = model(img, stream=True)
    detections = np.empty((0, 5))  # Initialize empty detections array

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box Extraction
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int
            w, h = x2 - x1, y2 - y1  # Calculate width & height

            # Confidence Score
            conf = math.ceil(box.conf[0].item() * 100) / 100  # Convert tensor to float
            print(f"Detected: {conf}")

            # Class Name Extraction
            cls = int(box.cls.item())  # Convert tensor to int
            currentClass = classNames[cls]

            # Filter for Specific Classes
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
               # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                #                   scale=0.6, thickness=1, offset=3)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))  # Add detection to array

    # Update tracker with detections
    resultsTracker = tracker.update(detections)

    # Draw tracking line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        # Draw bounding box and object ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Calculate center of object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if object crosses the defined counting line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # Change line color

    # Display Count on Screen
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    
    # Show Video Frames
    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Video & Close Windows
cap.release()
cv2.destroyAllWindows()
