import cv2
import numpy as np
import os
import time
from inference_sdk import InferenceHTTPClient

# --- Roboflow API Configuration for Emergency Vehicles ---
# The API key is now retrieved securely from Vercel's environment variables.
API_KEY = os.environ.get("API_KEY") 
MODEL_ID = "emergency-vehicle-detection-t2tck/1"

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# --- Local Model for General Vehicle Counting (using ONNX) ---
# This lightweight approach replaces the large ultralytics/PyTorch dependency.
ONNX_MODEL_PATH = "yolov8n.onnx"
net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)

# Get the names of the output layers
output_layer_names = net.getUnconnectedOutLayersNames()

# YOLOv8 class names for the COCO dataset
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ACCIDENT_CLASS = "accident"

def analyze_traffic_scene(filepath):
    """
    Analyzes a traffic scene using a robust two-step process:
    1. A local ONNX model counts all general vehicles for traffic flow.
    2. The Roboflow API is called with a specialist model for emergency vehicle detection.
    """
    frame = cv2.imread(filepath)
    if frame is None:
        raise IOError(f"Cannot open or read file: {filepath}")

    h, w, _ = frame.shape
    
    # --- Step 1: General Vehicle Counting (using local ONNX model) ---
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_outputs = net.forward(output_layer_names)
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Process outputs
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter for vehicle classes
            if confidence > 0.5 and class_names[class_id] in VEHICLE_CLASSES:
                center_x, center_y, box_w, box_h = detection[0:4] * np.array([w, h, w, h])
                x = int(center_x - box_w / 2)
                y = int(center_y - box_h / 2)
                boxes.append([x, y, int(box_w), int(box_h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    vehicle_count = 0
    vehicle_distribution = {cls: 0 for cls in VEHICLE_CLASSES}
    
    annotated_frame = frame.copy()
    
    if len(indices) > 0:
        for i in indices.flatten():
            class_name = class_names[class_ids[i]]
            vehicle_distribution[class_name] += 1
            vehicle_count += 1
            
            x, y, box_w, box_h = boxes[i]
            cv2.rectangle(annotated_frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
            label = f"{class_name}: {confidences[i]:.2f}"
            cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Step 2: Emergency Vehicle and Accident Detection (using Roboflow) ---
    has_emergency = False
    has_accident = False

    try:
        result = CLIENT.infer(filepath, model_id=MODEL_ID)
        
        for pred in result['predictions']:
            if pred['confidence'] < 0.40:
                continue
            
            class_name = pred['class'].lower().replace('fire_truck', 'fire truck')
            
            if class_name in ['ambulance', 'fire truck']:
                has_emergency = True
                
                # Draw the specialist's bounding box in a different color (RED)
                x0 = int(pred['x'] - pred['width'] / 2)
                y0 = int(pred['y'] - pred['height'] / 2)
                x1 = int(pred['x'] + pred['width'] / 2)
                y1 = int(pred['y'] + pred['height'] / 2)
                cv2.rectangle(annotated_frame, (x0, y0), (x1, y1), (0, 0, 255), 3)
                label = f"EMERGENCY: {pred['class']}"
                cv2.putText(annotated_frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            elif class_name == ACCIDENT_CLASS:
                has_accident = True
                
    except Exception as e:
        print(f"Error calling Roboflow API: {e}. Emergency detection will be skipped.")
        has_emergency = False

    # --- Save the Final Annotated Image ---
    timestamp = int(time.time())
    output_filename = f"detection_{timestamp}.jpg"
    output_path = os.path.join('static', 'detections', output_filename)
    cv2.imwrite(output_path, annotated_frame)
    
    return {
        'vehicle_count': vehicle_count,
        'vehicle_distribution': vehicle_distribution,
        'has_emergency': has_emergency,
        'has_accident': has_accident,
        'annotated_image_path': output_path
    }