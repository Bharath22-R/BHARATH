import cv2
import numpy as np
import pyttsx3
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize camera
cap = cv2.VideoCapture(0)

confidence_threshold = 0.3
distance_threshold = 1.0
detection_interval = 30  # Process every 30th frame

# Text-to-speech initialization
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Select a voice

def calculate_distance(bbox_width, frame_width):
    known_width = 0.5
    focal_length = (frame_width * distance_threshold) / known_width
    distance = (known_width * focal_length) / bbox_width
    return distance

def play_audio(message):
    engine.say(message)
    engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape
    center_region_start = int(width / 3)
    center_region_end = int(2 * width / 3)

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detected_object = None

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Check if the object is in the center region
                if center_region_start <= center_x <= center_region_end:
                    detected_object = classes[class_id]
                    distance = calculate_distance(w, width)
                    
                    # Check distance threshold
                    if distance <= distance_threshold:
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{detected_object} {round(confidence, 2)}",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
        if detected_object:
            break

    if detected_object:
        message = f"{detected_object} detected"
        print(message)
        play_audio(message)
       # time.sleep(10)  # Wait for 10 seconds before the next detection

    # Display the frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
