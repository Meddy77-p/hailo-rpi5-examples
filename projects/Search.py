import time
import board
import busio
from adafruit_motor import servo
import adafruit_pca9685
import cv2
import numpy as np
import subprocess
import json
import os

# === Servo setup (unchanged from your working code) ===
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50

pan_servo = servo.Servo(pca.channels[0])
tilt_servo = servo.Servo(pca.channels[1])
pan_servo.angle = 90
tilt_servo.angle = 90

# === Camera setup ===
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2
DEAD_ZONE = 10  # px
SENSITIVITY = 0.05  # lower = smoother

TARGET_CLASS = "person"  # can be "bottle", "car", etc.

# === Function to run inference using Hailo model ===
def run_hailo_inference(frame):
    temp_img = "frame.jpg"
    output_json = "output.json"

    # Save frame to disk
    cv2.imwrite(temp_img, frame)

    # Run the model using hailo_model_runner
    result = subprocess.run([
        "hailo_model_runner", "yolov8.hailom",
        "--input", temp_img, "--output", output_json
    ], capture_output=True)

    if result.returncode != 0:
        print("Hailo inference failed:", result.stderr.decode())
        return []

    if not os.path.exists(output_json):
        print("No output.json found.")
        return []

    with open(output_json, "r") as f:
        try:
            data = json.load(f)
            return data.get("objects", [])
        except json.JSONDecodeError:
            print("Invalid JSON in output")
            return []

# === Main tracking function ===
def track_object():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        detections = run_hailo_inference(frame)

        target_found = False
        for obj in detections:
            label = obj["label"]
            confidence = obj["confidence"]
            bbox = obj["bbox"]  # [x, y, w, h]

            if label.lower() == TARGET_CLASS and confidence > 0.5:
                x, y, w, h = bbox
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)

                # Draw detection on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Map position to servo angles
                pan_angle = np.interp(FRAME_WIDTH - center_x, [0, FRAME_WIDTH], [0, 180])  # reverse pan
                tilt_angle = np.interp(center_y, [0, FRAME_HEIGHT], [0, 180])

                # Dead zone
                if abs(pan_angle - pan_servo.angle) < DEAD_ZONE:
                    pan_angle = pan_servo.angle
                if abs(tilt_angle - tilt_servo.angle) < DEAD_ZONE:
                    tilt_angle = tilt_servo.angle

                # Clamp and apply angles
                pan_angle = max(0, min(180, pan_angle))
                tilt_angle = max(0, min(180, tilt_angle))
                pan_servo.angle = pan_angle
                tilt_servo.angle = tilt_angle

                print(f"Tracking: {label} | Pan: {pan_angle:.2f} | Tilt: {tilt_angle:.2f}")
                target_found = True
                break  # Only track the first matching object

        if not target_found:
            print("No target detected. Searching...")

        # Display
        frame = cv2.flip(frame, 1)
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the object tracker
track_object()
