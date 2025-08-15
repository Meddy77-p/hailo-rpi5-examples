from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import board
import busio
from adafruit_motor import servo
import adafruit_pca9685
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# Servo Setup
# -----------------------------------------------------------------------------------------------
# Set up the I2C connection and PCA9685 board
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50  # Set the frequency for the servos

# Create servo objects for pan and tilt (using channels 0 and 1)
pan_servo = servo.Servo(pca.channels[0])
tilt_servo_1 = servo.Servo(pca.channels[1])
tilt_servo_2 = servo.Servo(pca.channels[2])
tilt_servos = [tilt_servo_1, tilt_servo_2]  # List containing both tilt servos

# Set initial positions of the servos
pan_servo.angle = 90  # Start in the middle (horizontal)
for tilt_servo in tilt_servos:  # Iterate over tilt servos
    tilt_servo.angle = 90  # Start in the middle (vertical)

# -----------------------------------------------------------------------------------------------
# Tracking Parameters (OPTIMIZED FOR SMOOTH CAMERA TRACKING)
# -----------------------------------------------------------------------------------------------
DEAD_ZONE = 40  # Dead zone to prevent jittering
SMOOTHING_FACTOR = 0.15  # Much slower/smoother movement (was 0.6)
MAX_STEP_SIZE = 2  # Very small steps per frame (was 8)
MIN_CONFIDENCE = 0.4  # Reasonable confidence threshold
DETECTION_TIMEOUT = 3.0  # Time before giving up on lost target
PAN_SENSITIVITY = 30  # Reduced sensitivity for slower movement (was 120)
TILT_SENSITIVITY = 25  # Reduced sensitivity for slower movement (was 90)
MOVEMENT_DELAY = 0.05  # Small delay between movements to allow camera to settle

# Variables for smoothing
last_pan_angle = 90.0
last_tilt_angle = 90.0
last_valid_detection = time.time()
frame_width = 640  # Default frame width
frame_height = 480  # Default frame height

def smooth_angle_update(current_angle, target_angle, smoothing_factor, max_step):
    """Apply smoothing and step limiting to servo movements"""
    # Calculate the difference
    diff = target_angle - current_angle
    
    # Apply smoothing
    smoothed_diff = diff * smoothing_factor
    
    # Limit the step size
    if abs(smoothed_diff) > max_step:
        smoothed_diff = max_step if smoothed_diff > 0 else -max_step
    
    return current_angle + smoothed_diff

def track_person(bbox, frame_width, frame_height):
    """Track a person using servo control based on bounding box"""
    global last_pan_angle, last_tilt_angle, last_valid_detection
    
    # Get the center of the bounding box
    center_x = bbox.xmin() + (bbox.width() / 2)
    center_y = bbox.ymin() + (bbox.height() / 2)
    
    # Convert relative coordinates to pixel coordinates
    center_x_px = int(center_x * frame_width)
    center_y_px = int(center_y * frame_height)
    
    # Calculate center of frame
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # Calculate error (distance from center)
    error_x = center_x_px - frame_center_x
    error_y = center_y_px - frame_center_y
    
    # Only move if outside dead zone to prevent jittering
    if abs(error_x) > DEAD_ZONE or abs(error_y) > DEAD_ZONE:
        
        # Calculate target angles based on error with reduced sensitivity
        pan_adjustment = (error_x / frame_width) * PAN_SENSITIVITY
        tilt_adjustment = (error_y / frame_height) * TILT_SENSITIVITY
        
        # Calculate target angles
        target_pan = last_pan_angle + pan_adjustment
        target_tilt = last_tilt_angle + tilt_adjustment
        
        # Ensure angles are within valid servo range
        target_pan = max(0, min(180, target_pan))
        target_tilt = max(0, min(180, target_tilt))
        
        # Apply smoothing for very gradual movement
        new_pan = smooth_angle_update(last_pan_angle, target_pan, SMOOTHING_FACTOR, MAX_STEP_SIZE)
        new_tilt = smooth_angle_update(last_tilt_angle, target_tilt, SMOOTHING_FACTOR, MAX_STEP_SIZE)
        
        # Only update if there's a meaningful change (reduces servo jitter)
        if abs(new_pan - last_pan_angle) > 0.5 or abs(new_tilt - last_tilt_angle) > 0.5:
            try:
                # Update pan servo
                pan_servo.angle = new_pan
                
                # Update both tilt servos
                for tilt_servo in tilt_servos:
                    tilt_servo.angle = new_tilt
                
                # Small delay to allow camera to settle
                time.sleep(MOVEMENT_DELAY)
                
                # Store the new angles
                last_pan_angle = new_pan
                last_tilt_angle = new_tilt
                
                # Print the tracking information
                print(f"TRACKING: Person Center: ({center_x_px}, {center_y_px}) | Error: ({error_x:+4.0f}, {error_y:+4.0f}) | Pan: {new_pan:.1f} | Tilt: {new_tilt:.1f}")
            except Exception as e:
                print(f"Servo control error: {e}")
        else:
            print(f"TRACKING: Minor adjustment - no movement needed")
    else:
        print(f"TRACKING: Person centered (within dead zone)")
    
    # Update last valid detection time
    last_valid_detection = time.time()

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.tracked_person_id = None  # Track specific person ID
        self.person_priority = {}  # Track person priorities based on size/confidence
        self.detection_count = 0  # Add detection counter
    
    def new_function(self):  # New function example
        return "Tracking Person: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    global frame_width, frame_height, last_valid_detection
    
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    # Using the user_data to count the number of frames
    user_data.increment()
    frame_count = user_data.get_count()
    
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    
    # Update frame dimensions
    if width is not None and height is not None:
        frame_width = width
        frame_height = height
    
    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Parse the detections - only look for people
    person_detections = []
    detection_count = 0
    
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person" and confidence >= MIN_CONFIDENCE:
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            
            person_detections.append({
                'detection': detection,
                'bbox': bbox,
                'confidence': confidence,
                'track_id': track_id,
                'area': bbox.width() * bbox.height()  # Calculate area for priority
            })
            
            detection_count += 1
    
    # Update detection count
    user_data.detection_count = detection_count
    
    # Select person to track (prioritize by area - largest person)
    if person_detections:
        # Sort by area (largest first)
        person_detections.sort(key=lambda x: x['area'], reverse=True)
        
        # SIMPLIFIED: Always track the largest person for now
        selected_person = person_detections[0]
        user_data.tracked_person_id = selected_person['track_id']
        
        # Track the selected person
        track_person(selected_person['bbox'], frame_width, frame_height)
        
        # Print detailed tracking info
        bbox = selected_person['bbox']
        print(f"Frame {frame_count}: Tracking person ID {selected_person['track_id']} | "
              f"Conf: {selected_person['confidence']:.2f} | "
              f"BBox: ({bbox.xmin():.3f}, {bbox.ymin():.3f}, {bbox.xmax():.3f}, {bbox.ymax():.3f}) | "
              f"Area: {selected_person['area']:.3f}")
        
        # Draw tracking info on frame if available
        if user_data.use_frame and frame is not None:
            # Draw bounding box around tracked person
            x1 = int(bbox.xmin() * frame_width)
            y1 = int(bbox.ymin() * frame_height)
            x2 = int(bbox.xmax() * frame_width)
            y2 = int(bbox.ymax() * frame_height)
            
            # Draw green rectangle around tracked person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
            
            # Draw track ID
            cv2.putText(frame, f"Tracking ID: {selected_person['track_id']}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    else:
        # No person detected
        if (time.time() - last_valid_detection) > DETECTION_TIMEOUT:
            print(f"Frame {frame_count}: No person detected for {DETECTION_TIMEOUT}s")
            user_data.tracked_person_id = None
        else:
            print(f"Frame {frame_count}: No person detected (waiting...)")
    
    if user_data.use_frame and frame is not None:
        # Draw crosshair at center
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        cv2.line(frame, (frame_center_x - 20, frame_center_y), (frame_center_x + 20, frame_center_y), (0, 255, 255), 2)
        cv2.line(frame, (frame_center_x, frame_center_y - 20), (frame_center_x, frame_center_y + 20), (0, 255, 255), 2)
        
        # Draw dead zone
        cv2.rectangle(frame, 
                     (frame_center_x - DEAD_ZONE, frame_center_y - DEAD_ZONE),
                     (frame_center_x + DEAD_ZONE, frame_center_y + DEAD_ZONE),
                     (0, 100, 100), 1)
        
        # Print the detection count to the frame
        cv2.putText(frame, f"People Detected: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Example of how to use the new_variable and new_function from the user_data
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show servo angles
        cv2.putText(frame, f"Pan: {last_pan_angle:.1f} Tilt: {last_tilt_angle:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    
    print("Starting person tracking system...")
    print("Press Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nStopping person tracking system...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Reset servos to center position
        try:
            pan_servo.angle = 90
            for tilt_servo in tilt_servos:  # Iterate over tilt servos
                tilt_servo.angle = 90
            print("Servos reset to center position")
        except Exception as e:
            print(f"Error resetting servos: {e}")
