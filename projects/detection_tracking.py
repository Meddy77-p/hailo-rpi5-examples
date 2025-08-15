import time
import cv2
import numpy as np
import board
import busio
from adafruit_motor import servo
import adafruit_pca9685
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# Set up the I2C connection and PCA9685 board
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50  # Set the frequency for the servos

# Create servo objects for pan and tilt (using channels 0 and 1)
pan_servo = servo.Servo(pca.channels[0])
tilt_servo = servo.Servo(pca.channels[1])

# Set initial positions of the servos
pan_servo.angle = 90  # Start in the middle (horizontal)
tilt_servo.angle = 90  # Start in the middle (vertical)

# Define constants
DEAD_ZONE = 10  # Pixels around the center to prevent unnecessary movement
SLOW_MOVE_STEP = 1  # How much the servo moves per step (degrees)
MOVE_DELAY = 0.05  # Delay between each move step (for smoothness)

# -----------------------------------------------------------------------------------------------
# User-defined callback class to be used in the detection pipeline
# -----------------------------------------------------------------------------------------------
class UserAppCallbackClass(app_callback_class):
    def __init__(self):
        super().__init__()
        self.detection_count = 0

    def increment(self):
        self.detection_count += 1

    def get_count(self):
        return self.detection_count

    def app_callback(self, pad, info, user_data):
        # Get the GstBuffer from the probe info
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        
        # Get frame details and process detection
        frame = get_numpy_from_buffer(buffer)
        # Get the detected objects (persons)
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # For each detection, check if it is a person
        for detection in detections:
            label = detection.get_label()
            if label == "person":
                bbox = detection.get_bbox()
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)

                # Draw the bounding box and center point
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Control the servos to follow the person
                self.track_person(center_x, center_y, frame.shape)

        return Gst.PadProbeReturn.OK

    def track_person(self, center_x, center_y, frame_shape):
        frame_width = frame_shape[1]
        frame_height = frame_shape[0]

        # Calculate pan and tilt angles based on center of the bounding box
        pan_angle = np.interp(center_x, [0, frame_width], [0, 180])  # Horizontal axis (pan)
        tilt_angle = np.interp(center_y, [0, frame_height], [0, 180])  # Vertical axis (tilt)

        # Implement dead zone to avoid unnecessary movements
        if abs(pan_angle - pan_servo.angle) < DEAD_ZONE:
            pan_angle = pan_servo.angle
        if abs(tilt_angle - tilt_servo.angle) < DEAD_ZONE:
            tilt_angle = tilt_servo.angle

        # Move the servos smoothly to the target angles
        self.smooth_move_to(pan_servo, pan_servo.angle, pan_angle)
        self.smooth_move_to(tilt_servo, tilt_servo.angle, tilt_angle)

    def smooth_move_to(self, target_servo, current_angle, target_angle):
        """Smoothly move the servo to the target position in small steps."""
        while abs(target_angle - current_angle) > DEAD_ZONE:
            if target_angle > current_angle:
                current_angle += SLOW_MOVE_STEP
            elif target_angle < current_angle:
                current_angle -= SLOW_MOVE_STEP
            target_servo.angle = current_angle
            time.sleep(MOVE_DELAY)

# -----------------------------------------------------------------------------------------------
# Initialize Hailo pipeline for person detection
# -----------------------------------------------------------------------------------------------
def initialize_hailo_pipeline():
    user_data = UserAppCallbackClass()
    app_callback = user_data.app_callback

    # Initialize the Hailo detection application
    app = GStreamerDetectionApp(app_callback, user_data)
    return app

def track_person():
    # Initialize the Hailo person detection pipeline
    app = initialize_hailo_pipeline()

    # Open video capture device (you may change the device if needed)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret:
            print("Failed to grab frame")
            break

        # Let the Hailo app process the frame and track the person
        app.run()  # This will invoke the app_callback and process detections

        # Display the frame with tracking information
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        cv2.imshow("Frame", frame)

        # Wait for a key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the person tracking function
track_person()
