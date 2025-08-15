import time
import board
import busio
from adafruit_motor import servo
import adafruit_pca9685
import cv2
import numpy as np

# Set up the I2C connection and PCA9685 board
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50  # Set the frequency for the servos

# Create servo objects for pan and tilt (using channels 0 and 1)
pan_servo = servo.Servo(pca.channels[0])
tilt_servo = servo.Servo(pca.channels[1])

# Set initial positions of the servos (you can adjust these)
pan_servo.angle = 90  # Start in the middle (horizontal)
tilt_servo.angle = 90  # Start in the middle (vertical)

# Camera setup
cap = cv2.VideoCapture(0)  # Use the default camera (0 is typically the first camera)

# Set camera properties for better compatibility
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Blue color range in HSV (you can adjust this for better tracking)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Smoothing parameters
DEAD_ZONE = 30  # Increased dead zone to reduce jittering
SMOOTHING_FACTOR = 0.4  # How much to blend new position with old (0.1 = very smooth, 0.9 = very responsive)
MAX_STEP_SIZE = 4  # Maximum degrees the servo can move in one step
MIN_CONTOUR_AREA = 500  # Minimum area to consider a valid object

# Variables for smoothing
last_pan_angle = 90.0
last_tilt_angle = 90.0
last_valid_detection = time.time()
DETECTION_TIMEOUT = 2.0  # Seconds to wait before stopping tracking

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

def track_blue_color():
    global last_pan_angle, last_tilt_angle, last_valid_detection
    
    while True:
        ret, frame = cap.read()  # Capture a frame from the camera
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.1)  # Wait a bit before retrying
            continue
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Invalid frame received")
            time.sleep(0.1)
            continue
        
        # Convert the frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of the blue objects in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_detected = False
        
        if contours:
            # Find the largest contour (assuming it's the object to track)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only process if the contour is large enough
            if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                object_detected = True
                last_valid_detection = time.time()
                
                # Get the bounding box for the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Get the center of the object
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Draw a circle around the detected blue object
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Get the frame dimensions
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                
                # Calculate center of frame
                frame_center_x = frame_width // 2
                frame_center_y = frame_height // 2
                
                # Calculate error (distance from center)
                error_x = center_x - frame_center_x
                error_y = center_y - frame_center_y
                
                # Only move if outside dead zone
                if abs(error_x) > DEAD_ZONE or abs(error_y) > DEAD_ZONE:
                    # Calculate target angles based on error
                    # Reduce sensitivity by scaling the error
                    pan_adjustment = (error_x / frame_width) * 60  # Scale down movement
                    tilt_adjustment = (error_y / frame_height) * 60
                    
                    # Reverse pan logic
                    target_pan = last_pan_angle - pan_adjustment
                    target_tilt = last_tilt_angle + tilt_adjustment
                    
                    # Ensure angles are within valid servo range
                    target_pan = max(0, min(180, target_pan))
                    target_tilt = max(0, min(180, target_tilt))
                    
                    # Apply smoothing
                    new_pan = smooth_angle_update(last_pan_angle, target_pan, SMOOTHING_FACTOR, MAX_STEP_SIZE)
                    new_tilt = smooth_angle_update(last_tilt_angle, target_tilt, SMOOTHING_FACTOR, MAX_STEP_SIZE)
                    
                    # Update servo positions
                    pan_servo.angle = new_pan
                    tilt_servo.angle = new_tilt
                    
                    # Store the new angles
                    last_pan_angle = new_pan
                    last_tilt_angle = new_tilt
                    
                    # Print the tracking information
                    print(f"Center: ({center_x}, {center_y}) | Error: ({error_x:+4.0f}, {error_y:+4.0f}) | Pan: {new_pan:.1f} | Tilt: {new_tilt:.1f}")
                else:
                    print(f"Object centered - no movement needed")
        
        # Check for detection timeout
        if not object_detected and (time.time() - last_valid_detection) > DETECTION_TIMEOUT:
            print("No object detected for extended period")
        
        # Draw crosshair at center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        cv2.line(frame, (frame_center_x - 20, frame_center_y), (frame_center_x + 20, frame_center_y), (0, 255, 255), 2)
        cv2.line(frame, (frame_center_x, frame_center_y - 20), (frame_center_x, frame_center_y + 20), (0, 255, 255), 2)
        
        # Draw dead zone
        cv2.rectangle(frame, 
                     (frame_center_x - DEAD_ZONE, frame_center_y - DEAD_ZONE),
                     (frame_center_x + DEAD_ZONE, frame_center_y + DEAD_ZONE),
                     (0, 100, 100), 1)
        
        # Flip the frame horizontally for correct visualization
        frame = cv2.flip(frame, 1)  # Flip horizontally
        
        # Show the frame with the blue object highlighted
        try:
            cv2.imshow("Frame", frame)
        except cv2.error as e:
            print(f"Display error: {e}")
            # Continue without display if GUI is not available
            pass
        
        # Add a small delay to prevent overwhelming the servo
        time.sleep(0.05)  # 50ms delay = ~20 FPS
        
        # Wait for a key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close OpenCV windows
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

# Run the blue color tracking
track_blue_color()