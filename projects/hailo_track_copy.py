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
from collections import deque
import threading
from queue import Queue, Empty
import json
import csv
from datetime import datetime
import math
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -----------------------------------------------------------------------------------------------
DEAD_ZONE = 15
SMOOTHING_FACTOR = 0.35
MAX_STEP_SIZE = 5
MIN_CONFIDENCE = 0.3
DETECTION_TIMEOUT = 2.0
PAN_SENSITIVITY = 45
TILT_SENSITIVITY = 35
FRAME_SKIP_COUNT = 1
DETECTION_HISTORY_SIZE = 3

# Camera and distance calculation constants
CAMERA_FOV_HORIZONTAL = 73.3 # degrees - typical webcam horizontal FOV
CAMERA_FOV_VERTICAL = 58.3   # degrees - typical webcam vertical FOV
AVERAGE_PERSON_HEIGHT = 1.7   # meters (5'7")
AVERAGE_PERSON_WIDTH = 0.45   # meters (shoulder width)
FOCAL_LENGTH_PIXELS = 430 #Approximate focal length in pixels (calibrate for your camera)

# Servo mount geometry (adjust based on your setup)
SERVO_MOUNT_HEIGHT = 1.05    # meters - height of camera above ground
PAN_SERVO_OFFSET = 0.0        # meters - horizontal offset from center
TILT_SERVO_OFFSET = 0.0       # meters - vertical offset from rotation axis

# Camera tilt offset due to weight (positive = camera tilted down)
CAMERA_TILT_OFFSET = 5.0      # degrees - adjust based on your camera's actual tilt

# -----------------------------------------------------------------------------------------------
# DISTANCE CALCULATOR CLASS
# -----------------------------------------------------------------------------------------------
class DistanceCalculator:
    """Calculate distance to person using multiple methods"""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Calculate focal length based on FOV
        self.focal_length_x = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        
        # Distance history for smoothing
        self.distance_history = deque(maxlen=5)
        
    def update_frame_size(self, width, height):
        """Update frame dimensions and recalculate focal lengths"""
        self.frame_width = width
        self.frame_height = height
        self.focal_length_x = width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
    
    def calculate_distance_from_bbox(self, bbox):
        """Calculate distance using bounding box size"""
        # Get bbox dimensions in pixels
        bbox_width_pixels = bbox.width() * self.frame_width
        bbox_height_pixels = bbox.height() * self.frame_height
        
        # Method 1: Using height (more reliable for standing persons)
        distance_from_height = (AVERAGE_PERSON_HEIGHT * self.focal_length_y) / bbox_height_pixels
        
        # Method 2: Using width
        distance_from_width = (AVERAGE_PERSON_WIDTH * self.focal_length_x) / bbox_width_pixels
        
        # Weighted average (height is usually more reliable)
        distance = (distance_from_height * 0.7 + distance_from_width * 0.3)
        
        # Add to history and return smoothed value
        self.distance_history.append(distance)
        return self._get_smoothed_distance()
    
    def calculate_3d_position(self, bbox, pan_angle, tilt_angle, distance):
        """Calculate 3D position of person relative to servo mount"""
        # Get person center in normalized coordinates
        center_x = bbox.xmin() + bbox.width() / 2
        center_y = bbox.ymin() + bbox.height() / 2
        
        # Convert servo angles to radians
        pan_rad = math.radians(pan_angle - 90)  # 90 degrees is center
        
        # Apply camera tilt offset correction
        actual_tilt_angle = tilt_angle + CAMERA_TILT_OFFSET
        tilt_rad = math.radians(90 - actual_tilt_angle)  # 90 degrees is level
        
        # Calculate horizontal distance (ground projection)
        horizontal_distance = distance * math.cos(tilt_rad)
        
        # Calculate 3D coordinates
        x = horizontal_distance * math.sin(pan_rad)
        y = horizontal_distance * math.cos(pan_rad)
        z = distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT
        
        return x, y, z
    
    def calculate_angular_size(self, bbox):
        """Calculate angular size of detected person"""
        # Angular width and height in degrees
        angular_width = (bbox.width() * CAMERA_FOV_HORIZONTAL)
        angular_height = (bbox.height() * CAMERA_FOV_VERTICAL)
        
        return angular_width, angular_height
    
    def _get_smoothed_distance(self):
        """Get smoothed distance from history"""
        if not self.distance_history:
            return 0.0
        
        # Remove outliers (simple method)
        sorted_distances = sorted(self.distance_history)
        if len(sorted_distances) >= 3:
            # Remove highest and lowest
            filtered = sorted_distances[1:-1]
        else:
            filtered = sorted_distances
        
        return sum(filtered) / len(filtered) if filtered else sorted_distances[0]

# -----------------------------------------------------------------------------------------------
# ENHANCED DATA LOGGER CLASS
# -----------------------------------------------------------------------------------------------
class ServoDataLogger:
    """Enhanced data logging with distance tracking"""
    
    def __init__(self, log_dir="servo_logs"):
        # Create log directory relative to the script location
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / log_dir
        
        print(f"🔍 Script location: {script_dir}")
        print(f"🔍 Current working directory: {Path.cwd()}")
        print(f"📊 Creating logs in: {self.log_dir}")
        
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"session_{timestamp}.json"
        
        # Enhanced CSV headers with distance data
        self.csv_headers = [
            'timestamp', 'frame_count', 'pan_angle', 'tilt_angle', 
            'pan_velocity', 'tilt_velocity', 'detection_confidence', 
            'person_detected', 'tracking_active', 'target_lost_frames',
            'distance_meters', 'x_position', 'y_position', 'z_position',
            'angular_width', 'angular_height', 'bbox_width', 'bbox_height'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_headers)
        
        # Session tracking with distance statistics
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'log_files': {
                'csv': str(self.csv_file),
                'json': str(self.json_file)
            },
            'statistics': {
                'total_detections': 0, 
                'total_movements': 0,
                'min_distance': float('inf'),
                'max_distance': 0.0,
                'avg_distance': 0.0,
                'distance_samples': 0
            },
            'events': []
        }
        
        print(f"📊 Data logging to: {self.log_dir}")
        print(f"   CSV: {self.csv_file.name}")
        print(f"   JSON: {self.json_file.name}")
        
    def log_frame_data(self, frame_count, pan_angle, tilt_angle, pan_velocity, 
                      tilt_velocity, detection_confidence, person_detected, 
                      tracking_active, target_lost_frames, distance_data=None):
        """Log enhanced frame data including distance"""
        try:
            # Default distance values
            distance = x_pos = y_pos = z_pos = 0.0
            angular_width = angular_height = bbox_width = bbox_height = 0.0
            
            if distance_data:
                distance = distance_data.get('distance', 0.0)
                x_pos = distance_data.get('x_position', 0.0)
                y_pos = distance_data.get('y_position', 0.0)
                z_pos = distance_data.get('z_position', 0.0)
                angular_width = distance_data.get('angular_width', 0.0)
                angular_height = distance_data.get('angular_height', 0.0)
                bbox_width = distance_data.get('bbox_width', 0.0)
                bbox_height = distance_data.get('bbox_height', 0.0)
                
                # Update distance statistics
                if distance > 0:
                    stats = self.session_data['statistics']
                    stats['min_distance'] = min(stats['min_distance'], distance)
                    stats['max_distance'] = max(stats['max_distance'], distance)
                    stats['distance_samples'] += 1
                    # Running average
                    stats['avg_distance'] = (
                        (stats['avg_distance'] * (stats['distance_samples'] - 1) + distance) / 
                        stats['distance_samples']
                    )
            
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), frame_count, pan_angle, tilt_angle,
                    pan_velocity, tilt_velocity, detection_confidence,
                    person_detected, tracking_active, target_lost_frames,
                    distance, x_pos, y_pos, z_pos,
                    angular_width, angular_height, bbox_width, bbox_height
                ])
            
            # Update other statistics
            if person_detected:
                self.session_data['statistics']['total_detections'] += 1
            if abs(pan_velocity) > 1 or abs(tilt_velocity) > 1:
                self.session_data['statistics']['total_movements'] += 1
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_event(self, event_type, description):
        """Log significant events"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description
        }
        self.session_data['events'].append(event)
        print(f"📝 {event_type}: {description}")
    
    def finalize_session(self):
        """Save final session data"""
        self.session_data['end_time'] = datetime.now().isoformat()
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            stats = self.session_data['statistics']
            print(f"\n📊 Session Complete:")
            print(f"   Detections: {stats['total_detections']}")
            print(f"   Movements: {stats['total_movements']}")
            if stats['distance_samples'] > 0:
                print(f"   Distance Range: {stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m")
                print(f"   Average Distance: {stats['avg_distance']:.2f}m")
            print(f"   Data saved to: {self.log_dir}")
            
            # Verify files exist
            print(f"\n📁 Final file verification:")
            if self.csv_file.exists():
                size = self.csv_file.stat().st_size
                print(f"✅ CSV: {self.csv_file} ({size} bytes)")
            else:
                print(f"❌ CSV missing: {self.csv_file}")
                
            if self.json_file.exists():
                size = self.json_file.stat().st_size
                print(f"✅ JSON: {self.json_file} ({size} bytes)")
            else:
                print(f"❌ JSON missing: {self.json_file}")
            
        except Exception as e:
            print(f"Session save error: {e}")

# -----------------------------------------------------------------------------------------------
# SERVO CONTROLLER (unchanged)
# -----------------------------------------------------------------------------------------------
class FastServoController:
    """Optimized servo controller with logging"""
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # Hardware setup
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = adafruit_pca9685.PCA9685(self.i2c)
        self.pca.frequency = 50
        
        self.pan_servo = servo.Servo(self.pca.channels[0])
        self.tilt_servo = servo.Servo(self.pca.channels[2])
        
        # State variables with tilt offset
        self.current_pan = 90.0
        self.current_tilt = 90.0 - CAMERA_TILT_OFFSET  # Compensate for camera tilt
        self.velocity_pan = 0.0
        self.velocity_tilt = 0.0
        self.last_update_time = time.time()
        
        # Initialize servos
        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        
        # Threading
        self.command_queue = Queue(maxsize=5)
        self.running = True
        self.servo_thread = threading.Thread(target=self._servo_worker, daemon=True)
        self.servo_thread.start()
        
        if self.logger:
            self.logger.log_event('servo_init', 'Servo controller initialized')
        
    def _servo_worker(self):
        """Background servo movement thread"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.05)
                if command is None:
                    break
                    
                pan_angle, tilt_angle = command
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                # Calculate velocities
                if dt > 0:
                    self.velocity_pan = (pan_angle - self.current_pan) / dt
                    self.velocity_tilt = (tilt_angle - self.current_tilt) / dt
                
                # Move servos if significant change
                if (abs(pan_angle - self.current_pan) > 0.1 or 
                    abs(tilt_angle - self.current_tilt) > 0.1):
                    
                    try:
                        self.pan_servo.angle = pan_angle
                        self.tilt_servo.angle = tilt_angle
                        self.current_pan = pan_angle
                        self.current_tilt = tilt_angle
                        time.sleep(0.005)  # 5ms delay
                        
                    except Exception as e:
                        print(f"Servo movement error: {e}")
                
                self.last_update_time = current_time
                
            except Empty:
                continue
            except Exception as e:
                print(f"Servo thread error: {e}")
    
    def move_to(self, pan_angle, tilt_angle):
        """Queue servo movement"""
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        
        try:
            # Clear queue for latest command
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            self.command_queue.put_nowait((pan_angle, tilt_angle))
        except:
            pass
    
    def get_current_state(self):
        """Get current angles and velocities"""
        return self.current_pan, self.current_tilt, self.velocity_pan, self.velocity_tilt
    
    def shutdown(self):
        """Clean shutdown"""
        if self.logger:
            self.logger.log_event('servo_shutdown', 'Servo controller shutting down')
        
        self.running = False
        self.command_queue.put(None)
        self.servo_thread.join(timeout=1.0)
        
        try:
            self.pan_servo.angle = 90
            self.tilt_servo.angle = 90
        except:
            pass

# -----------------------------------------------------------------------------------------------
# ENHANCED TRACKER WITH DISTANCE
# -----------------------------------------------------------------------------------------------
class UltraFastTracker:
    """Enhanced person tracker with distance estimation"""
    
    def __init__(self, servo_controller, logger=None):
        self.servo = servo_controller
        self.logger = logger
        
        # Frame properties
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        # Distance calculator
        self.distance_calculator = DistanceCalculator(self.frame_width, self.frame_height)
        
        # Tracking state
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        self.lock_on_target = False
        self.frame_skip_counter = 0
        self.current_distance = 0.0
        self.current_3d_position = (0.0, 0.0, 0.0)
        
        # Movement history for smoothing
        self.pan_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.tilt_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        
    def update_frame_properties(self, width, height):
        """Update frame dimensions"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            self.distance_calculator.update_frame_size(width, height)
            
            if self.logger:
                self.logger.log_event('resolution_change', f'Frame: {width}x{height}')
        
    def track_person(self, bbox, confidence, frame_count):
        """Track detected person with distance estimation"""
        self.frame_skip_counter += 1
        if self.frame_skip_counter < FRAME_SKIP_COUNT:
            return
        self.frame_skip_counter = 0
        
        # Calculate distance
        self.current_distance = self.distance_calculator.calculate_distance_from_bbox(bbox)
        
        # Get current servo angles
        current_pan, current_tilt, pan_vel, tilt_vel = self.servo.get_current_state()
        
        # Calculate 3D position
        x, y, z = self.distance_calculator.calculate_3d_position(
            bbox, current_pan, current_tilt, self.current_distance
        )
        self.current_3d_position = (x, y, z)
        
        # Calculate angular size
        angular_width, angular_height = self.distance_calculator.calculate_angular_size(bbox)
        
        # Calculate person center
        center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
        center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
        
        # Calculate error from frame center
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        
        # Adjust dead zone based on distance (closer = smaller dead zone)
        dynamic_dead_zone = DEAD_ZONE * (1 + self.current_distance / 10.0)
        
        # Move if outside dead zone
        if abs(error_x) > dynamic_dead_zone or abs(error_y) > dynamic_dead_zone:
            # Calculate adjustments with distance compensation
            distance_factor = min(2.0, max(0.5, 2.0 / self.current_distance))
            
            pan_adjustment = -error_x * (PAN_SENSITIVITY / self.frame_width) * distance_factor
            tilt_adjustment = error_y * (TILT_SENSITIVITY / self.frame_height) * distance_factor
            
            # Apply confidence boost
            confidence_multiplier = min(2.0, confidence + 0.5)
            pan_adjustment *= confidence_multiplier
            tilt_adjustment *= confidence_multiplier
            
            # Smooth movement
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            new_pan = self._smooth_angle(current_pan, target_pan)
            new_tilt = self._smooth_angle(current_tilt, target_tilt)
            
            # Apply smoothing history
            self.pan_history.append(new_pan)
            self.tilt_history.append(new_tilt)
            
            if len(self.pan_history) >= 2:
                weights = [1.0, 2.0, 3.0][:len(self.pan_history)]
                avg_pan = sum(w * angle for w, angle in zip(weights, self.pan_history)) / sum(weights)
                avg_tilt = sum(w * angle for w, angle in zip(weights, self.tilt_history)) / sum(weights)
            else:
                avg_pan, avg_tilt = new_pan, new_tilt
            
            self.servo.move_to(avg_pan, avg_tilt)
            
            if not self.lock_on_target:
                self.lock_on_target = True
                if self.logger:
                    self.logger.log_event('target_lock', 
                        f'Target locked at {self.current_distance:.2f}m')
                print(f"🎯 Target locked at {self.current_distance:.2f}m")
        
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        
        # Log frame data with distance information
        if self.logger:
            distance_data = {
                'distance': self.current_distance,
                'x_position': x,
                'y_position': y,
                'z_position': z,
                'angular_width': angular_width,
                'angular_height': angular_height,
                'bbox_width': bbox.width(),
                'bbox_height': bbox.height()
            }
            
            self.logger.log_frame_data(
                frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                confidence, True, self.is_tracking_active(), self.target_lost_frames,
                distance_data
            )
    
    def handle_lost_target(self, frame_count):
        """Handle when no person detected"""
        self.target_lost_frames += 1
        
        if self.target_lost_frames > 10 and self.lock_on_target:
            self.lock_on_target = False
            if self.logger:
                self.logger.log_event('target_lost', 'Target lost - scanning mode')
            print("🔍 Target lost - scanning mode")
        
        # Log no-detection frame
        if self.logger:
            current_pan, current_tilt, pan_vel, tilt_vel = self.servo.get_current_state()
            self.logger.log_frame_data(
                frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                0.0, False, self.is_tracking_active(), self.target_lost_frames,
                None
            )
    
    def _smooth_angle(self, current, target):
        """Apply smoothing to angle changes"""
        diff = (target - current) * SMOOTHING_FACTOR
        diff = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, diff))
        return current + diff
    
    def is_tracking_active(self):
        """Check if tracking is active"""
        return (time.time() - self.last_detection_time) < DETECTION_TIMEOUT
    
    def get_current_distance_info(self):
        """Get current distance and position information"""
        return {
            'distance': self.current_distance,
            'position_3d': self.current_3d_position
        }

# -----------------------------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------------------------
print("Initializing ULTRA-FAST servo system with DISTANCE TRACKING...")
data_logger = ServoDataLogger()
fast_servo_controller = FastServoController(data_logger)
tracker = UltraFastTracker(fast_servo_controller, data_logger)

class OptimizedAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        
    def new_function(self):
        return "Ultra-Fast Tracking with Distance Estimation: "

# -----------------------------------------------------------------------------------------------
# MAIN CALLBACK
# -----------------------------------------------------------------------------------------------
def ultra_fast_app_callback(pad, info, user_data):
    """Main processing callback with distance tracking"""
    
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    frame_count = user_data.get_count()
    
    # Update frame properties periodically
    if frame_count % 30 == 0:
        format, width, height = get_caps_from_pad(pad)
        if width and height:
            tracker.update_frame_properties(width, height)
    
    # Get frame for display
    frame = None
    if user_data.use_frame:
        format, width, height = get_caps_from_pad(pad)
        if format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Process detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Find best person detection
    best_person = None
    best_area = 0
    
    for detection in detections:
        if detection.get_label() == "person":
            confidence = detection.get_confidence()
            if confidence >= MIN_CONFIDENCE:
                bbox = detection.get_bbox()
                area = bbox.width() * bbox.height()
                
                if area > best_area:
                    best_area = area
                    best_person = {'bbox': bbox, 'confidence': confidence}
    
    # Track or handle lost target
    if best_person:
        tracker.track_person(best_person['bbox'], best_person['confidence'], frame_count)
        
        # Calibration mode
        if calibration_helper.calibrating:
            calibration_helper.add_measurement(best_person['bbox'], tracker.frame_height)
        
        if frame_count % 60 == 0:
            distance_info = tracker.get_current_distance_info()
            print(f"🏃 Tracking: Conf {best_person['confidence']:.2f}, "
                  f"Distance: {distance_info['distance']:.2f}m")
    else:
        tracker.handle_lost_target(frame_count)
    
    # Enhanced frame annotation
    if user_data.use_frame and frame is not None:
        center_x, center_y = tracker.frame_center_x, tracker.frame_center_y
        
        # Crosshair
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 1)
        
        # Status text with distance
        if best_person:
            distance_info = tracker.get_current_distance_info()
            distance = distance_info['distance']
            x, y, z = distance_info['position_3d']
            
            if calibration_helper.calibrating:
                cv2.putText(frame, "CALIBRATION MODE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Place person at {CALIBRATION_DISTANCE}m", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Measurements: {len(calibration_helper.measurements)}/10", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"TRACKING: {distance:.2f}m", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"3D Pos: ({x:.1f}, {y:.1f}, {z:.1f})m", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw bounding box if visible
            if best_person['bbox']:
                bbox = best_person['bbox']
                x1 = int(bbox.xmin() * frame.shape[1])
                y1 = int(bbox.ymin() * frame.shape[0])
                x2 = int((bbox.xmin() + bbox.width()) * frame.shape[1])
                y2 = int((bbox.ymin() + bbox.height()) * frame.shape[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Distance label on bbox
                cv2.putText(frame, f"{distance:.1f}m", 
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Servo info
        pan, tilt, pan_vel, tilt_vel = fast_servo_controller.get_current_state()
        cv2.putText(frame, f"Pan: {pan:.1f}° ({pan_vel:.1f}°/s)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Tilt: {tilt:.1f}° ({tilt_vel:.1f}°/s)", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# CALIBRATION MODE
# -----------------------------------------------------------------------------------------------
CALIBRATION_MODE = True  # Set to True to enable calibration mode
CALIBRATION_DISTANCE = 2.0  # meters - distance to place person for calibration

class CalibrationHelper:
    """Helper class for camera calibration"""
    
    def __init__(self):
        self.measurements = []
        self.calibrating = CALIBRATION_MODE
        
    def add_measurement(self, bbox, frame_height):
        """Add a bounding box measurement during calibration"""
        if not self.calibrating:
            return
            
        # Calculate person height in pixels
        person_height_pixels = bbox.height() * frame_height
        
        if person_height_pixels > 50:  # Minimum height threshold
            self.measurements.append(person_height_pixels)
            
            if len(self.measurements) >= 10:
                # Calculate average after 10 measurements
                avg_height = sum(self.measurements) / len(self.measurements)
                focal_length = self.calculate_focal_length(CALIBRATION_DISTANCE, avg_height)
                
                print(f"\n📸 CALIBRATION COMPLETE!")
                print(f"   Average person height: {avg_height:.1f} pixels")
                print(f"   Calculated focal length: {focal_length:.1f} pixels")
                print(f"   Current focal length: {FOCAL_LENGTH_PIXELS} pixels")
                print(f"\n✏️  Update your code:")
                print(f"   FOCAL_LENGTH_PIXELS = {focal_length:.1f}")
                print(f"\n💡 Also update CAMERA_FOV values if needed:")
                print(f"   CAMERA_FOV_HORIZONTAL = {self.calculate_fov_horizontal(focal_length):.1f}")
                print(f"   CAMERA_FOV_VERTICAL = {self.calculate_fov_vertical(focal_length):.1f}")
                
                self.calibrating = False
                self.measurements = []
    
    def calculate_focal_length(self, known_distance, height_pixels):
        """Calculate focal length from known distance and pixel height"""
        return (height_pixels * known_distance) / AVERAGE_PERSON_HEIGHT
    
    def calculate_fov_horizontal(self, focal_length):
        """Calculate horizontal FOV from focal length"""
        return 2 * math.degrees(math.atan(tracker.frame_width / (2 * focal_length)))
    
    def calculate_fov_vertical(self, focal_length):
        """Calculate vertical FOV from focal length"""
        return 2 * math.degrees(math.atan(tracker.frame_height / (2 * focal_length)))

# Initialize calibration helper
calibration_helper = CalibrationHelper()

# -----------------------------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    user_data = OptimizedAppCallback()
    app = GStreamerDetectionApp(ultra_fast_app_callback, user_data)
    
    print("🚀 Starting ULTRA-FAST tracking with DISTANCE ESTIMATION...")
    
    if CALIBRATION_MODE:
        print("\n🎯 CALIBRATION MODE ACTIVE!")
        print(f"   1. Place a person at EXACTLY {CALIBRATION_DISTANCE} meters from camera")
        print("   2. Make sure they're standing upright and fully visible")
        print("   3. System will take 10 measurements and calculate focal length")
        print("   4. Update the constants with the calculated values\n")
    
    print("📊 Data output location:")
    print(f"   Directory: {data_logger.log_dir}")
    print(f"   CSV file: {data_logger.csv_file.name}")
    print(f"   JSON file: {data_logger.json_file.name}")
    print("\n📏 Distance Tracking Configuration:")
    print(f"   Camera FOV: {CAMERA_FOV_HORIZONTAL}° x {CAMERA_FOV_VERTICAL}°")
    print(f"   Camera tilt offset: {CAMERA_TILT_OFFSET}° (compensating for weight)")
    print(f"   Average person height: {AVERAGE_PERSON_HEIGHT}m")
    print(f"   Servo mount height: {SERVO_MOUNT_HEIGHT}m")
    print(f"   Current focal length: {FOCAL_LENGTH_PIXELS} pixels")
    
    if not CALIBRATION_MODE:
        print("\n💡 TIP: To calibrate your camera:")
        print("   1. Set CALIBRATION_MODE = True at the top of the script")
        print("   2. Restart the program and follow instructions")
    
    print("\nPress Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping system...")
    except Exception as e:
        print(f"❌ Error: {e}")
        data_logger.log_event('error', f'Application error: {str(e)}')
    finally:
        print("📊 Finalizing logs...")
        data_logger.finalize_session()
        fast_servo_controller.shutdown()
        print("✅ Shutdown complete")
