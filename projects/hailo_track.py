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
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# ULTRA-FAST SERVO CONTROLLER
# -----------------------------------------------------------------------------------------------
class FastServoController:
    """Ultra-fast servo controller with aggressive optimization"""
    
    def __init__(self):
        # Hardware setup
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = adafruit_pca9685.PCA9685(self.i2c)
        self.pca.frequency = 50
        
        self.pan_servo = servo.Servo(self.pca.channels[0])
        self.tilt_servo = servo.Servo(self.pca.channels[2])
        
        # Initial positions
        self.current_pan = 90.0
        self.current_tilt = 90.0
        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        
        # Thread control - higher priority processing
        self.command_queue = Queue(maxsize=5)  # Smaller queue for faster response
        self.running = True
        self.servo_thread = threading.Thread(target=self._servo_worker, daemon=True)
        self.servo_thread.start()
        
        # FASTER MOVEMENT SETTINGS
        self.movement_threshold = 0.1  # Much more sensitive (was 0.3)
        self.servo_delay = 0.005  # Faster servo updates (was 0.02)
        
        # Predictive movement
        self.velocity_pan = 0.0
        self.velocity_tilt = 0.0
        self.last_angles = (90.0, 90.0)
        self.last_update_time = time.time()
        
    def _servo_worker(self):
        """Optimized background thread for ultra-fast servo movements"""
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.05)  # Faster timeout
                if command is None:
                    break
                    
                pan_angle, tilt_angle = command
                current_time = time.time()
                
                # Calculate velocity for predictive movement
                dt = current_time - self.last_update_time
                if dt > 0:
                    self.velocity_pan = (pan_angle - self.current_pan) / dt
                    self.velocity_tilt = (tilt_angle - self.current_tilt) / dt
                
                # Apply predictive offset for smoother tracking
                prediction_time = 0.05  # Predict 50ms ahead
                predicted_pan = pan_angle + (self.velocity_pan * prediction_time)
                predicted_tilt = tilt_angle + (self.velocity_tilt * prediction_time)
                
                # Clamp predicted angles
                predicted_pan = max(0, min(180, predicted_pan))
                predicted_tilt = max(0, min(180, predicted_tilt))
                
                # Move if any change (much more sensitive)
                if (abs(predicted_pan - self.current_pan) > self.movement_threshold or 
                    abs(predicted_tilt - self.current_tilt) > self.movement_threshold):
                    
                    try:
                        # Move both servos simultaneously for better performance
                        self.pan_servo.angle = predicted_pan
                        self.tilt_servo.angle = predicted_tilt
                        
                        self.current_pan = predicted_pan
                        self.current_tilt = predicted_tilt
                        
                        time.sleep(self.servo_delay)  # Much faster delay
                        
                    except Exception as e:
                        print(f"Servo movement error: {e}")
                
                self.last_update_time = current_time
                self.last_angles = (pan_angle, tilt_angle)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Servo thread error: {e}")
    
    def move_to(self, pan_angle, tilt_angle):
        """Ultra-fast servo movement queueing"""
        # Clamp angles
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        
        try:
            # Clear queue and add latest command for fastest response
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            
            self.command_queue.put_nowait((pan_angle, tilt_angle))
        except:
            pass
    
    def get_current_angles(self):
        """Get current servo angles"""
        return self.current_pan, self.current_tilt
    
    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.command_queue.put(None)
        self.servo_thread.join(timeout=1.0)
        # Reset to center
        try:
            self.pan_servo.angle = 90
            self.tilt_servo.angle = 90
        except:
            pass

# -----------------------------------------------------------------------------------------------
# Faster Tracking Parameters - More Aggressive Settings
# -----------------------------------------------------------------------------------------------
FAST_DEAD_ZONE = 15          # Smaller dead zone (was 40)
FAST_SMOOTHING_FACTOR = 0.35 # More responsive (was 0.15)
FAST_MAX_STEP_SIZE = 5       # Larger steps allowed (was 2)
MIN_CONFIDENCE = 0.3         # Lower confidence for faster detection (was 0.4)
DETECTION_TIMEOUT = 2.0      # Shorter timeout (was 3.0)
FAST_PAN_SENSITIVITY = 45    # Higher sensitivity (was 30)
FAST_TILT_SENSITIVITY = 35   # Higher sensitivity (was 25)

# More aggressive frame processing
FAST_FRAME_SKIP_COUNT = 1    # Process every frame (was 2)
DETECTION_HISTORY_SIZE = 3   # Smaller history for faster response (was 5)

class UltraFastTracker:
    """Ultra-responsive person tracker with minimal latency"""
    
    def __init__(self, servo_controller):
        self.servo = servo_controller
        
        # Smaller buffers for faster response
        self.pan_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.tilt_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        
        # Cached values
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        # Higher sensitivity ratios
        self.pan_ratio = FAST_PAN_SENSITIVITY / 640.0
        self.tilt_ratio = FAST_TILT_SENSITIVITY / 480.0
        
        # Tracking state
        self.last_detection_time = time.time()
        self.tracked_id = None
        self.frame_skip_counter = 0
        
        # Aggressive tracking mode
        self.aggressive_mode = True
        self.lock_on_target = False
        self.target_lost_frames = 0
        
    def update_frame_properties(self, width, height):
        """Update cached frame properties when resolution changes"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            # Update pre-computed ratios
            self.pan_ratio = FAST_PAN_SENSITIVITY / width
            self.tilt_ratio = FAST_TILT_SENSITIVITY / height
        
    def track_person(self, bbox, confidence=1.0):
        """Ultra-fast person tracking with aggressive response"""
        
        # Process more frames for faster tracking
        self.frame_skip_counter += 1
        if self.frame_skip_counter < FAST_FRAME_SKIP_COUNT:
            return
        self.frame_skip_counter = 0
        
        # Calculate center
        center_x_rel = bbox.xmin() + (bbox.width() * 0.5)
        center_y_rel = bbox.ymin() + (bbox.height() * 0.5)
        
        center_x_px = center_x_rel * self.frame_width
        center_y_px = center_y_rel * self.frame_height
        
        # Calculate error from center
        error_x = center_x_px - self.frame_center_x
        error_y = center_y_px - self.frame_center_y
        
        # More aggressive dead zone
        if abs(error_x) > FAST_DEAD_ZONE or abs(error_y) > FAST_DEAD_ZONE:
            current_pan, current_tilt = self.servo.get_current_angles()
            
            # More responsive adjustment calculations
            pan_adjustment = -error_x * self.pan_ratio
            tilt_adjustment = error_y * self.tilt_ratio
            
            # Confidence-based acceleration
            confidence_multiplier = min(2.0, confidence + 0.5)  # Boost for confident detections
            pan_adjustment *= confidence_multiplier
            tilt_adjustment *= confidence_multiplier
            
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            # Less smoothing for faster response
            new_pan = self.fast_smooth_angle_update(current_pan, target_pan)
            new_tilt = self.fast_smooth_angle_update(current_tilt, target_tilt)
            
            # Shorter history for faster response
            self.pan_history.append(new_pan)
            self.tilt_history.append(new_tilt)
            
            # Direct movement without heavy averaging
            if len(self.pan_history) >= 2:
                # Weighted average favoring recent movements
                weights = [1.0, 2.0, 3.0][:len(self.pan_history)]
                weighted_sum_pan = sum(w * angle for w, angle in zip(weights, self.pan_history))
                weighted_sum_tilt = sum(w * angle for w, angle in zip(weights, self.tilt_history))
                weight_total = sum(weights[:len(self.pan_history)])
                
                avg_pan = weighted_sum_pan / weight_total
                avg_tilt = weighted_sum_tilt / weight_total
            else:
                avg_pan, avg_tilt = new_pan, new_tilt
            
            # Immediate servo command
            self.servo.move_to(avg_pan, avg_tilt)
            
            self.last_detection_time = time.time()
            self.target_lost_frames = 0
            
            if not self.lock_on_target:
                self.lock_on_target = True
                print("🎯 Target locked - entering fast tracking mode")
        
        else:
            # Still in dead zone but update tracking time
            self.last_detection_time = time.time()
    
    def fast_smooth_angle_update(self, current_angle, target_angle):
        """Faster smoothing with larger step sizes"""
        diff = (target_angle - current_angle) * FAST_SMOOTHING_FACTOR
        
        # Larger step limiting for faster movement
        diff = max(-FAST_MAX_STEP_SIZE, min(FAST_MAX_STEP_SIZE, diff))
            
        return current_angle + diff
    
    def handle_lost_target(self):
        """Handle when target is lost"""
        self.target_lost_frames += 1
        
        if self.target_lost_frames > 10:  # Lost for more than 10 frames
            if self.lock_on_target:
                print("🔍 Target lost - scanning mode")
                self.lock_on_target = False
    
    def is_tracking_active(self):
        """Check if we have recent detections"""
        return (time.time() - self.last_detection_time) < DETECTION_TIMEOUT

# Initialize optimized components
print("Initializing ULTRA-FAST servo system...")
fast_servo_controller = FastServoController()
tracker = UltraFastTracker(fast_servo_controller)

# -----------------------------------------------------------------------------------------------
# Optimized User-defined class
# -----------------------------------------------------------------------------------------------
class OptimizedAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.detection_cache = {}  # Cache detection results
        self.frame_counter = 0
        
        # Performance tracking
        self.callback_times = deque(maxlen=60)  # Track last 60 frame times
        self.last_fps_print = time.time()
        
    def new_function(self):
        return "Ultra-Fast Tracking: "

# -----------------------------------------------------------------------------------------------
# Ultra-Fast Callback Function
# -----------------------------------------------------------------------------------------------
def ultra_fast_app_callback(pad, info, user_data):
    """Minimal latency callback optimized for speed"""
    
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    frame_count = user_data.get_count()
    
    # Update frame properties less frequently for performance
    if frame_count % 30 == 0:  # Only every 30 frames
        format, width, height = get_caps_from_pad(pad)
        if width and height:
            tracker.update_frame_properties(width, height)
    
    # Get frame only when needed
    frame = None
    if user_data.use_frame:
        format, width, height = get_caps_from_pad(pad)
        if format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Ultra-fast detection processing
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Streamlined person detection
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
                    best_person = {'detection': detection, 'bbox': bbox, 'confidence': confidence}
    
    # Track immediately if person found
    if best_person:
        tracker.track_person(best_person['bbox'], best_person['confidence'])
        
        # Minimal logging
        if frame_count % 60 == 0:
            print(f"🏃 Fast tracking: Conf {best_person['confidence']:.2f} | Area {best_area:.3f}")
    else:
        tracker.handle_lost_target()
    
    # Minimal frame annotation for speed
    if user_data.use_frame and frame is not None:
        # Just crosshair and basic info
        center_x, center_y = tracker.frame_center_x, tracker.frame_center_y
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 1)
        
        if best_person:
            cv2.putText(frame, "FAST TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show servo angles
        pan, tilt = fast_servo_controller.get_current_angles()
        cv2.putText(frame, f"Pan: {pan:.1f} Tilt: {tilt:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    # Create optimized app callback instance
    user_data = OptimizedAppCallback()
    app = GStreamerDetectionApp(ultra_fast_app_callback, user_data)
    
    print("Starting ULTRA-FAST person tracking system...")
    print("🚀 Performance improvements:")
    print("- 4x faster servo response (5ms delays)")
    print("- Predictive movement algorithms")
    print("- Target locking system") 
    print("- Reduced dead zones (15px)")
    print("- Higher sensitivity settings")
    print("- Every frame processing")
    print("Press Ctrl+C to stop")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping ultra-fast tracking system...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean shutdown
        fast_servo_controller.shutdown()
        print("✅ System shutdown complete")