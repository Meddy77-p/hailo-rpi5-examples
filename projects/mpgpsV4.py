#!/usr/bin/env python3
"""
Enhanced Servo Tracking System with Ultra-Smooth Camera Movement
For Raspberry Pi 5 with CubeOrange and Hailo AI
Enhanced with: Smooth camera movement, reduced jitter, improved tracking
"""

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
from pymavlink import mavutil
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
import statistics
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# ENHANCED SMOOTH CAMERA CONFIGURATION
# ==============================================================================

# Ultra-smooth servo tracking parameters
DEAD_ZONE = 15              # Increased to reduce small jittery movements
SMOOTHING_FACTOR = 0.85     # Much higher for ultra-smooth movement
MAX_STEP_SIZE = 3           # Reduced for gentler movements
MIN_CONFIDENCE = 0.45       # Higher threshold for stable detections
DETECTION_TIMEOUT = 2.0     # Slightly longer timeout
PAN_SENSITIVITY = 25        # Reduced for less aggressive tracking
TILT_SENSITIVITY = 20       # Reduced for less aggressive tracking
FRAME_SKIP_COUNT = 0        # No frame skipping
DETECTION_HISTORY_SIZE = 10 # Increased for better smoothing

# Advanced smoothing parameters
VELOCITY_LIMIT = 20.0           # Maximum degrees per second for smooth movement
ACCELERATION_LIMIT = 40.0       # Maximum degrees per second squared
EXPONENTIAL_SMOOTH_ALPHA = 0.25 # For exponential moving average
MOTION_PREDICTION_FACTOR = 0.7  # How much to trust motion prediction
DISTANCE_SMOOTH_FACTOR = 2.0    # Distance-based smoothing multiplier

# Enhanced camera parameters
CAMERA_FOV_HORIZONTAL = 79.9
CAMERA_FOV_VERTICAL = 64.3
AVERAGE_PERSON_HEIGHT = 1.7
AVERAGE_PERSON_WIDTH = 0.45
FOCAL_LENGTH_PIXELS = 382

# Physical setup
SERVO_MOUNT_HEIGHT = 1.3
CAMERA_TILT_OFFSET = 5.0

# Enhanced MAVLink configuration
MAVLINK_CONNECTION = '/dev/serial0'
MAVLINK_BAUD = 57600
MAVLINK_SYSTEM_ID = 255
MAVLINK_COMPONENT_ID = 190
MAVLINK_HEARTBEAT_INTERVAL = 1.0
MAVLINK_RECONNECT_INTERVAL = 5.0
MAX_MAVLINK_RETRIES = 10

# Enhanced GPS waypoint parameters
GPS_UPDATE_INTERVAL = 0.5
MIN_DISTANCE_FOR_GPS = 2.5
MAX_GPS_POINTS = 150
WAYPOINT_ALTITUDE_OFFSET = 0.0
WAYPOINT_MODE = "ADD"
WAYPOINT_CLEAR_TIMEOUT = 300
MAX_WAYPOINTS = 25

# Enhanced calibration parameters
CALIBRATION_MODE = True
CALIBRATION_DISTANCE = 2.0
CALIBRATION_SAMPLES_REQUIRED = 20
CALIBRATION_TIMEOUT = 60.0
AUTO_CALIBRATION_ENABLED = True
CALIBRATION_ACCURACY_THRESHOLD = 0.95

# Performance optimization
THREAD_POOL_SIZE = 4
PROCESSING_QUEUE_SIZE = 10
ALTITUDE_UPDATE_RATE = 5.0

# ==============================================================================
# ENHANCED MAVLINK GPS HANDLER (Keeping original implementation)
# ==============================================================================

class EnhancedMAVLinkHandler:
    def __init__(self, connection_string=MAVLINK_CONNECTION, baud=MAVLINK_BAUD):
        self.connection_string = connection_string
        self.baud = baud
        self.mavlink_connection = None
        
        # Enhanced position data
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.current_alt_rel = 0.0
        self.current_alt_terrain = 0.0
        self.barometric_altitude = 0.0
        self.current_heading = 0.0
        
        # Enhanced GPS status
        self.gps_fix_type = 0
        self.satellites_visible = 0
        self.gps_accuracy = 0.0
        self.hdop = 0.0
        self.vdop = 0.0
        
        # Home position
        self.home_lat = None
        self.home_lon = None
        self.home_alt = None
        self.home_set = False
        
        # Mission management
        self.mission_count = 0
        self.current_wp_seq = 0
        self.active_waypoint = None
        self.waypoint_reached_threshold = 5.0
        self.last_waypoint_time = 0
        self.waypoint_mode = WAYPOINT_MODE
        
        # Connection management
        self.running = True
        self.connected = False
        self.mavlink_thread = None
        self.heartbeat_thread = None
        self.last_heartbeat = 0
        self.last_gps_update = 0
        self.connection_retries = 0
        self.last_reconnect_attempt = 0
        
        # Data storage
        self.gps_points = deque(maxlen=MAX_GPS_POINTS)
        self.altitude_history = deque(maxlen=50)
        self.last_point_time = 0
        self.EARTH_RADIUS = 6371000
        
        # Statistics
        self.total_messages_received = 0
        self.connection_uptime = 0
        
        self.connect_with_retry()
    
    def connect_with_retry(self):
        """Enhanced connection with automatic retry"""
        while self.connection_retries < MAX_MAVLINK_RETRIES and self.running:
            if self.connect():
                return True
            
            self.connection_retries += 1
            wait_time = min(30, MAVLINK_RECONNECT_INTERVAL * self.connection_retries)
            print(f"🔄 Retry {self.connection_retries}/{MAX_MAVLINK_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
        
        print("❌ MAVLink connection failed after all retries")
        return False
    
    def connect(self):
        try:
            print(f"🛰️ Connecting to CubeOrange at {self.connection_string}...")
            self.mavlink_connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
                source_system=MAVLINK_SYSTEM_ID,
                source_component=MAVLINK_COMPONENT_ID,
                autoreconnect=True
            )
            
            print("⏳ Waiting for heartbeat...")
            self.mavlink_connection.wait_heartbeat(timeout=15)
            print("✅ MAVLink connection established!")
            
            self.connected = True
            self.connection_retries = 0
            self.last_heartbeat = time.time()
            self.connection_uptime = time.time()
            
            self.request_enhanced_data_streams()
            
            self.mavlink_thread = threading.Thread(target=self._enhanced_mavlink_receiver, daemon=True)
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
            
            self.mavlink_thread.start()
            self.heartbeat_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ MAVLink connection failed: {e}")
            self.connected = False
            return False
    
    def request_enhanced_data_streams(self):
        """Request all necessary data streams from CubeOrange"""
        try:
            streams = [
                (mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS, 10),
                (mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 5),
                (mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 5),
                (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10),
                (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10),
                (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 10),
                (mavutil.mavlink.MAV_DATA_STREAM_EXTRA3, 5),
            ]
            
            for stream_id, rate in streams:
                self.mavlink_connection.mav.request_data_stream_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component,
                    stream_id, rate, 1
                )
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error requesting enhanced data streams: {e}")
    
    def _enhanced_mavlink_receiver(self):
        """Enhanced message receiver with comprehensive data handling"""
        while self.running:
            try:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                
                self.total_messages_received += 1
                msg_type = msg.get_type()
                current_time = time.time()
                
                if msg_type == 'GPS_RAW_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    self.gps_fix_type = msg.fix_type
                    self.satellites_visible = msg.satellites_visible
                    self.gps_accuracy = getattr(msg, 'eph', 0) / 100.0
                    self.last_gps_update = current_time
                    
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    self.current_alt_rel = msg.relative_alt / 1000.0
                    self.current_heading = msg.hdg / 100.0
                    
                elif msg_type == 'SCALED_PRESSURE':
                    self.barometric_altitude = msg.press_abs
                    
                elif msg_type == 'TERRAIN_REPORT':
                    self.current_alt_terrain = msg.current_height
                    
                elif msg_type == 'ALTITUDE':
                    self.altitude_history.append({
                        'timestamp': current_time,
                        'altitude_monotonic': msg.altitude_monotonic,
                        'altitude_amsl': msg.altitude_amsl,
                        'altitude_local': msg.altitude_local,
                        'altitude_relative': msg.altitude_relative,
                        'altitude_terrain': msg.altitude_terrain,
                        'bottom_clearance': msg.bottom_clearance
                    })
                
                elif msg_type == 'ATTITUDE':
                    yaw_rad = msg.yaw
                    self.current_heading = math.degrees(yaw_rad) % 360
                    
                elif msg_type == 'HOME_POSITION':
                    self.home_lat = msg.latitude / 1e7
                    self.home_lon = msg.longitude / 1e7
                    self.home_alt = msg.altitude / 1000.0
                    self.home_set = True
                    print(f"🏠 Home position set: {self.home_lat:.6f}, {self.home_lon:.6f}")
                    
                elif msg_type == 'MISSION_CURRENT':
                    self.current_wp_seq = msg.seq
                    
                elif msg_type == 'MISSION_ITEM_REACHED':
                    print(f"✅ Reached waypoint {msg.seq}")
                    
                elif msg_type == 'HEARTBEAT':
                    self.last_heartbeat = current_time
                    
            except Exception as e:
                if self.running and self.connected:
                    print(f"MAVLink receive error: {e}")
                    self.check_connection_health()
    
    def _heartbeat_monitor(self):
        """Monitor connection health and attempt reconnection if needed"""
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - self.last_heartbeat > 10.0 and self.connected:
                    print("⚠️ MAVLink heartbeat lost - attempting reconnection...")
                    self.connected = False
                    self.reconnect()
                
                time.sleep(MAVLINK_HEARTBEAT_INTERVAL)
                
            except Exception as e:
                print(f"Heartbeat monitor error: {e}")
    
    def check_connection_health(self):
        """Check and report connection health"""
        current_time = time.time()
        
        if current_time - self.last_heartbeat > 5.0:
            self.connected = False
            print("⚠️ Connection health check failed")
    
    def reconnect(self):
        """Attempt to reconnect to MAVLink"""
        if not self.running:
            return
            
        current_time = time.time()
        if current_time - self.last_reconnect_attempt < MAVLINK_RECONNECT_INTERVAL:
            return
            
        self.last_reconnect_attempt = current_time
        print("🔄 Attempting MAVLink reconnection...")
        
        try:
            if self.mavlink_connection:
                self.mavlink_connection.close()
            
            self.connect()
            
        except Exception as e:
            print(f"Reconnection failed: {e}")
    
    def get_enhanced_status(self):
        """Get comprehensive system status"""
        current_time = time.time()
        return {
            'connected': self.connected,
            'connection_uptime': current_time - self.connection_uptime if self.connected else 0,
            'total_messages': self.total_messages_received,
            'gps_fix': self.gps_fix_type,
            'satellites': self.satellites_visible,
            'gps_accuracy': self.gps_accuracy,
            'latitude': self.current_lat,
            'longitude': self.current_lon,
            'altitude': self.current_alt,
            'altitude_relative': self.current_alt_rel,
            'altitude_terrain': self.current_alt_terrain,
            'barometric_altitude': self.barometric_altitude,
            'heading': self.current_heading,
            'last_gps_update': current_time - self.last_gps_update,
            'last_heartbeat': current_time - self.last_heartbeat,
            'points_logged': len(self.gps_points),
            'mission_count': self.mission_count,
            'home_set': self.home_set,
            'altitude_data': {
                'gps_altitude': self.current_alt,
                'relative_altitude': self.current_alt_rel,
                'terrain_altitude': self.current_alt_terrain,
                'barometric_altitude': self.barometric_altitude
            }
        }
    
    def shutdown(self):
        """Enhanced shutdown procedure"""
        print("🛰️ Shutting down enhanced MAVLink handler...")
        self.running = False
        
        if self.mavlink_thread and self.mavlink_thread.is_alive():
            self.mavlink_thread.join(timeout=3.0)
            
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=3.0)
            
        if self.mavlink_connection:
            try:
                self.mavlink_connection.close()
            except:
                pass
        
        self.connected = False

# ==============================================================================
# ULTRA-SMOOTH SERVO CONTROLLER WITH ADVANCED FILTERING
# ==============================================================================

class UltraSmoothServoController:
    def __init__(self, logger=None):
        self.logger = logger
        
        # Enhanced I2C setup
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)
        self.pca = adafruit_pca9685.PCA9685(self.i2c)
        self.pca.frequency = 60
        
        self.pan_servo = servo.Servo(self.pca.channels[0], min_pulse=500, max_pulse=2500)
        self.tilt_servo = servo.Servo(self.pca.channels[1], min_pulse=500, max_pulse=2500)
        
        # Enhanced position tracking
        self.current_pan = 90.0
        self.current_tilt = 90.0 - CAMERA_TILT_OFFSET
        self.target_pan = self.current_pan
        self.target_tilt = self.current_tilt
        
        # Ultra-smooth filtering
        self.smooth_pan = self.current_pan
        self.smooth_tilt = self.current_tilt
        self.exponential_pan = self.current_pan
        self.exponential_tilt = self.current_tilt
        
        # Velocity and acceleration tracking
        self.velocity_pan = 0.0
        self.velocity_tilt = 0.0
        self.acceleration_pan = 0.0
        self.acceleration_tilt = 0.0
        self.last_velocity_pan = 0.0
        self.last_velocity_tilt = 0.0
        self.last_update_time = time.time()
        
        # Multi-stage filtering
        self.position_filter_pan = deque(maxlen=5)
        self.position_filter_tilt = deque(maxlen=5)
        self.velocity_filter_pan = deque(maxlen=3)
        self.velocity_filter_tilt = deque(maxlen=3)
        
        # Kalman filters for ultra-smooth movement
        self.pan_kalman = self._create_kalman_filter()
        self.tilt_kalman = self._create_kalman_filter()
        
        # Command processing
        self.command_queue = Queue(maxsize=2)
        self.running = True
        self.servo_thread = threading.Thread(target=self._ultra_smooth_servo_worker, daemon=True)
        
        # Initialize servos
        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        time.sleep(0.2)  # Allow servos to reach position
        
        # Initialize filters
        for _ in range(5):
            self.position_filter_pan.append(self.current_pan)
            self.position_filter_tilt.append(self.current_tilt)
        
        self.servo_thread.start()
        
        if self.logger:
            self.logger.log_event('servo_init', 'Ultra-smooth servo controller initialized')
        
        print("🎯 Ultra-Smooth Servo Controller initialized with advanced filtering")
    
    def _create_kalman_filter(self):
        """Create Kalman filter for ultra-smooth servo movement"""
        class ServoKalmanFilter:
            def __init__(self):
                self.x = 90.0    # Initial position
                self.P = 1.0     # Uncertainty
                self.Q = 0.01    # Very low process noise for smooth movement
                self.R = 0.1     # Low measurement noise
                
            def update(self, measurement):
                # Prediction step
                self.P = self.P + self.Q
                
                # Update step
                K = self.P / (self.P + self.R)
                self.x = self.x + K * (measurement - self.x)
                self.P = (1 - K) * self.P
                
                return self.x
        
        return ServoKalmanFilter()
    
    def _ultra_smooth_servo_worker(self):
        """Ultra-smooth servo worker with advanced filtering"""
        while self.running:
            try:
                command = None
                try:
                    command = self.command_queue.get(timeout=0.005)
                except Empty:
                    current_time = time.time()
                    dt = current_time - self.last_update_time
                    
                    if dt > 0.003:  # 333Hz maximum update rate
                        self._smooth_interpolate(dt)
                        self.last_update_time = current_time
                    continue
                
                if command is None:
                    break
                
                target_pan, target_tilt = command
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                # Update targets
                self.target_pan = target_pan
                self.target_tilt = target_tilt
                
                # Multi-stage smoothing
                self._apply_multi_stage_smoothing(target_pan, target_tilt, dt)
                
                self.last_update_time = current_time
                
            except Exception as e:
                print(f"Ultra-smooth servo thread error: {e}")
                time.sleep(0.001)
    
    def _apply_multi_stage_smoothing(self, target_pan, target_tilt, dt):
        """Apply multiple stages of smoothing for ultra-smooth movement"""
        
        # Stage 1: Exponential moving average
        self.exponential_pan = (EXPONENTIAL_SMOOTH_ALPHA * target_pan + 
                               (1 - EXPONENTIAL_SMOOTH_ALPHA) * self.exponential_pan)
        self.exponential_tilt = (EXPONENTIAL_SMOOTH_ALPHA * target_tilt + 
                                (1 - EXPONENTIAL_SMOOTH_ALPHA) * self.exponential_tilt)
        
        # Stage 2: Velocity limiting
        velocity_limited_pan, velocity_limited_tilt = self._apply_velocity_limiting(
            self.exponential_pan, self.exponential_tilt, dt
        )
        
        # Stage 3: Kalman filtering
        kalman_pan = self.pan_kalman.update(velocity_limited_pan)
        kalman_tilt = self.tilt_kalman.update(velocity_limited_tilt)
        
        # Stage 4: Final smoothing with S-curve
        final_pan, final_tilt = self._apply_s_curve_smoothing(kalman_pan, kalman_tilt)
        
        # Execute movement
        self._execute_smooth_movement(final_pan, final_tilt)
    
    def _apply_velocity_limiting(self, target_pan, target_tilt, dt):
        """Apply velocity and acceleration limiting for natural movement"""
        if dt <= 0:
            return target_pan, target_tilt
        
        # Calculate desired velocities
        desired_vel_pan = (target_pan - self.current_pan) / dt
        desired_vel_tilt = (target_tilt - self.current_tilt) / dt
        
        # Limit velocities
        limited_vel_pan = max(-VELOCITY_LIMIT, min(VELOCITY_LIMIT, desired_vel_pan))
        limited_vel_tilt = max(-VELOCITY_LIMIT, min(VELOCITY_LIMIT, desired_vel_tilt))
        
        # Apply acceleration limiting
        if dt > 0:
            max_accel_pan = ACCELERATION_LIMIT * dt
            max_accel_tilt = ACCELERATION_LIMIT * dt
            
            vel_change_pan = limited_vel_pan - self.velocity_pan
            vel_change_tilt = limited_vel_tilt - self.velocity_tilt
            
            vel_change_pan = max(-max_accel_pan, min(max_accel_pan, vel_change_pan))
            vel_change_tilt = max(-max_accel_tilt, min(max_accel_tilt, vel_change_tilt))
            
            self.velocity_pan += vel_change_pan
            self.velocity_tilt += vel_change_tilt
        
        # Calculate limited positions
        limited_pan = self.current_pan + self.velocity_pan * dt
        limited_tilt = self.current_tilt + self.velocity_tilt * dt
        
        return limited_pan, limited_tilt
    
    def _apply_s_curve_smoothing(self, target_pan, target_tilt):
        """Apply S-curve smoothing for natural acceleration/deceleration"""
        def s_curve(current, target, factor=0.3):
            diff = target - current
            if abs(diff) < 0.1:
                return target
            
            # S-curve formula for smooth acceleration
            normalized = min(abs(diff) / MAX_STEP_SIZE, 1.0)
            smoothed = normalized * normalized * (3.0 - 2.0 * normalized)
            
            return current + (diff * factor * smoothed)
        
        smooth_pan = s_curve(self.current_pan, target_pan)
        smooth_tilt = s_curve(self.current_tilt, target_tilt)
        
        return smooth_pan, smooth_tilt
    
    def _smooth_interpolate(self, dt):
        """Smooth interpolation for continuous movement"""
        if abs(self.target_pan - self.current_pan) > 0.05 or abs(self.target_tilt - self.current_tilt) > 0.05:
            # Progressive smoothing factor
            smoothing = min(0.9, SMOOTHING_FACTOR * (1 + dt * 5))
            
            new_pan = self.current_pan + (self.target_pan - self.current_pan) * smoothing
            new_tilt = self.current_tilt + (self.target_tilt - self.current_tilt) * smoothing
            
            self._execute_smooth_movement(new_pan, new_tilt)
    
    def _execute_smooth_movement(self, pan_angle, tilt_angle):
        """Execute movement with final filtering and validation"""
        # Add to position filters
        self.position_filter_pan.append(pan_angle)
        self.position_filter_tilt.append(pan_angle)
        
        # Apply median filtering for final smoothness
        if len(self.position_filter_pan) >= 3:
            filtered_pan = statistics.median(list(self.position_filter_pan)[-3:])
            filtered_tilt = statistics.median(list(self.position_filter_tilt)[-3:])
        else:
            filtered_pan, filtered_tilt = pan_angle, tilt_angle
        
        # Check movement threshold
        pan_diff = abs(filtered_pan - self.current_pan)
        tilt_diff = abs(filtered_tilt - self.current_tilt)
        
        if pan_diff > 0.02 or tilt_diff > 0.02:  # Very small threshold for ultra-smooth movement
            try:
                # Clamp to safe ranges
                safe_pan = max(10, min(170, filtered_pan))
                safe_tilt = max(10, min(170, filtered_tilt))
                
                # Update servos simultaneously
                if pan_diff > 0.02:
                    self.pan_servo.angle = safe_pan
                    self.current_pan = safe_pan
                
                if tilt_diff > 0.02:
                    self.tilt_servo.angle = safe_tilt
                    self.current_tilt = safe_tilt
                
                # Minimal delay for servo response
                time.sleep(0.002)
                
            except Exception as e:
                print(f"Servo movement error: {e}")
    
    def move_to_ultra_smooth(self, pan_angle, tilt_angle):
        """Ultra-smooth move command with advanced filtering"""
        # Clamp angles to safe ranges
        pan_angle = max(10, min(170, pan_angle))
        tilt_angle = max(10, min(170, tilt_angle))
        
        # Clear queue for immediate response
        try:
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            
            # Add new command
            self.command_queue.put_nowait((pan_angle, tilt_angle))
            
        except:
            # If queue is full, force update
            self.target_pan = pan_angle
            self.target_tilt = tilt_angle
    
    def get_enhanced_state(self):
        """Get comprehensive servo state"""
        return {
            'current_pan': self.current_pan,
            'current_tilt': self.current_tilt,
            'target_pan': self.target_pan,
            'target_tilt': self.target_tilt,
            'smooth_pan': self.smooth_pan,
            'smooth_tilt': self.smooth_tilt,
            'exponential_pan': self.exponential_pan,
            'exponential_tilt': self.exponential_tilt,
            'velocity_pan': self.velocity_pan,
            'velocity_tilt': self.velocity_tilt,
            'acceleration_pan': self.acceleration_pan,
            'acceleration_tilt': self.acceleration_tilt,
            'queue_size': self.command_queue.qsize()
        }
    
    def shutdown(self):
        """Enhanced shutdown procedure"""
        if self.logger:
            self.logger.log_event('servo_shutdown', 'Ultra-smooth servo controller shutting down')
        
        self.running = False
        
        try:
            self.command_queue.put_nowait(None)
        except:
            pass
        
        if self.servo_thread.is_alive():
            self.servo_thread.join(timeout=2.0)
        
        try:
            self.pan_servo.angle = 90
            self.tilt_servo.angle = 90
            time.sleep(0.5)
        except:
            pass

# ==============================================================================
# ENHANCED SMOOTH POSITION PREDICTOR
# ==============================================================================

class SmoothPositionPredictor:
    def __init__(self):
        self.position_history = deque(maxlen=8)
        self.time_history = deque(maxlen=8)
        self.velocity_history = deque(maxlen=5)
        self.smoothed_velocity = (0.0, 0.0)
        
    def add_position(self, x, y, timestamp):
        """Add position with enhanced velocity calculation"""
        self.position_history.append((x, y))
        self.time_history.append(timestamp)
        
        # Calculate smoothed velocity
        if len(self.position_history) >= 2:
            self._update_smoothed_velocity()
    
    def _update_smoothed_velocity(self):
        """Calculate smoothed velocity using multiple data points"""
        velocities = []
        
        # Calculate velocities from recent positions
        for i in range(1, min(4, len(self.position_history))):
            pos_prev = self.position_history[-(i+1)]
            pos_curr = self.position_history[-i]
            time_prev = self.time_history[-(i+1)]
            time_curr = self.time_history[-i]
            
            dt = time_curr - time_prev
            if dt > 0:
                vx = (pos_curr[0] - pos_prev[0]) / dt
                vy = (pos_curr[1] - pos_prev[1]) / dt
                velocities.append((vx, vy))
        
        if velocities:
            # Exponential weighted average of velocities
            weights = [0.5, 0.3, 0.2][:len(velocities)]
            total_weight = sum(weights)
            
            avg_vx = sum(v[0] * w for v, w in zip(velocities, weights)) / total_weight
            avg_vy = sum(v[1] * w for v, w in zip(velocities, weights)) / total_weight
            
            # Apply smoothing to velocity
            alpha = 0.7
            self.smoothed_velocity = (
                alpha * avg_vx + (1 - alpha) * self.smoothed_velocity[0],
                alpha * avg_vy + (1 - alpha) * self.smoothed_velocity[1]
            )
    
    def predict_smooth_position(self, dt=0.033):
        """Predict next position with enhanced smoothing"""
        if not self.position_history:
            return None
        
        last_pos = self.position_history[-1]
        
        if len(self.position_history) < 3:
            return last_pos
        
        # Use smoothed velocity for prediction
        vx, vy = self.smoothed_velocity
        
        # Apply motion prediction factor for stability
        pred_x = last_pos[0] + vx * dt * MOTION_PREDICTION_FACTOR
        pred_y = last_pos[1] + vy * dt * MOTION_PREDICTION_FACTOR
        
        return pred_x, pred_y

# ==============================================================================
# ENHANCED DISTANCE CALCULATOR WITH SMOOTH FILTERING
# ==============================================================================

class SmoothDistanceCalculator:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Enhanced focal length calculation
        self.focal_length_x = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        
        # Advanced smoothing filters
        self.distance_history = deque(maxlen=15)  # Increased for better smoothing
        self.kalman_filter = self._initialize_kalman_filter()
        self.exponential_filter = self._initialize_exponential_filter()
        
        # Multi-method estimation
        self.estimation_methods = ['height', 'width', 'area', 'aspect_ratio']
        self.method_weights = {'height': 0.4, 'width': 0.3, 'area': 0.2, 'aspect_ratio': 0.1}
        
        # Quality tracking
        self.accuracy_history = deque(maxlen=50)
        self.confidence_threshold = 0.7
        
    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for distance smoothing"""
        class DistanceKalmanFilter:
            def __init__(self):
                self.x = 5.0     # Initial distance estimate
                self.P = 10.0    # Initial uncertainty
                self.Q = 0.05    # Very low process noise for smooth distance
                self.R = 0.3     # Measurement noise
            
            def update(self, measurement):
                # Prediction step
                self.P = self.P + self.Q
                
                # Update step
                K = self.P / (self.P + self.R)
                self.x = self.x + K * (measurement - self.x)
                self.P = (1 - K) * self.P
                
                return self.x
        
        return DistanceKalmanFilter()
    
    def _initialize_exponential_filter(self):
        """Initialize exponential filter for distance smoothing"""
        class ExponentialFilter:
            def __init__(self, alpha=0.3):
                self.alpha = alpha
                self.value = 5.0
                
            def update(self, measurement):
                self.value = self.alpha * measurement + (1 - self.alpha) * self.value
                return self.value
        
        return ExponentialFilter()
    
    def calculate_smooth_distance(self, bbox, confidence=1.0):
        """Calculate distance with enhanced smoothing"""
        bbox_width_pixels = bbox.width() * self.frame_width
        bbox_height_pixels = bbox.height() * self.frame_height
        bbox_area_pixels = bbox_width_pixels * bbox_height_pixels
        
        distances = {}
        
        # Method 1: Height-based distance (most reliable)
        if bbox_height_pixels > 10:
            distances['height'] = (AVERAGE_PERSON_HEIGHT * self.focal_length_y) / bbox_height_pixels
        
        # Method 2: Width-based distance
        if bbox_width_pixels > 5:
            distances['width'] = (AVERAGE_PERSON_WIDTH * self.focal_length_x) / bbox_width_pixels
        
        # Method 3: Area-based distance
        if bbox_area_pixels > 50:
            reference_area = AVERAGE_PERSON_HEIGHT * AVERAGE_PERSON_WIDTH
            pixel_area_at_1m = reference_area * self.focal_length_x * self.focal_length_y
            distances['area'] = math.sqrt(pixel_area_at_1m / bbox_area_pixels)
        
        # Method 4: Aspect ratio correction
        aspect_ratio = bbox_width_pixels / max(bbox_height_pixels, 1)
        expected_aspect = AVERAGE_PERSON_WIDTH / AVERAGE_PERSON_HEIGHT
        if 'height' in distances:
            aspect_correction = expected_aspect / max(aspect_ratio, 0.1)
            distances['aspect_ratio'] = distances['height'] * aspect_correction
        
        # Weighted combination
        raw_distance = self._combine_distance_estimates(distances, confidence)
        
        # Apply multi-stage smoothing
        exponential_distance = self.exponential_filter.update(raw_distance)
        kalman_distance = self.kalman_filter.update(exponential_distance)
        
        # Add to history
        self.distance_history.append(kalman_distance)
        
        # Final smoothed distance
        return self._get_final_smoothed_distance()
    
    def _combine_distance_estimates(self, distances, detection_confidence):
        """Combine multiple distance estimates with adaptive weighting"""
        if not distances:
            return 5.0
        
        # Adaptive weighting based on confidence
        adaptive_weights = self.method_weights.copy()
        
        if detection_confidence > 0.8:
            adaptive_weights['height'] *= 1.2
            adaptive_weights['width'] *= 0.9
        
        total_weight = 0
        weighted_sum = 0
        
        for method, distance in distances.items():
            if method in adaptive_weights and 0.5 < distance < 50:
                weight = adaptive_weights[method]
                weighted_sum += distance * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return distances.get('height', distances.get('width', 5.0))
    
    def _get_final_smoothed_distance(self):
        """Get final smoothed distance with outlier removal"""
        if not self.distance_history:
            return 0.0
        
        if len(self.distance_history) < 3:
            return self.distance_history[-1]
        
        # Remove outliers using IQR method
        distances = list(self.distance_history)
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered_distances = [d for d in distances if lower_bound <= d <= upper_bound]
        else:
            filtered_distances = distances
        
        if not filtered_distances:
            filtered_distances = distances
        
        # Weighted moving average with recent bias
        weights = np.exp(np.linspace(-1, 0, len(filtered_distances)))
        weights /= weights.sum()
        
        return np.average(filtered_distances, weights=weights)
    
    def update_frame_size(self, width, height):
        """Update frame size and recalculate focal lengths"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.focal_length_x = width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
            self.focal_length_y = height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))

# ==============================================================================
# ULTRA-SMOOTH TRACKER WITH ADVANCED FILTERING
# ==============================================================================

class UltraSmoothTracker:
    def __init__(self, servo_controller, logger=None, mavlink_handler=None):
        self.servo = servo_controller
        self.logger = logger
        self.mavlink_handler = mavlink_handler
        
        # Enhanced frame properties
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        # Enhanced distance calculator
        self.distance_calculator = SmoothDistanceCalculator(self.frame_width, self.frame_height)
        
        # Enhanced tracking state
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        self.lock_on_target = False
        self.tracking_quality = "NONE"
        
        # Position tracking with smoothing
        self.current_distance = 0.0
        self.current_3d_position = (0.0, 0.0, 0.0, 0.0)
        self.last_3d_position = (0.0, 0.0, 0.0, 0.0)
        
        # Enhanced smoothing with multiple filters
        self.pan_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.tilt_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.position_predictor = SmoothPositionPredictor()
        
        # Exponential filters for servo angles
        self.pan_exponential_filter = self._create_exponential_filter()
        self.tilt_exponential_filter = self._create_exponential_filter()
        
        # Adaptive parameters with distance-based scaling
        self.dynamic_sensitivity = {'pan': PAN_SENSITIVITY, 'tilt': TILT_SENSITIVITY}
        self.dynamic_dead_zone = DEAD_ZONE
        self.dynamic_smoothing = SMOOTHING_FACTOR
        
        # Performance metrics
        self.frames_processed = 0
        self.successful_tracks = 0
        self.average_confidence = 0.0
        
        print("🎯 Ultra-Smooth Tracker with advanced filtering initialized")
    
    def _create_exponential_filter(self, alpha=0.4):
        """Create exponential filter for angle smoothing"""
        class AngleExponentialFilter:
            def __init__(self, alpha):
                self.alpha = alpha
                self.value = 90.0
                
            def update(self, measurement):
                self.value = self.alpha * measurement + (1 - self.alpha) * self.value
                return self.value
        
        return AngleExponentialFilter(alpha)
    
    def update_frame_properties(self, width, height):
        """Update frame properties with enhanced handling"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            
            # Update distance calculator
            self.distance_calculator.update_frame_size(width, height)
            
            # Adjust sensitivity based on resolution
            resolution_factor = math.sqrt((width * height) / (640 * 480))
            self.dynamic_sensitivity['pan'] = PAN_SENSITIVITY * resolution_factor
            self.dynamic_sensitivity['tilt'] = TILT_SENSITIVITY * resolution_factor
    
    def track_person_ultra_smooth(self, bbox, confidence, frame_count):
        """Ultra-smooth person tracking with advanced filtering"""
        current_time = time.time()
        self.frames_processed += 1
        
        # Enhanced distance calculation with smoothing
        self.current_distance = self.distance_calculator.calculate_smooth_distance(bbox, confidence)
        
        # Get current servo state
        servo_state = self.servo.get_enhanced_state()
        current_pan = servo_state['current_pan']
        current_tilt = servo_state['current_tilt']
        
        # Calculate target center with sub-pixel precision
        center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
        center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
        
        # Add to position predictor
        self.position_predictor.add_position(center_x, center_y, current_time)
        
        # Use prediction for smoother tracking
        predicted_position = self.position_predictor.predict_smooth_position()
        if predicted_position and confidence > 0.6:
            center_x, center_y = predicted_position
        
        # Calculate errors with enhanced precision
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        
        # Dynamic parameters based on distance and confidence
        distance_factor = max(0.3, min(2.5, DISTANCE_SMOOTH_FACTOR / max(self.current_distance, 0.5)))
        confidence_factor = max(0.5, min(1.5, confidence + 0.2))
        
        # Update dynamic dead zone
        self.dynamic_dead_zone = DEAD_ZONE * distance_factor / confidence_factor
        
        # Enhanced movement calculation with ultra-smooth filtering
        if abs(error_x) > self.dynamic_dead_zone or abs(error_y) > self.dynamic_dead_zone:
            
            # Adaptive sensitivity with multiple factors
            adaptive_pan_sens = self.dynamic_sensitivity['pan'] * distance_factor * confidence_factor
            adaptive_tilt_sens = self.dynamic_sensitivity['tilt'] * distance_factor * confidence_factor
            
            # Calculate base adjustments
            pan_adjustment = -error_x * (adaptive_pan_sens / self.frame_width)
            tilt_adjustment = error_y * (adaptive_tilt_sens / self.frame_height)
            
            # Apply step size limits
            pan_adjustment = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, pan_adjustment))
            tilt_adjustment = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, tilt_adjustment))
            
            # Calculate target angles
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            # Multi-stage smoothing process
            
            # Stage 1: Exponential filtering
            smooth_pan = self.pan_exponential_filter.update(target_pan)
            smooth_tilt = self.tilt_exponential_filter.update(target_tilt)
            
            # Stage 2: History-based smoothing
            self.pan_history.append(smooth_pan)
            self.tilt_history.append(smooth_tilt)
            
            # Stage 3: Weighted average with recent bias
            if len(self.pan_history) >= 3:
                weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0][:len(self.pan_history)])
                weights = weights / weights.sum()
                
                final_pan = np.average(list(self.pan_history), weights=weights)
                final_tilt = np.average(list(self.tilt_history), weights=weights)
            else:
                final_pan, final_tilt = smooth_pan, smooth_tilt
            
            # Stage 4: Distance-based fine-tuning
            if self.current_distance < 3.0:  # Close targets need extra smoothing
                extra_smooth_factor = 0.7
                final_pan = current_pan + (final_pan - current_pan) * extra_smooth_factor
                final_tilt = current_tilt + (final_tilt - current_tilt) * extra_smooth_factor
            
            # Ultra-smooth servo command
            self.servo.move_to_ultra_smooth(final_pan, final_tilt)
            
            # Update tracking state
            if not self.lock_on_target:
                self.lock_on_target = True
                self.tracking_quality = "LOCKED"
                if self.logger:
                    self.logger.log_event('target_lock', f'Ultra-smooth lock at {self.current_distance:.2f}m')
                print(f"🎯 Ultra-smooth target lock: {self.current_distance:.2f}m, Conf: {confidence:.3f}")
        
        # Update tracking metrics
        self.last_detection_time = current_time
        self.target_lost_frames = 0
        self.successful_tracks += 1
        self.average_confidence = (self.average_confidence * (self.successful_tracks - 1) + confidence) / self.successful_tracks
        
        # GPS waypoint generation if available
        if self.mavlink_handler and self.current_distance >= MIN_DISTANCE_FOR_GPS:
            x, y, z_rel, z_abs = self._calculate_3d_position(current_pan, current_tilt)
            self.current_3d_position = (x, y, z_rel, z_abs)
        
        # Enhanced logging
        if self.logger:
            self._log_enhanced_tracking_data(frame_count, servo_state, confidence)
    
    def _calculate_3d_position(self, pan_angle, tilt_angle):
        """Calculate 3D position with enhanced accuracy"""
        pan_rad = math.radians(pan_angle - 90)
        actual_tilt_angle = tilt_angle + CAMERA_TILT_OFFSET
        tilt_rad = math.radians(90 - actual_tilt_angle)
        
        horizontal_distance = self.current_distance * math.cos(tilt_rad)
        
        x = horizontal_distance * math.sin(pan_rad)
        y = horizontal_distance * math.cos(pan_rad)
        z_relative = self.current_distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT
        
        # Get altitude data for absolute positioning
        z_absolute = z_relative
        if self.mavlink_handler:
            status = self.mavlink_handler.get_enhanced_status()
            if status.get('altitude_relative', 0) > 0:
                z_absolute = z_relative + status['altitude_relative']
        
        return x, y, z_relative, z_absolute
    
    def handle_lost_target_smooth(self, frame_count):
        """Enhanced lost target handling with smooth search"""
        self.target_lost_frames += 1
        current_time = time.time()
        
        # Gradual transition to search mode
        if self.target_lost_frames == 1:
            # First frame lost - maintain position briefly
            pass
        elif 2 <= self.target_lost_frames <= 5:
            # Gentle search around last known position
            servo_state = self.servo.get_enhanced_state()
            search_radius = 2.0 * self.target_lost_frames
            
            # Smooth oscillation pattern
            angle_offset = math.sin(current_time * 2) * search_radius
            new_pan = servo_state['current_pan'] + angle_offset
            
            new_pan = max(10, min(170, new_pan))
            self.servo.move_to_ultra_smooth(new_pan, servo_state['current_tilt'])
        
        elif self.target_lost_frames > 15:
            # Wider search pattern with smooth movement
            if self.lock_on_target:
                self.lock_on_target = False
                self.tracking_quality = "SEARCHING"
                if self.logger:
                    self.logger.log_event('target_lost', 'Ultra-smooth search mode activated')
                print("🔍 Target lost - ultra-smooth search mode activated")
        
        # Enhanced logging for lost target
        if self.logger:
            servo_state = self.servo.get_enhanced_state()
            self._log_enhanced_tracking_data(frame_count, servo_state, 0.0, False)
    
    def _log_enhanced_tracking_data(self, frame_count, servo_state, confidence, person_detected=True):
        """Enhanced logging with comprehensive tracking data"""
        if not self.logger:
            return
        
        distance_data = {
            'distance': self.current_distance,
            'x_position': self.current_3d_position[0],
            'y_position': self.current_3d_position[1],
            'z_position_relative': self.current_3d_position[2],
            'z_position_absolute': self.current_3d_position[3],
            'tracking_quality': self.tracking_quality,
            'dynamic_dead_zone': self.dynamic_dead_zone,
            'dynamic_smoothing': self.dynamic_smoothing,
            'prediction_used': True,  # Always using prediction in smooth mode
            'frames_processed': self.frames_processed,
            'successful_tracks': self.successful_tracks
        }
        
        self.logger.log_enhanced_frame_data(
            frame_count, servo_state, confidence, person_detected,
            self.is_tracking_active(), self.target_lost_frames, distance_data
        )
    
    def is_tracking_active(self):
        """Enhanced tracking active check"""
        time_since_detection = time.time() - self.last_detection_time
        return time_since_detection < DETECTION_TIMEOUT and self.lock_on_target
    
    def get_enhanced_tracking_info(self):
        """Get comprehensive tracking information"""
        return {
            'distance': self.current_distance,
            'position_3d': self.current_3d_position,
            'tracking_quality': self.tracking_quality,
            'lock_on_target': self.lock_on_target,
            'frames_processed': self.frames_processed,
            'successful_tracks': self.successful_tracks,
            'average_confidence': self.average_confidence,
            'dynamic_dead_zone': self.dynamic_dead_zone,
            'dynamic_smoothing': self.dynamic_smoothing,
            'target_lost_frames': self.target_lost_frames
        }

# ==============================================================================
# ROBUST AUTOMATIC CALIBRATION SYSTEM (Keeping original implementation)
# ==============================================================================

class RobustAutoCalibration:
    def __init__(self, logger=None):
        self.logger = logger
        self.calibration_active = AUTO_CALIBRATION_ENABLED
        self.calibration_complete = False
        
        self.measurements = []
        self.distance_measurements = []
        self.angle_measurements = []
        self.confidence_measurements = []
        
        self.start_time = time.time()
        self.samples_collected = 0
        self.required_samples = CALIBRATION_SAMPLES_REQUIRED
        self.calibration_timeout = CALIBRATION_TIMEOUT
        
        self.focal_length_estimates = []
        self.height_pixel_ratios = []
        
        self.current_phase = "INITIALIZATION"
        self.phases = ["INITIALIZATION", "DATA_COLLECTION", "ANALYSIS", "VALIDATION", "COMPLETE"]
        self.phase_start_time = time.time()
        
        print("🎯 Robust Auto-Calibration System Initialized")
        if self.calibration_active:
            print(f"   Target samples: {self.required_samples}")
            print(f"   Timeout: {self.calibration_timeout}s")
    
    def add_calibration_measurement(self, bbox, distance, confidence, frame_height):
        """Add measurement for automatic calibration"""
        if not self.calibration_active or self.calibration_complete:
            return
        
        current_time = time.time()
        
        if current_time - self.start_time > self.calibration_timeout:
            self._force_complete_calibration()
            return
        
        if self.current_phase == "INITIALIZATION":
            self._advance_phase("DATA_COLLECTION")
        
        person_height_pixels = bbox.height() * frame_height
        
        if person_height_pixels < 30 or person_height_pixels > frame_height * 0.8:
            return
        
        if confidence < 0.4:
            return
        
        if distance < 1.0 or distance > 15.0:
            return
        
        measurement = {
            'timestamp': current_time,
            'height_pixels': person_height_pixels,
            'distance': distance,
            'confidence': confidence,
            'bbox_width': bbox.width() * frame_height,
            'bbox_area': bbox.width() * bbox.height() * frame_height * frame_height
        }
        
        self.measurements.append(measurement)
        self.samples_collected += 1
        
        focal_estimate = (person_height_pixels * distance) / AVERAGE_PERSON_HEIGHT
        self.focal_length_estimates.append(focal_estimate)
        
        if self.samples_collected % 5 == 0:
            progress = (self.samples_collected / self.required_samples) * 100
            print(f"📊 Calibration progress: {self.samples_collected}/{self.required_samples} ({progress:.1f}%)")
        
        if self.samples_collected >= self.required_samples:
            self._advance_phase("ANALYSIS")
            self._analyze_calibration_data()
    
    def _advance_phase(self, new_phase):
        """Advance to next calibration phase"""
        self.current_phase = new_phase
        self.phase_start_time = time.time()
        print(f"🔄 Calibration phase: {new_phase}")
        
        if self.logger:
            self.logger.log_event('calibration_phase', f'Advanced to {new_phase}')
    
    def _analyze_calibration_data(self):
        """Analyze collected calibration data"""
        if len(self.focal_length_estimates) < 5:
            print("⚠️ Insufficient calibration data")
            self._force_complete_calibration()
            return
        
        focal_estimates = np.array(self.focal_length_estimates)
        
        Q1 = np.percentile(focal_estimates, 25)
        Q3 = np.percentile(focal_estimates, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered_estimates = focal_estimates[
            (focal_estimates >= lower_bound) & (focal_estimates <= upper_bound)
        ]
        
        if len(filtered_estimates) < 3:
            print("⚠️ Too many outliers in calibration data")
            self._force_complete_calibration()
            return
        
        final_focal_length = np.median(filtered_estimates)
        focal_std = np.std(filtered_estimates)
        focal_confidence = 1.0 - (focal_std / final_focal_length)
        
        print(f"\n🎯 AUTOMATIC CALIBRATION ANALYSIS:")
        print(f"   Samples analyzed: {len(filtered_estimates)}")
        print(f"   Focal length: {final_focal_length:.1f} pixels")
        print(f"   Standard deviation: {focal_std:.2f}")
        print(f"   Confidence: {focal_confidence:.3f}")
        
        self._advance_phase("VALIDATION")
        
        if focal_confidence >= CALIBRATION_ACCURACY_THRESHOLD:
            self._complete_calibration(final_focal_length, focal_confidence)
        else:
            print(f"⚠️ Calibration confidence ({focal_confidence:.3f}) below threshold ({CALIBRATION_ACCURACY_THRESHOLD})")
            self._extend_calibration()
    
    def _complete_calibration(self, focal_length, confidence):
        """Complete the calibration process"""
        self._advance_phase("COMPLETE")
        self.calibration_complete = True
        self.calibration_active = False
        
        print(f"\n✅ AUTOMATIC CALIBRATION COMPLETE!")
        print(f"   Final focal length: {focal_length:.1f} pixels")
        print(f"   Confidence level: {confidence:.3f}")
        print(f"   Total time: {time.time() - self.start_time:.1f}s")
        print(f"   Recommended update: FOCAL_LENGTH_PIXELS = {focal_length:.1f}")
        
        if self.logger:
            self.logger.log_event('calibration_complete', 
                                f'Focal length: {focal_length:.1f}, Confidence: {confidence:.3f}')
        
        return focal_length
    
    def _extend_calibration(self):
        """Extend calibration for more samples"""
        self.required_samples += 10
        self.calibration_timeout += 30
        print(f"🔄 Extending calibration: target now {self.required_samples} samples")
        self._advance_phase("DATA_COLLECTION")
    
    def _force_complete_calibration(self):
        """Force complete calibration with available data"""
        if len(self.focal_length_estimates) > 0:
            focal_length = np.median(self.focal_length_estimates)
            confidence = max(0.5, 1.0 - (np.std(self.focal_length_estimates) / focal_length))
            self._complete_calibration(focal_length, confidence)
        else:
            print("❌ Calibration failed: no valid measurements")
            self.calibration_active = False
            self.calibration_complete = True
    
    def get_calibration_status(self):
        """Get current calibration status"""
        return {
            'active': self.calibration_active,
            'complete': self.calibration_complete,
            'phase': self.current_phase,
            'samples_collected': self.samples_collected,
            'required_samples': self.required_samples,
            'progress': min(100, (self.samples_collected / self.required_samples) * 100),
            'time_elapsed': time.time() - self.start_time,
            'timeout_remaining': max(0, self.calibration_timeout - (time.time() - self.start_time))
        }

# ==============================================================================
# ENHANCED DATA LOGGER
# ==============================================================================

class EnhancedDataLogger:
    def __init__(self, log_dir="servo_logs", gps_handler=None):
        self.gps_handler = gps_handler
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"smooth_servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"smooth_session_{timestamp}.json"
        self.gps_csv_file = self.log_dir / f"smooth_gps_points_{timestamp}.csv"
        
        # Enhanced CSV headers for smooth tracking
        self.csv_headers = [
            'timestamp', 'frame_count', 'pan_angle', 'tilt_angle', 'target_pan', 'target_tilt',
            'smooth_pan', 'smooth_tilt', 'exponential_pan', 'exponential_tilt',
            'pan_velocity', 'tilt_velocity', 'pan_acceleration', 'tilt_acceleration',
            'detection_confidence', 'person_detected', 'tracking_active', 'tracking_quality',
            'target_lost_frames', 'distance_meters', 'x_position', 'y_position', 
            'z_position_relative', 'z_position_absolute',
            'dynamic_dead_zone', 'dynamic_smoothing', 'prediction_used',
            'frames_processed', 'successful_tracks', 'servo_queue_size'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_headers)
        
        # Enhanced session data
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'enhanced_features': [
                'ultra_smooth_servos', 'multi_stage_filtering', 'predictive_tracking',
                'exponential_smoothing', 'kalman_filtering', 'velocity_limiting',
                'acceleration_control', 'distance_based_adaptation'
            ],
            'log_files': {
                'csv': str(self.csv_file),
                'json': str(self.json_file),
                'gps_csv': str(self.gps_csv_file) if self.gps_handler else None
            },
            'smoothing_parameters': {
                'smoothing_factor': SMOOTHING_FACTOR,
                'velocity_limit': VELOCITY_LIMIT,
                'acceleration_limit': ACCELERATION_LIMIT,
                'exponential_alpha': EXPONENTIAL_SMOOTH_ALPHA,
                'motion_prediction_factor': MOTION_PREDICTION_FACTOR,
                'distance_smooth_factor': DISTANCE_SMOOTH_FACTOR
            },
            'statistics': {
                'total_detections': 0,
                'total_movements': 0,
                'smooth_movements': 0,
                'min_distance': float('inf'),
                'max_distance': 0.0,
                'avg_distance': 0.0,
                'distance_samples': 0,
                'tracking_quality_distribution': {},
                'prediction_usage': 0,
                'calibration_samples': 0
            },
            'events': [],
            'performance_metrics': {
                'avg_processing_time': 0.0,
                'max_processing_time': 0.0,
                'smoothness_score': 0.0,
                'jitter_reduction': 0.0
            }
        }
        
        print(f"📊 Enhanced smooth data logging to: {self.log_dir}")
    
    def log_enhanced_frame_data(self, frame_count, servo_state, detection_confidence, 
                               person_detected, tracking_active, target_lost_frames, distance_data=None):
        """Enhanced frame data logging with smooth tracking metrics"""
        try:
            # Extract enhanced servo data
            pan_angle = servo_state.get('current_pan', 0)
            tilt_angle = servo_state.get('current_tilt', 0)
            target_pan = servo_state.get('target_pan', 0)
            target_tilt = servo_state.get('target_tilt', 0)
            smooth_pan = servo_state.get('smooth_pan', 0)
            smooth_tilt = servo_state.get('smooth_tilt', 0)
            exponential_pan = servo_state.get('exponential_pan', 0)
            exponential_tilt = servo_state.get('exponential_tilt', 0)
            pan_velocity = servo_state.get('velocity_pan', 0)
            tilt_velocity = servo_state.get('velocity_tilt', 0)
            pan_acceleration = servo_state.get('acceleration_pan', 0)
            tilt_acceleration = servo_state.get('acceleration_tilt', 0)
            servo_queue_size = servo_state.get('queue_size', 0)
            
            # Initialize default values
            distance = x_pos = y_pos = z_pos_rel = z_pos_abs = 0.0
            dynamic_dead_zone = dynamic_smoothing = 0.0
            prediction_used = False
            frames_processed = successful_tracks = 0
            tracking_quality = 'NONE'
            
            if distance_data:
                distance = distance_data.get('distance', 0.0)
                x_pos = distance_data.get('x_position', 0.0)
                y_pos = distance_data.get('y_position', 0.0)
                z_pos_rel = distance_data.get('z_position_relative', 0.0)
                z_pos_abs = distance_data.get('z_position_absolute', 0.0)
                dynamic_dead_zone = distance_data.get('dynamic_dead_zone', 0.0)
                dynamic_smoothing = distance_data.get('dynamic_smoothing', 0.0)
                prediction_used = distance_data.get('prediction_used', False)
                frames_processed = distance_data.get('frames_processed', 0)
                successful_tracks = distance_data.get('successful_tracks', 0)
                tracking_quality = distance_data.get('tracking_quality', 'NONE')
            
            # Update statistics
            stats = self.session_data['statistics']
            if person_detected:
                stats['total_detections'] += 1
                
            if abs(pan_velocity) > 1 or abs(tilt_velocity) > 1:
                stats['total_movements'] += 1
                
                # Count as smooth movement if velocity is within limits
                if abs(pan_velocity) <= VELOCITY_LIMIT and abs(tilt_velocity) <= VELOCITY_LIMIT:
                    stats['smooth_movements'] += 1
                
            if distance > 0:
                stats['min_distance'] = min(stats['min_distance'], distance)
                stats['max_distance'] = max(stats['max_distance'], distance)
                stats['distance_samples'] += 1
                stats['avg_distance'] = (
                    (stats['avg_distance'] * (stats['distance_samples'] - 1) + distance) /
                    stats['distance_samples']
                )
            
            # Track quality distribution
            if tracking_quality not in stats['tracking_quality_distribution']:
                stats['tracking_quality_distribution'][tracking_quality] = 0
            stats['tracking_quality_distribution'][tracking_quality] += 1
            
            if prediction_used:
                stats['prediction_usage'] += 1
            
            # Write to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), frame_count, pan_angle, tilt_angle, target_pan, target_tilt,
                    smooth_pan, smooth_tilt, exponential_pan, exponential_tilt,
                    pan_velocity, tilt_velocity, pan_acceleration, tilt_acceleration,
                    detection_confidence, person_detected, tracking_active, tracking_quality,
                    target_lost_frames, distance, x_pos, y_pos, z_pos_rel, z_pos_abs,
                    dynamic_dead_zone, dynamic_smoothing, prediction_used,
                    frames_processed, successful_tracks, servo_queue_size
                ])
                
        except Exception as e:
            print(f"Enhanced logging error: {e}")
    
    def log_event(self, event_type, description):
        """Enhanced event logging"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description
        }
        self.session_data['events'].append(event)
        print(f"📝 {event_type}: {description}")
    
    def finalize_enhanced_session(self):
        """Finalize enhanced session with comprehensive statistics"""
        self.session_data['end_time'] = datetime.now().isoformat()
        
        # Calculate smoothness metrics
        stats = self.session_data['statistics']
        if stats['total_movements'] > 0:
            smoothness_ratio = stats['smooth_movements'] / stats['total_movements']
            self.session_data['performance_metrics']['smoothness_score'] = smoothness_ratio
            
            # Calculate jitter reduction (estimated)
            if stats['prediction_usage'] > 0:
                jitter_reduction = min(0.9, stats['prediction_usage'] / stats['total_detections'])
                self.session_data['performance_metrics']['jitter_reduction'] = jitter_reduction
        
        # Add comprehensive system status
        if self.gps_handler:
            self.session_data['final_gps_status'] = self.gps_handler.get_enhanced_status()
        
        # Performance summary
        if stats['total_detections'] > 0:
            self.session_data['performance_summary'] = {
                'smoothness_ratio': stats['smooth_movements'] / max(1, stats['total_movements']),
                'prediction_usage_rate': stats['prediction_usage'] / max(1, stats['total_detections']),
                'average_distance': stats['avg_distance'],
                'distance_range': f"{stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m",
                'total_smooth_movements': stats['smooth_movements']
            }
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            # Print comprehensive summary
            print(f"\n📊 Enhanced Smooth Session Complete:")
            print(f"   Total Detections: {stats['total_detections']}")
            print(f"   Total Movements: {stats['total_movements']}")
            print(f"   Smooth Movements: {stats['smooth_movements']}")
            print(f"   Smoothness Ratio: {stats['smooth_movements'] / max(1, stats['total_movements']):.2f}")
            print(f"   Prediction Usage: {stats['prediction_usage']}")
            print(f"   Calibration Samples: {stats['calibration_samples']}")
            
            if stats['distance_samples'] > 0:
                print(f"   Distance Range: {stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m")
                print(f"   Average Distance: {stats['avg_distance']:.2f}m")
            
            print(f"   Quality Distribution: {stats['tracking_quality_distribution']}")
            
        except Exception as e:
            print(f"Enhanced session save error: {e}")

# ==============================================================================
# ENHANCED CALLBACK CLASS
# ==============================================================================

class EnhancedSmoothAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.processing_times = deque(maxlen=100)
        self.last_fps_update = time.time()
        self.fps = 0.0
        self.smoothness_metrics = {
            'velocity_history': deque(maxlen=50),
            'jitter_count': 0,
            'smooth_transitions': 0
        }
    
    def new_function(self):
        return "Enhanced Ultra-Smooth Tracking with Advanced Filtering: "
    
    def get_fps(self):
        return self.fps
    
    def update_fps(self):
        current_time = time.time()
        if current_time - self.last_fps_update > 1.0:
            self.fps = self.frame_counter / max(1, current_time - self.last_fps_update)
            self.frame_counter = 0
            self.last_fps_update = current_time
    
    def track_smoothness_metric(self, velocity_pan, velocity_tilt):
        """Track smoothness metrics for performance analysis"""
        velocity_magnitude = math.sqrt(velocity_pan**2 + velocity_tilt**2)
        self.smoothness_metrics['velocity_history'].append(velocity_magnitude)
        
        # Detect jitter (rapid velocity changes)
        if len(self.smoothness_metrics['velocity_history']) >= 3:
            recent_velocities = list(self.smoothness_metrics['velocity_history'])[-3:]
            velocity_variance = np.var(recent_velocities)
            
            if velocity_variance > 100:  # High variance indicates jitter
                self.smoothness_metrics['jitter_count'] += 1
            else:
                self.smoothness_metrics['smooth_transitions'] += 1

# ==============================================================================
# ENHANCED MAIN CALLBACK FUNCTION
# ==============================================================================

def enhanced_ultra_smooth_app_callback(pad, info, user_data):
    """Enhanced callback with ultra-smooth tracking"""
    start_time = time.time()
    
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    frame_count = user_data.get_count()
    user_data.frame_counter += 1
    user_data.update_fps()
    
    # Dynamic frame size detection
    if frame_count % 30 == 0:
        format, width, height = get_caps_from_pad(pad)
        if width and height:
            smooth_tracker.update_frame_properties(width, height)
    
    # Enhanced frame processing
    frame = None
    if user_data.use_frame:
        format, width, height = get_caps_from_pad(pad)
        if format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Enhanced detection processing with stability filtering
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Advanced detection filtering for smooth tracking
    valid_detections = []
    for detection in detections:
        if detection.get_label() == "person":
            confidence = detection.get_confidence()
            if confidence >= MIN_CONFIDENCE:
                bbox = detection.get_bbox()
                area = bbox.width() * bbox.height()
                
                # Enhanced quality metrics for smooth tracking
                aspect_ratio = bbox.width() / max(bbox.height(), 0.001)
                reasonable_aspect = 0.2 < aspect_ratio < 2.0
                reasonable_size = 0.002 < area < 0.4  # Slightly tighter bounds for stability
                center_bias = 1.0  # Prefer detections closer to center for stability
                
                # Calculate center distance bias
                bbox_center_x = bbox.xmin() + bbox.width() * 0.5
                bbox_center_y = bbox.ymin() + bbox.height() * 0.5
                center_distance = math.sqrt((bbox_center_x - 0.5)**2 + (bbox_center_y - 0.5)**2)
                center_bias = max(0.5, 1.0 - center_distance)
                
                if reasonable_aspect and reasonable_size:
                    stability_score = confidence * area * center_bias * (1.0 if 0.3 < aspect_ratio < 0.8 else 0.8)
                    
                    valid_detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center_bias': center_bias,
                        'stability_score': stability_score
                    })
    
    # Select most stable detection for smooth tracking
    best_detection = None
    if valid_detections:
        # Sort by stability score (prioritizes stable, centered detections)
        valid_detections.sort(key=lambda x: x['stability_score'], reverse=True)
        best_detection = valid_detections[0]
        
        # Additional stability check - reject if confidence dropped significantly
        if hasattr(smooth_tracker, 'last_confidence'):
            confidence_drop = smooth_tracker.last_confidence - best_detection['confidence']
            if confidence_drop > 0.3 and smooth_tracker.last_confidence > 0.7:
                # Significant confidence drop - maintain last position briefly
                best_detection = None
        
        if best_detection:
            smooth_tracker.last_confidence = best_detection['confidence']
    
    # Ultra-smooth tracking
    if best_detection:
        smooth_tracker.track_person_ultra_smooth(
            best_detection['bbox'], 
            best_detection['confidence'], 
            frame_count
        )
        
        # Track smoothness metrics
        servo_state = ultra_smooth_servo_controller.get_enhanced_state()
        user_data.track_smoothness_metric(
            servo_state['velocity_pan'], 
            servo_state['velocity_tilt']
        )
        
        # Enhanced calibration
        if auto_calibration.calibration_active:
            distance_info = smooth_tracker.get_enhanced_tracking_info()
            auto_calibration.add_calibration_measurement(
                best_detection['bbox'],
                distance_info['distance'],
                best_detection['confidence'],
                smooth_tracker.frame_height
            )
        
        # Enhanced progress reporting
        if frame_count % 90 == 0:  # Every 3 seconds at 30fps
            tracking_info = smooth_tracker.get_enhanced_tracking_info()
            smoothness_ratio = (user_data.smoothness_metrics['smooth_transitions'] / 
                              max(1, user_data.smoothness_metrics['smooth_transitions'] + 
                                  user_data.smoothness_metrics['jitter_count']))
            
            print(f"🏃 Ultra-Smooth Tracking: Conf {best_detection['confidence']:.3f}, "
                  f"Dist: {tracking_info['distance']:.2f}m, "
                  f"Quality: {tracking_info['tracking_quality']}, "
                  f"Smoothness: {smoothness_ratio:.2f}, "
                  f"FPS: {user_data.get_fps():.1f}")
    else:
        smooth_tracker.handle_lost_target_smooth(frame_count)
    
    # Enhanced frame visualization
    if user_data.use_frame and frame is not None:
        frame = enhanced_smooth_draw_overlay(frame, best_detection, frame_count)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Performance tracking
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    
    return Gst.PadProbeReturn.OK

def enhanced_smooth_draw_overlay(frame, best_detection, frame_count):
    """Enhanced overlay drawing with smooth tracking information"""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Draw enhanced crosshair with smoother lines
    cv2.line(frame, (center_x - 25, center_y), (center_x + 25, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 25), (0, 255, 255), 2)
    cv2.circle(frame, (center_x, center_y), 6, (0, 255, 255), -1)
    cv2.circle(frame, (center_x, center_y), 12, (0, 255, 255), 1)
    
    # Enhanced status display
    if best_detection:
        tracking_info = smooth_tracker.get_enhanced_tracking_info()
        distance = tracking_info['distance']
        quality = tracking_info['tracking_quality']
        
        # Calibration status
        if auto_calibration.calibration_active:
            cal_status = auto_calibration.get_calibration_status()
            cv2.putText(frame, f"CALIBRATION: {cal_status['phase']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Progress: {cal_status['progress']:.1f}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"ULTRA-SMOOTH TRACKING: {quality}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Smoothness indicators
        cv2.putText(frame, f"Smoothness: ULTRA", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Dead Zone: {tracking_info['dynamic_dead_zone']:.1f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Enhanced bounding box with stability indicator
        bbox = best_detection['bbox']
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int((bbox.xmin() + bbox.width()) * width)
        y2 = int((bbox.ymin() + bbox.height()) * height)
        
        # Color based on quality and stability
        if quality == "LOCKED":
            box_color = (0, 255, 0)
            stability_indicator = "STABLE"
        elif quality == "SEARCHING":
            box_color = (255, 255, 0)
            stability_indicator = "SEARCH"
        else:
            box_color = (255, 0, 0)
            stability_indicator = "UNSTABLE"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Stability score visualization
        stability_score = best_detection.get('stability_score', 0)
        cv2.putText(frame, f"{distance:.1f}m | {best_detection['confidence']:.3f} | {stability_indicator}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Center bias indicator (shows how centered the detection is)
        center_bias = best_detection.get('center_bias', 0)
        bias_color = (0, int(255 * center_bias), int(255 * (1 - center_bias)))
        cv2.circle(frame, (x1 + (x2-x1)//2, y1 + (y2-y1)//2), 8, bias_color, 2)
    
    # Enhanced servo state display with smoothness metrics
    servo_state = ultra_smooth_servo_controller.get_enhanced_state()
    cv2.putText(frame, f"Pan: {servo_state['current_pan']:.1f}° (S:{servo_state['smooth_pan']:.1f}°)", 
               (10, height - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Tilt: {servo_state['current_tilt']:.1f}° (S:{servo_state['smooth_tilt']:.1f}°)", 
               (10, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Vel: {servo_state['velocity_pan']:.1f}°/s, {servo_state['velocity_tilt']:.1f}°/s", 
               (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Accel: {servo_state['acceleration_pan']:.1f}°/s², {servo_state['acceleration_tilt']:.1f}°/s²", 
               (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Queue: {servo_state['queue_size']}", 
               (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Velocity limit indicators
    vel_limit_color = (0, 255, 0) if abs(servo_state['velocity_pan']) <= VELOCITY_LIMIT and abs(servo_state['velocity_tilt']) <= VELOCITY_LIMIT else (0, 0, 255)
    cv2.putText(frame, f"Vel Limit: {VELOCITY_LIMIT}°/s", 
               (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_limit_color, 2)
    
    # Enhanced GPS/MAVLink status
    if mavlink_handler:
        status = mavlink_handler.get_enhanced_status()
        
        # Connection status
        if status['connected']:
            connection_color = (0, 255, 0)
            conn_text = f"MAVLink: CONNECTED"
        else:
            connection_color = (0, 0, 255)
            conn_text = "MAVLink: DISCONNECTED"
        
        cv2.putText(frame, conn_text, (width - 300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, connection_color, 2)
        
        # GPS status
        if status['gps_fix'] >= 3:
            gps_color = (0, 255, 0)
            gps_text = f"GPS: {status['satellites']} sats"
        else:
            gps_color = (0, 0, 255)
            gps_text = f"GPS: {status['satellites']} sats (NO FIX)"
        
        cv2.putText(frame, gps_text, (width - 300, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 2)
    
    # Performance metrics
    fps_text = f"FPS: {user_data.get_fps():.1f}"
    cv2.putText(frame, fps_text, (width - 120, height - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

# ==============================================================================
# ENHANCED INITIALIZATION AND MAIN EXECUTION
# ==============================================================================

def initialize_ultra_smooth_system():
    """Initialize all ultra-smooth system components"""
    global mavlink_handler, data_logger, ultra_smooth_servo_controller, smooth_tracker, auto_calibration
    
    print("🚀 Initializing ULTRA-SMOOTH servo system...")
    
    # Initialize enhanced MAVLink connection
    mavlink_handler = None
    try:
        mavlink_handler = EnhancedMAVLinkHandler()
        if mavlink_handler.connected:
            print("✅ Enhanced MAVLink handler initialized successfully")
        else:
            print("⚠️ MAVLink connection failed, continuing without GPS features")
            mavlink_handler = None
    except Exception as e:
        print(f"⚠️ Enhanced MAVLink initialization failed: {e}")
        print("   Continuing without GPS waypoint functionality")
        mavlink_handler = None
    
    # Initialize enhanced data logger
    data_logger = EnhancedDataLogger(gps_handler=mavlink_handler)
    data_logger.log_event('system_init', 'Ultra-smooth servo tracking system starting')
    
    # Initialize ultra-smooth servo controller
    ultra_smooth_servo_controller = UltraSmoothServoController(data_logger)
    data_logger.log_event('servo_init', 'Ultra-smooth servo controller initialized')
    
    # Initialize robust auto-calibration
    auto_calibration = RobustAutoCalibration(data_logger)
    data_logger.log_event('calibration_init', 'Robust auto-calibration system initialized')
    
    # Initialize ultra-smooth tracker
    smooth_tracker = UltraSmoothTracker(ultra_smooth_servo_controller, data_logger, mavlink_handler)
    data_logger.log_event('tracker_init', 'Ultra-smooth tracker initialized')
    
    return True

def print_ultra_smooth_system_status():
    """Print comprehensive ultra-smooth system status"""
    print("\n" + "="*80)
    print("🎯 ULTRA-SMOOTH SERVO TRACKING SYSTEM STATUS")
    print("="*80)
    
    # MAVLink status
    if mavlink_handler:
        status = mavlink_handler.get_enhanced_status()
        print(f"\n🛰️ Enhanced MAVLink Status:")
        print(f"   Connection: {'✅ CONNECTED' if status['connected'] else '❌ DISCONNECTED'}")
        if status['connected']:
            print(f"   Uptime: {status['connection_uptime']:.1f}s")
            print(f"   Messages: {status['total_messages']}")
            print(f"   GPS Fix: {status['gps_fix']} ({'3D' if status['gps_fix'] >= 3 else '2D' if status['gps_fix'] >= 2 else 'NO FIX'})")
            print(f"   Satellites: {status['satellites']}")
            print(f"   Position: {status['latitude']:.6f}, {status['longitude']:.6f}")
            print(f"   Altitude: {status['altitude']:.1f}m")
    else:
        print(f"\n🛰️ MAVLink Status: ❌ NOT AVAILABLE")
    
    # Ultra-smooth servo status
    servo_state = ultra_smooth_servo_controller.get_enhanced_state()
    print(f"\n🔧 Ultra-Smooth Servo Status:")
    print(f"   Pan: {servo_state['current_pan']:.1f}° → {servo_state['target_pan']:.1f}° (S: {servo_state['smooth_pan']:.1f}°)")
    print(f"   Tilt: {servo_state['current_tilt']:.1f}° → {servo_state['target_tilt']:.1f}° (S: {servo_state['smooth_tilt']:.1f}°)")
    print(f"   Velocities: {servo_state['velocity_pan']:.1f}°/s, {servo_state['velocity_tilt']:.1f}°/s (Limit: {VELOCITY_LIMIT}°/s)")
    print(f"   Accelerations: {servo_state['acceleration_pan']:.1f}°/s², {servo_state['acceleration_tilt']:.1f}°/s² (Limit: {ACCELERATION_LIMIT}°/s²)")
    print(f"   Queue Size: {servo_state['queue_size']}")
    
    # Smoothing parameters
    print(f"\n📊 Ultra-Smooth Parameters:")
    print(f"   Smoothing Factor: {SMOOTHING_FACTOR}")
    print(f"   Dead Zone: {DEAD_ZONE} pixels")
    print(f"   Max Step Size: {MAX_STEP_SIZE}°")
    print(f"   Exponential Alpha: {EXPONENTIAL_SMOOTH_ALPHA}")
    print(f"   Motion Prediction Factor: {MOTION_PREDICTION_FACTOR}")
    print(f"   Distance Smooth Factor: {DISTANCE_SMOOTH_FACTOR}")
    
    # Calibration status
    if auto_calibration:
        cal_status = auto_calibration.get_calibration_status()
        print(f"\n📸 Auto-Calibration Status:")
        print(f"   Active: {'✅' if cal_status['active'] else '❌'}")
        print(f"   Complete: {'✅' if cal_status['complete'] else '❌'}")
        print(f"   Phase: {cal_status['phase']}")
        print(f"   Progress: {cal_status['progress']:.1f}% ({cal_status['samples_collected']}/{cal_status['required_samples']})")
    
    # Tracking status
    tracking_info = smooth_tracker.get_enhanced_tracking_info()
    print(f"\n🎯 Ultra-Smooth Tracking Status:")
    print(f"   Quality: {tracking_info['tracking_quality']}")
    print(f"   Lock: {'✅' if tracking_info['lock_on_target'] else '❌'}")
    print(f"   Frames Processed: {tracking_info['frames_processed']}")
    print(f"   Successful Tracks: {tracking_info['successful_tracks']}")
    print(f"   Average Confidence: {tracking_info['average_confidence']:.3f}")
    print(f"   Current Distance: {tracking_info['distance']:.2f}m")
    print(f"   Dynamic Dead Zone: {tracking_info['dynamic_dead_zone']:.1f}")
    print(f"   Dynamic Smoothing: {tracking_info['dynamic_smoothing']:.3f}")
    
    print("\n" + "="*80)

def ultra_smooth_shutdown_sequence():
    """Ultra-smooth shutdown with proper cleanup"""
    print("\n🛑 Ultra-smooth shutdown sequence initiated...")
    
    try:
        # Stop calibration
        if auto_calibration and auto_calibration.calibration_active:
            print("📸 Stopping auto-calibration...")
            auto_calibration.calibration_active = False
        
        # Finalize data logging
        print("📊 Finalizing ultra-smooth logs...")
        data_logger.finalize_enhanced_session()
        
        # Shutdown servo controller
        print("🔧 Shutting down ultra-smooth servo controller...")
        ultra_smooth_servo_controller.shutdown()
        
        # Shutdown MAVLink
        if mavlink_handler:
            print("🛰️ Closing enhanced MAVLink connection...")
            mavlink_handler.shutdown()
        
        print("✅ Ultra-smooth shutdown complete")
        
    except Exception as e:
        print(f"❌ Error during ultra-smooth shutdown: {e}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Set up environment
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    try:
        # Initialize ultra-smooth system
        if not initialize_ultra_smooth_system():
            print("❌ Ultra-smooth system initialization failed")
            exit(1)
        
        # Print system status
        print_ultra_smooth_system_status()
        
        # Initialize enhanced callback
        user_data = EnhancedSmoothAppCallback()
        app = GStreamerDetectionApp(enhanced_ultra_smooth_app_callback, user_data)
        
        print(f"\n🚀 Starting ULTRA-SMOOTH tracking system...")
        print(f"📊 Enhanced data output: {data_logger.log_dir}")
        print("\nPress Ctrl+C to stop")
        print("\nULTRA-SMOOTH FEATURES ACTIVE:")
        print("  ✅ Multi-stage filtering (Exponential + Kalman + Median)")
        print("  ✅ Velocity and acceleration limiting")
        print("  ✅ Distance-based adaptive smoothing")
        print("  ✅ Predictive tracking with smooth interpolation")
        print("  ✅ S-curve acceleration for natural movement")
        print("  ✅ Enhanced jitter reduction")
        print("  ✅ Stability-based detection filtering")
        print("  ✅ Center-biased target selection")
        print("  ✅ Ultra-responsive servo control")
        print("  ✅ Advanced smoothness metrics tracking")
        
        print(f"\n📈 SMOOTHING PARAMETERS:")
        print(f"  • Smoothing Factor: {SMOOTHING_FACTOR} (Higher = Smoother)")
        print(f"  • Dead Zone: {DEAD_ZONE} pixels (Reduces small movements)")
        print(f"  • Max Step Size: {MAX_STEP_SIZE}° (Limits large jumps)")
        print(f"  • Velocity Limit: {VELOCITY_LIMIT}°/s (Natural movement speed)")
        print(f"  • Acceleration Limit: {ACCELERATION_LIMIT}°/s² (Smooth acceleration)")
        print(f"  • Detection History: {DETECTION_HISTORY_SIZE} frames")
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 User requested shutdown...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        if data_logger:
            data_logger.log_event('error', f'Application error: {str(e)}')
    finally:
        ultra_smooth_shutdown_sequence()

# Global variables (initialized in main)
mavlink_handler = None
data_logger = None
ultra_smooth_servo_controller = None
smooth_tracker = None
auto_calibration = None
user_data = None
