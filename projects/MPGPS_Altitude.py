#!/usr/bin/env python3
"""
Enhanced Servo Tracking System with GPS Waypoint Generation and Temperature Monitoring
For Raspberry Pi 5 with CubeOrange and Hailo AI
- Uses CubeOrange altitude data
- Automatic camera calibration at startup
- Real-time RPi5 temperature monitoring with thermal warnings
- Robust error handling and validation
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
import psutil

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Servo tracking parameters
DEAD_ZONE = 15
SMOOTHING_FACTOR = 0.35
MAX_STEP_SIZE = 5
MIN_CONFIDENCE = 0.3
DETECTION_TIMEOUT = 2.0
PAN_SENSITIVITY = 45
TILT_SENSITIVITY = 35
FRAME_SKIP_COUNT = 1
DETECTION_HISTORY_SIZE = 3

# Camera parameters
CAMERA_FOV_HORIZONTAL = 79.9
CAMERA_FOV_VERTICAL = 64.3
AVERAGE_PERSON_HEIGHT = 1.7
AVERAGE_PERSON_WIDTH = 0.45

# Physical setup
SERVO_MOUNT_HEIGHT = 0.978
CAMERA_TILT_OFFSET = 5.0

# MAVLink configuration
MAVLINK_CONNECTION = '/dev/serial0'
MAVLINK_BAUD = 57600
MAVLINK_SYSTEM_ID = 255
MAVLINK_COMPONENT_ID = 190

# GPS waypoint parameters
GPS_UPDATE_INTERVAL = 1.0
MIN_DISTANCE_FOR_GPS = 3.0
MAX_GPS_POINTS = 100
WAYPOINT_ALTITUDE_OFFSET = 0.0
WAYPOINT_MODE = "ADD"
WAYPOINT_CLEAR_TIMEOUT = 300
MAX_WAYPOINTS = 15

# Calibration parameters
CALIBRATION_SAMPLES_REQUIRED = 30
CALIBRATION_MIN_DISTANCE = 1.5  # meters
CALIBRATION_MAX_DISTANCE = 5.0  # meters
CALIBRATION_TIMEOUT = 60  # seconds
CALIBRATION_MIN_CONFIDENCE = 0.5
CALIBRATION_MIN_BBOX_HEIGHT = 50  # pixels

# Temperature monitoring parameters
TEMP_UPDATE_INTERVAL = 2.0  # seconds
TEMP_WARNING_THRESHOLD = 70.0  # Celsius
TEMP_CRITICAL_THRESHOLD = 80.0  # Celsius
TEMP_THROTTLE_THRESHOLD = 75.0  # Celsius for performance throttling
TEMP_HISTORY_SIZE = 30  # Keep 1 minute of history at 2s intervals

# ==============================================================================
# RASPBERRY PI 5 TEMPERATURE MONITOR
# ==============================================================================

class RPi5TemperatureMonitor:
    def __init__(self, logger=None):
        self.logger = logger
        self.temp_sensor_path = '/sys/class/thermal/thermal_zone0/temp'
        self.current_temp = 0.0
        self.temp_history = deque(maxlen=TEMP_HISTORY_SIZE)
        self.last_update = 0
        self.last_warning_time = 0
        self.warning_cooldown = 30  # seconds between warnings
        self.running = True
        self.temp_thread = None
        self.max_temp = 0.0
        self.min_temp = 100.0
        self.temp_alerts = {
            'warning_triggered': False,
            'critical_triggered': False,
            'throttle_triggered': False,
            'last_alert_time': 0
        }
        
        # CPU frequency monitoring
        self.current_cpu_freq = 0.0
        self.cpu_usage = 0.0
        
        # Check if temperature sensor is available
        self.temp_available = self._check_temp_sensor()
        
        if self.temp_available:
            # Start monitoring thread
            self.temp_thread = threading.Thread(target=self._temperature_worker, daemon=True)
            self.temp_thread.start()
            
            if self.logger:
                self.logger.log_event('temp_monitor_init', 'RPi5 temperature monitoring initialized')
            print("🌡️  RPi5 temperature monitoring initialized")
        else:
            print("⚠️  Temperature sensor not available - monitoring disabled")
            if self.logger:
                self.logger.log_event('temp_monitor_init', 'Temperature sensor not available')
    
    def _check_temp_sensor(self):
        """Check if the temperature sensor is accessible"""
        try:
            with open(self.temp_sensor_path, 'r') as f:
                temp_raw = f.read().strip()
                temp = float(temp_raw) / 1000.0
                return 0 < temp < 200  # Reasonable temperature range
        except (FileNotFoundError, PermissionError, ValueError):
            return False
    
    def _temperature_worker(self):
        """Background thread to continuously monitor temperature"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - self.last_update >= TEMP_UPDATE_INTERVAL:
                    # Read temperature
                    temp = self._read_temperature()
                    
                    if temp is not None:
                        self.current_temp = temp
                        self.temp_history.append({
                            'timestamp': current_time,
                            'temperature': temp
                        })
                        
                        # Update min/max
                        self.max_temp = max(self.max_temp, temp)
                        if self.min_temp == 100.0:  # First reading
                            self.min_temp = temp
                        else:
                            self.min_temp = min(self.min_temp, temp)
                        
                        # Read CPU info
                        self._update_cpu_info()
                        
                        # Check for thermal alerts
                        self._check_thermal_alerts(temp, current_time)
                        
                        self.last_update = current_time
                        consecutive_errors = 0  # Reset error counter
                        
                    else:
                        consecutive_errors += 1
                        if consecutive_errors <= max_consecutive_errors:
                            print(f"Temperature read error ({consecutive_errors}/{max_consecutive_errors})")
                
                time.sleep(0.5)  # Check every 0.5 seconds, but only update every TEMP_UPDATE_INTERVAL
                
            except Exception as e:
                consecutive_errors += 1
                if self.running and consecutive_errors <= max_consecutive_errors:
                    print(f"Temperature monitoring error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("❌ Too many temperature monitoring errors. Disabling monitoring.")
                    self.temp_available = False
                    break
                
                time.sleep(1)
    
    def _read_temperature(self):
        """Read current temperature from sensor"""
        try:
            with open(self.temp_sensor_path, 'r') as f:
                temp_raw = f.read().strip()
                temp = float(temp_raw) / 1000.0
                return temp
        except (FileNotFoundError, PermissionError, ValueError) as e:
            return None
    
    def _update_cpu_info(self):
        """Update CPU frequency and usage information"""
        try:
            # Get CPU frequency
            self.current_cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # Get CPU usage (non-blocking)
            self.cpu_usage = psutil.cpu_percent(interval=None)
            
        except Exception:
            self.current_cpu_freq = 0.0
            self.cpu_usage = 0.0
    
    def _check_thermal_alerts(self, temp, current_time):
        """Check for thermal alerts and warnings"""
        alert_triggered = False
        
        # Critical temperature
        if temp >= TEMP_CRITICAL_THRESHOLD and not self.temp_alerts['critical_triggered']:
            self.temp_alerts['critical_triggered'] = True
            self.temp_alerts['last_alert_time'] = current_time
            alert_triggered = True
            
            warning_msg = f"🚨 CRITICAL: RPi5 temperature {temp:.1f}°C - System may throttle!"
            print(warning_msg)
            
            if self.logger:
                self.logger.log_event('temp_critical', warning_msg, {
                    'temperature': temp,
                    'threshold': TEMP_CRITICAL_THRESHOLD,
                    'cpu_freq': self.current_cpu_freq,
                    'cpu_usage': self.cpu_usage
                })
        
        # Warning temperature
        elif temp >= TEMP_WARNING_THRESHOLD and not self.temp_alerts['warning_triggered']:
            self.temp_alerts['warning_triggered'] = True
            self.temp_alerts['last_alert_time'] = current_time
            alert_triggered = True
            
            warning_msg = f"⚠️  WARNING: RPi5 temperature {temp:.1f}°C - Consider cooling"
            print(warning_msg)
            
            if self.logger:
                self.logger.log_event('temp_warning', warning_msg, {
                    'temperature': temp,
                    'threshold': TEMP_WARNING_THRESHOLD,
                    'cpu_freq': self.current_cpu_freq,
                    'cpu_usage': self.cpu_usage
                })
        
        # Throttle warning
        elif temp >= TEMP_THROTTLE_THRESHOLD and not self.temp_alerts['throttle_triggered']:
            self.temp_alerts['throttle_triggered'] = True
            alert_triggered = True
            
            warning_msg = f"🐌 THROTTLE WARNING: RPi5 temperature {temp:.1f}°C - Performance may be reduced"
            print(warning_msg)
            
            if self.logger:
                self.logger.log_event('temp_throttle', warning_msg, {
                    'temperature': temp,
                    'threshold': TEMP_THROTTLE_THRESHOLD,
                    'cpu_freq': self.current_cpu_freq
                })
        
        # Reset alerts when temperature drops
        if temp < TEMP_WARNING_THRESHOLD - 5:  # 5°C hysteresis
            if self.temp_alerts['warning_triggered'] or self.temp_alerts['critical_triggered']:
                self.temp_alerts['warning_triggered'] = False
                self.temp_alerts['critical_triggered'] = False
                self.temp_alerts['throttle_triggered'] = False
                
                if current_time - self.temp_alerts['last_alert_time'] > 60:  # Don't spam recovery messages
                    recovery_msg = f"✅ Temperature recovered: {temp:.1f}°C"
                    print(recovery_msg)
                    
                    if self.logger:
                        self.logger.log_event('temp_recovery', recovery_msg, {
                            'temperature': temp,
                            'cpu_freq': self.current_cpu_freq
                        })
    
    def get_temperature_info(self):
        """Get comprehensive temperature information"""
        if not self.temp_available:
            return {
                'available': False,
                'temperature': 0.0,
                'status': 'UNAVAILABLE'
            }
        
        # Determine status
        if self.current_temp >= TEMP_CRITICAL_THRESHOLD:
            status = 'CRITICAL'
            status_color = (0, 0, 255)  # Red
        elif self.current_temp >= TEMP_WARNING_THRESHOLD:
            status = 'WARNING'
            status_color = (0, 165, 255)  # Orange
        elif self.current_temp >= TEMP_THROTTLE_THRESHOLD:
            status = 'HIGH'
            status_color = (0, 255, 255)  # Yellow
        else:
            status = 'NORMAL'
            status_color = (0, 255, 0)  # Green
        
        # Calculate temperature trend
        trend = 'STABLE'
        if len(self.temp_history) >= 3:
            recent_temps = [t['temperature'] for t in list(self.temp_history)[-3:]]
            if recent_temps[-1] > recent_temps[0] + 2:
                trend = 'RISING'
            elif recent_temps[-1] < recent_temps[0] - 2:
                trend = 'FALLING'
        
        return {
            'available': True,
            'temperature': self.current_temp,
            'status': status,
            'status_color': status_color,
            'trend': trend,
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'cpu_freq': self.current_cpu_freq,
            'cpu_usage': self.cpu_usage,
            'history_size': len(self.temp_history),
            'alerts': self.temp_alerts.copy(),
            'thresholds': {
                'warning': TEMP_WARNING_THRESHOLD,
                'critical': TEMP_CRITICAL_THRESHOLD,
                'throttle': TEMP_THROTTLE_THRESHOLD
            }
        }
    
    def get_temperature_history(self):
        """Get temperature history for plotting/analysis"""
        if not self.temp_available:
            return []
        return list(self.temp_history)
    
    def should_throttle_performance(self):
        """Check if performance should be throttled due to temperature"""
        return self.current_temp >= TEMP_THROTTLE_THRESHOLD
    
    def shutdown(self):
        """Shutdown temperature monitoring gracefully"""
        print("🌡️  Shutting down temperature monitor...")
        self.running = False
        
        if self.temp_thread and self.temp_thread.is_alive():
            self.temp_thread.join(timeout=2.0)
        
        if self.logger:
            self.logger.log_event('temp_monitor_shutdown', 'Temperature monitoring shutdown', {
                'final_temp': self.current_temp,
                'max_temp': self.max_temp,
                'min_temp': self.min_temp,
                'total_readings': len(self.temp_history)
            })
        
        print("✅ Temperature monitor shutdown complete")

# ==============================================================================
# ENHANCED CALIBRATION SYSTEM
# ==============================================================================

class AutoCalibrationSystem:
    def __init__(self):
        self.is_calibrating = True
        self.calibration_complete = False
        self.measurements = []
        self.start_time = time.time()
        self.focal_length_x = None
        self.focal_length_y = None
        self.calibration_distances = [2.0, 3.0, 4.0]  # Known distances for calibration
        self.current_cal_distance_idx = 0
        self.distance_measurements = {dist: [] for dist in self.calibration_distances}
        self.validation_samples = []
        
    def add_measurement(self, bbox, frame_width, frame_height, confidence):
        """Add a calibration measurement with validation"""
        if not self.is_calibrating or self.calibration_complete:
            return False
            
        # Check if measurement is valid
        if confidence < CALIBRATION_MIN_CONFIDENCE:
            return False
            
        bbox_height_pixels = bbox.height() * frame_height
        bbox_width_pixels = bbox.width() * frame_width
        
        if bbox_height_pixels < CALIBRATION_MIN_BBOX_HEIGHT:
            return False
            
        # Calculate aspect ratio to filter out non-person detections
        aspect_ratio = bbox_width_pixels / bbox_height_pixels
        if aspect_ratio < 0.2 or aspect_ratio > 0.8:  # Person aspect ratio range
            return False
            
        measurement = {
            'timestamp': time.time(),
            'bbox_height': bbox_height_pixels,
            'bbox_width': bbox_width_pixels,
            'confidence': confidence,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'aspect_ratio': aspect_ratio
        }
        
        self.measurements.append(measurement)
        
        # Check if we have enough samples or timeout
        elapsed_time = time.time() - self.start_time
        
        if len(self.measurements) >= CALIBRATION_SAMPLES_REQUIRED or elapsed_time > CALIBRATION_TIMEOUT:
            return self._complete_calibration()
            
        return True
    
    def _complete_calibration(self):
        """Complete the calibration process and calculate focal lengths"""
        if len(self.measurements) < 10:
            print("❌ Insufficient calibration data. Using default values.")
            self._use_defaults()
            return False
            
        # Filter out outliers using statistical methods
        heights = [m['bbox_height'] for m in self.measurements]
        widths = [m['bbox_width'] for m in self.measurements]
        confidences = [m['confidence'] for m in self.measurements]
        
        # Remove outliers using IQR method
        heights_filtered = self._remove_outliers(heights)
        widths_filtered = self._remove_outliers(widths)
        
        if len(heights_filtered) < 5:
            print("❌ Too many outliers in calibration data. Using defaults.")
            self._use_defaults()
            return False
            
        # Calculate focal lengths using multiple reference distances
        avg_height = statistics.median(heights_filtered)
        avg_width = statistics.median(widths_filtered)
        avg_confidence = statistics.mean(confidences)
        
        # Use a reference distance (assume person is 3 meters away during calibration)
        reference_distance = 3.0
        
        # Calculate focal lengths
        self.focal_length_y = (avg_height * reference_distance) / AVERAGE_PERSON_HEIGHT
        self.focal_length_x = (avg_width * reference_distance) / AVERAGE_PERSON_WIDTH
        
        # Validate the calculated focal lengths
        frame_width = self.measurements[0]['frame_width']
        frame_height = self.measurements[0]['frame_height']
        
        # Cross-check with theoretical focal lengths
        theoretical_fx = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        theoretical_fy = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        
        # If calculated values are too far from theoretical, use a weighted average
        fx_diff = abs(self.focal_length_x - theoretical_fx) / theoretical_fx
        fy_diff = abs(self.focal_length_y - theoretical_fy) / theoretical_fy
        
        if fx_diff > 0.3:  # More than 30% difference
            self.focal_length_x = 0.7 * theoretical_fx + 0.3 * self.focal_length_x
            print(f"⚠️  Focal length X adjusted due to large deviation")
            
        if fy_diff > 0.3:
            self.focal_length_y = 0.7 * theoretical_fy + 0.3 * self.focal_length_y
            print(f"⚠️  Focal length Y adjusted due to large deviation")
        
        self.calibration_complete = True
        self.is_calibrating = False
        
        print(f"\n📸 CAMERA CALIBRATION COMPLETE!")
        print(f"   Samples collected: {len(self.measurements)}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Average person height: {avg_height:.1f} pixels")
        print(f"   Average person width: {avg_width:.1f} pixels")
        print(f"   Calculated focal length X: {self.focal_length_x:.1f} pixels")
        print(f"   Calculated focal length Y: {self.focal_length_y:.1f} pixels")
        print(f"   Theoretical focal length X: {theoretical_fx:.1f} pixels")
        print(f"   Theoretical focal length Y: {theoretical_fy:.1f} pixels")
        print(f"   Deviation X: {fx_diff*100:.1f}%, Y: {fy_diff*100:.1f}%")
        
        return True
    
    def _remove_outliers(self, data):
        """Remove outliers using the IQR method"""
        if len(data) < 4:
            return data
            
        q1 = statistics.quantiles(data, n=4)[0]
        q3 = statistics.quantiles(data, n=4)[2]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered = [x for x in data if lower_bound <= x <= upper_bound]
        return filtered if len(filtered) >= len(data) * 0.5 else data
    
    def _use_defaults(self):
        """Use default focal length values based on camera FOV"""
        # Use default frame size for calculation
        default_width, default_height = 640, 480
        
        self.focal_length_x = default_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = default_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        
        self.calibration_complete = True
        self.is_calibrating = False
        
        print(f"   Using default focal lengths: X={self.focal_length_x:.1f}, Y={self.focal_length_y:.1f}")
    
    def get_progress(self):
        """Get calibration progress percentage"""
        if self.calibration_complete:
            return 100
        
        sample_progress = (len(self.measurements) / CALIBRATION_SAMPLES_REQUIRED) * 100
        time_progress = ((time.time() - self.start_time) / CALIBRATION_TIMEOUT) * 100
        
        return min(100, max(sample_progress, time_progress))
    
    def get_status_text(self):
        """Get calibration status text for display"""
        if self.calibration_complete:
            return "CALIBRATION COMPLETE"
        
        progress = self.get_progress()
        samples = len(self.measurements)
        elapsed = int(time.time() - self.start_time)
        
        return f"CALIBRATING: {progress:.0f}% ({samples}/{CALIBRATION_SAMPLES_REQUIRED}) {elapsed}s"

# ==============================================================================
# ENHANCED MAVLINK GPS HANDLER
# ==============================================================================

class MAVLinkGPSHandler:
    def __init__(self, connection_string=MAVLINK_CONNECTION, baud=MAVLINK_BAUD):
        self.connection_string = connection_string
        self.baud = baud
        self.mavlink_connection = None
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0  # From GPS
        self.current_relative_alt = 0.0  # Relative to home
        self.current_terrain_alt = 0.0  # Above terrain
        self.barometric_alt = 0.0  # From barometer
        self.current_heading = 0.0
        self.gps_fix_type = 0
        self.satellites_visible = 0
        self.home_lat = None
        self.home_lon = None
        self.home_alt = None
        self.mission_count = 0
        self.current_wp_seq = 0
        self.active_waypoint = None
        self.waypoint_reached_threshold = 5.0
        self.last_waypoint_time = 0
        self.waypoint_mode = WAYPOINT_MODE
        self.running = True
        self.mavlink_thread = None
        self.last_gps_update = 0
        self.gps_points = deque(maxlen=MAX_GPS_POINTS)
        self.last_point_time = 0
        self.EARTH_RADIUS = 6371000
        self.altitude_source = "GPS"  # GPS, BAROMETRIC, RELATIVE, TERRAIN
        self.connection_healthy = False
        self.last_heartbeat = 0
        self.connect()
    
    def connect(self):
        try:
            print(f"🛰️ Connecting to CubeOrange at {self.connection_string}...")
            self.mavlink_connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
                source_system=MAVLINK_SYSTEM_ID,
                source_component=MAVLINK_COMPONENT_ID
            )
            print("⏳ Waiting for heartbeat...")
            self.mavlink_connection.wait_heartbeat(timeout=10)
            print("✅ MAVLink connection established!")
            self.connection_healthy = True
            self.last_heartbeat = time.time()
            self.request_data_streams()
            self.mavlink_thread = threading.Thread(target=self._mavlink_receiver, daemon=True)
            self.mavlink_thread.start()
            return True
        except Exception as e:
            print(f"❌ MAVLink connection failed: {e}")
            self.connection_healthy = False
            return False
    
    def request_data_streams(self):
        """Request all necessary data streams from the flight controller"""
        try:
            # Request multiple data streams
            stream_requests = [
                (mavutil.mavlink.MAV_DATA_STREAM_ALL, 4),
                (mavutil.mavlink.MAV_DATA_STREAM_GPS_RAW, 5),
                (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5),
                (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 4),
                (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 4),
            ]
            
            for stream_type, rate in stream_requests:
                self.mavlink_connection.mav.request_data_stream_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component,
                    stream_type,
                    rate, 1
                )
                time.sleep(0.1)  # Small delay between requests
                
        except Exception as e:
            print(f"Error requesting data streams: {e}")
    
    def _mavlink_receiver(self):
        """Enhanced MAVLink message receiver with better altitude handling"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.5)
                
                if msg is None:
                    continue
                
                consecutive_errors = 0  # Reset error counter on successful receive
                msg_type = msg.get_type()
                
                # Update heartbeat tracking
                if msg_type == 'HEARTBEAT':
                    self.last_heartbeat = time.time()
                    self.connection_healthy = True
                
                # GPS data - primary altitude source
                elif msg_type == 'GPS_RAW_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0  # GPS altitude (MSL)
                    self.gps_fix_type = msg.fix_type
                    self.satellites_visible = msg.satellites_visible
                    self.last_gps_update = time.time()
                
                # Global position with multiple altitude references
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0  # GPS altitude (MSL)
                    self.current_relative_alt = msg.relative_alt / 1000.0  # Relative to home
                    self.current_heading = msg.hdg / 100.0
                
                # Barometric altitude
                elif msg_type == 'VFR_HUD':
                    self.barometric_alt = msg.alt
                
                # Terrain altitude
                elif msg_type == 'TERRAIN_REPORT':
                    self.current_terrain_alt = msg.current_height
                
                # Attitude for heading backup
                elif msg_type == 'ATTITUDE':
                    yaw_rad = msg.yaw
                    if self.current_heading == 0:  # Use as backup if GPS heading unavailable
                        self.current_heading = math.degrees(yaw_rad) % 360
                
                # Home position
                elif msg_type == 'HOME_POSITION':
                    self.home_lat = msg.latitude / 1e7
                    self.home_lon = msg.longitude / 1e7
                    self.home_alt = msg.altitude / 1000.0
                    print(f"🏠 Home position set: {self.home_lat:.6f}, {self.home_lon:.6f}, {self.home_alt:.1f}m")
                
                # Mission status
                elif msg_type == 'MISSION_CURRENT':
                    self.current_wp_seq = msg.seq
                    
                elif msg_type == 'MISSION_ITEM_REACHED':
                    print(f"✅ Reached waypoint {msg.seq}")
                    
            except Exception as e:
                consecutive_errors += 1
                if self.running and consecutive_errors <= max_consecutive_errors:
                    print(f"MAVLink receive error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                if consecutive_errors >= max_consecutive_errors:
                    print("❌ Too many consecutive MAVLink errors. Connection may be lost.")
                    self.connection_healthy = False
                    time.sleep(1)  # Wait before retrying
                    consecutive_errors = 0
    
    def get_altitude_for_waypoint(self):
        """Get the best altitude for waypoint creation based on available data"""
        current_time = time.time()
        
        # Check connection health
        if not self.connection_healthy or (current_time - self.last_heartbeat) > 5:
            print("⚠️  MAVLink connection unhealthy for waypoint altitude")
            return None
        
        # Prefer GPS altitude if we have a good fix
        if self.gps_fix_type >= 3 and self.current_alt > 0:
            altitude = self.current_alt + WAYPOINT_ALTITUDE_OFFSET
            print(f"📍 Using GPS altitude: {altitude:.1f}m MSL")
            return altitude
            
        # Fall back to relative altitude if available
        elif self.current_relative_alt > 0:
            altitude = self.current_relative_alt + WAYPOINT_ALTITUDE_OFFSET
            if self.home_alt:
                altitude += self.home_alt
            print(f"📍 Using relative altitude: {altitude:.1f}m")
            return altitude
            
        # Use barometric altitude as last resort
        elif self.barometric_alt > 0:
            altitude = self.barometric_alt + WAYPOINT_ALTITUDE_OFFSET
            print(f"📍 Using barometric altitude: {altitude:.1f}m")
            return altitude
            
        else:
            print("❌ No valid altitude data available for waypoint")
            return None
    
    def calculate_gps_position(self, x_meters, y_meters):
        """Calculate GPS position from relative coordinates"""
        if self.gps_fix_type < 3:
            return None, None
        
        bearing_to_target = math.degrees(math.atan2(x_meters, y_meters))
        absolute_bearing = (self.current_heading + bearing_to_target) % 360
        distance = math.sqrt(x_meters**2 + y_meters**2)
        
        lat_rad = math.radians(self.current_lat)
        lon_rad = math.radians(self.current_lon)
        bearing_rad = math.radians(absolute_bearing)
        
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance / self.EARTH_RADIUS) +
            math.cos(lat_rad) * math.sin(distance / self.EARTH_RADIUS) * math.cos(bearing_rad)
        )
        
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance / self.EARTH_RADIUS) * math.cos(lat_rad),
            math.cos(distance / self.EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad)
    
    def add_detection_point(self, x_meters, y_meters, z_meters, confidence):
        """Add a detection point and create waypoint with enhanced altitude handling"""
        current_time = time.time()
        if current_time - self.last_point_time < GPS_UPDATE_INTERVAL:
            return None
        
        distance = math.sqrt(x_meters**2 + y_meters**2)
        if distance < MIN_DISTANCE_FOR_GPS:
            return None
        
        lat, lon = self.calculate_gps_position(x_meters, y_meters)
        if lat is None or lon is None:
            return None
        
        # Use enhanced altitude calculation
        waypoint_alt = self.get_altitude_for_waypoint()
        if waypoint_alt is None:
            print("❌ Cannot create waypoint: no valid altitude data")
            return None
        
        gps_point = {
            'timestamp': current_time,
            'latitude': lat,
            'longitude': lon,
            'altitude': waypoint_alt,
            'altitude_source': self.altitude_source,
            'relative_x': x_meters,
            'relative_y': y_meters,
            'relative_z': z_meters,
            'distance': distance,
            'confidence': confidence,
            'vehicle_lat': self.current_lat,
            'vehicle_lon': self.current_lon,
            'vehicle_alt_gps': self.current_alt,
            'vehicle_alt_relative': self.current_relative_alt,
            'vehicle_alt_barometric': self.barometric_alt,
            'vehicle_heading': self.current_heading,
            'gps_fix_type': self.gps_fix_type,
            'satellites': self.satellites_visible
        }
        
        self.gps_points.append(gps_point)
        self.last_point_time = current_time
        
        success = False
        if self.waypoint_mode == "REPLACE":
            self.clear_mission()
            success = self.upload_waypoint(lat, lon, waypoint_alt, 1)
        elif self.waypoint_mode == "ADD":
            if len(self.gps_points) <= MAX_WAYPOINTS:
                wp_seq = self.get_current_mission_count()
                success = self.upload_waypoint(lat, lon, waypoint_alt, wp_seq)
        elif self.waypoint_mode == "CLEAR_OLD":
            if current_time - self.last_waypoint_time > WAYPOINT_CLEAR_TIMEOUT:
                self.clear_mission()
            wp_seq = self.get_current_mission_count()
            success = self.upload_waypoint(lat, lon, waypoint_alt, wp_seq)
        
        if success:
            self.last_waypoint_time = current_time
            bearing = math.degrees(math.atan2(x_meters, y_meters))
            bearing = (self.current_heading + bearing) % 360
            print(f"🎯 New waypoint: {distance:.1f}m @ {bearing:.0f}° Alt: {waypoint_alt:.1f}m")
            self.notify_mission_changed()
        
        return gps_point
    
    def get_current_mission_count(self):
        """Get current mission count with timeout and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Clear any pending messages
                while self.mavlink_connection.recv_match(blocking=False):
                    pass
                
                self.mavlink_connection.mav.mission_request_list_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component
                )
                
                msg = self.mavlink_connection.recv_match(type='MISSION_COUNT', blocking=True, timeout=3)
                if msg:
                    self.mission_count = msg.count
                    return msg.count
                    
            except Exception as e:
                print(f"Error getting mission count (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    
        print("Using default mission count: 1")
        return 1
    
    def upload_waypoint(self, lat, lon, alt, seq):
        """Upload waypoint with enhanced error handling and validation"""
        try:
            print(f"📤 Uploading waypoint {seq}: {lat:.6f}, {lon:.6f}, {alt:.1f}m")
            
            # Validate inputs
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180) or alt < -1000 or alt > 50000:
                print(f"❌ Invalid waypoint coordinates: {lat}, {lon}, {alt}")
                return False
            
            self.mavlink_connection.mav.mission_count_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                seq + 1, 0
            )
            
            start_time = time.time()
            items_sent = set()
            timeout = 8  # Increased timeout
            
            while time.time() - start_time < timeout:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=1.0)
                
                if msg:
                    msg_type = msg.get_type()
                    
                    if msg_type in ['MISSION_REQUEST', 'MISSION_REQUEST_INT']:
                        requested_seq = msg.seq
                        
                        if requested_seq not in items_sent:
                            if requested_seq == 0:
                                self.send_home_position()
                            elif requested_seq == seq:
                                self.mavlink_connection.mav.mission_item_int_send(
                                    self.mavlink_connection.target_system,
                                    self.mavlink_connection.target_component,
                                    seq,
                                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                    0, 1, 0, 5, 0, float('nan'),
                                    int(lat * 1e7),
                                    int(lon * 1e7),
                                    alt, 0
                                )
                            items_sent.add(requested_seq)
                            
                    elif msg_type == 'MISSION_ACK':
                        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                            print("✅ Waypoint uploaded successfully!")
                            return True
                        else:
                            print(f"❌ Mission rejected: {msg.type}")
                            return False
            
            print("❌ Timeout waiting for mission protocol")
            return False
            
        except Exception as e:
            print(f"Error uploading waypoint: {e}")
            return False
    
    def send_home_position(self):
        """Send home position with validation"""
        try:
            if self.home_lat and self.home_lon and self.home_alt:
                lat, lon, alt = self.home_lat, self.home_lon, self.home_alt
            elif self.current_lat != 0 and self.current_lon != 0:
                lat, lon, alt = self.current_lat, self.current_lon, self.current_alt
            else:
                print("❌ No valid home position available")
                return False
            
            self.mavlink_connection.mav.mission_item_int_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                1, 1, 0, 0, 0, 0,
                int(lat * 1e7),
                int(lon * 1e7),
                0, 0
            )
            return True
            
        except Exception as e:
            print(f"Error sending home position: {e}")
            return False
    
    def notify_mission_changed(self):
        """Notify flight controller of mission changes"""
        try:
            self.mavlink_connection.mav.mission_current_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                self.current_wp_seq
            )
        except Exception as e:
            print(f"Error notifying mission change: {e}")
    
    def clear_mission(self):
        """Clear mission with proper acknowledgment"""
        try:
            self.mavlink_connection.mav.mission_clear_all_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component
            )
            msg = self.mavlink_connection.recv_match(type='MISSION_ACK', blocking=True, timeout=3)
            if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("🗑️  Mission cleared")
                self.mission_count = 0
                return True
            else:
                print("❌ Mission clear failed or timed out")
        except Exception as e:
            print(f"Error clearing mission: {e}")
        return False
    
    def get_status(self):
        """Get comprehensive status information"""
        current_time = time.time()
        return {
            'connected': self.connection_healthy,
            'gps_fix': self.gps_fix_type,
            'satellites': self.satellites_visible,
            'latitude': self.current_lat,
            'longitude': self.current_lon,
            'altitude_gps': self.current_alt,
            'altitude_relative': self.current_relative_alt,
            'altitude_barometric': self.barometric_alt,
            'altitude_terrain': self.current_terrain_alt,
            'heading': self.current_heading,
            'last_update': current_time - self.last_gps_update,
            'last_heartbeat': current_time - self.last_heartbeat,
            'points_logged': len(self.gps_points),
            'mission_count': self.mission_count,
            'home_position': {
                'lat': self.home_lat,
                'lon': self.home_lon,
                'alt': self.home_alt
            }
        }
    
    def shutdown(self):
        """Shutdown MAVLink connection gracefully"""
        print("🛰️ Shutting down MAVLink connection...")
        self.running = False
        
        if self.mavlink_thread and self.mavlink_thread.is_alive():
            self.mavlink_thread.join(timeout=2.0)
        
        if self.mavlink_connection:
            try:
                self.mavlink_connection.close()
            except:
                pass
        
        self.connection_healthy = False
        print("✅ MAVLink connection closed")

# ==============================================================================
# ENHANCED DATA LOGGER
# ==============================================================================

class ServoDataLogger:
    def __init__(self, log_dir="servo_logs", gps_handler=None, temp_monitor=None):
        self.gps_handler = gps_handler
        self.temp_monitor = temp_monitor
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"session_{timestamp}.json"
        self.gps_csv_file = self.log_dir / f"gps_points_{timestamp}.csv"
        self.calibration_file = self.log_dir / f"calibration_{timestamp}.json"
        self.temp_csv_file = self.log_dir / f"temperature_{timestamp}.csv"
        
        # Enhanced CSV headers with temperature information
        self.csv_headers = [
            'timestamp', 'frame_count', 'pan_angle', 'tilt_angle',
            'pan_velocity', 'tilt_velocity', 'detection_confidence',
            'person_detected', 'tracking_active', 'target_lost_frames',
            'distance_meters', 'x_position', 'y_position', 'z_position',
            'angular_width', 'angular_height', 'bbox_width', 'bbox_height',
            'gps_latitude', 'gps_longitude', 'gps_altitude', 'altitude_source',
            'calibration_status', 'focal_length_x', 'focal_length_y',
            'cpu_temp', 'cpu_freq', 'cpu_usage', 'temp_status'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_headers)
        
        # Enhanced GPS headers with multiple altitude sources
        self.gps_headers = [
            'timestamp', 'detection_lat', 'detection_lon', 'detection_alt', 'altitude_source',
            'vehicle_lat', 'vehicle_lon', 'vehicle_heading',
            'vehicle_alt_gps', 'vehicle_alt_relative', 'vehicle_alt_barometric',
            'relative_x', 'relative_y', 'relative_z', 'confidence',
            'gps_fix_type', 'satellites', 'distance'
        ]
        
        with open(self.gps_csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.gps_headers)
        
        # Temperature CSV headers
        self.temp_headers = [
            'timestamp', 'temperature', 'cpu_freq', 'cpu_usage',
            'temp_status', 'thermal_alerts', 'min_temp', 'max_temp'
        ]
        
        with open(self.temp_csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.temp_headers)
        
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'log_files': {
                'csv': str(self.csv_file),
                'json': str(self.json_file),
                'gps_csv': str(self.gps_csv_file),
                'calibration': str(self.calibration_file),
                'temperature': str(self.temp_csv_file)
            },
            'statistics': {
                'total_detections': 0,
                'total_movements': 0,
                'min_distance': float('inf'),
                'max_distance': 0.0,
                'avg_distance': 0.0,
                'distance_samples': 0,
                'gps_points_created': 0,
                'calibration_samples': 0,
                'temp_readings': 0,
                'max_temp': 0.0,
                'min_temp': 100.0,
                'temp_alerts': 0
            },
            'events': [],
            'calibration_data': None,
            'temperature_data': {
                'monitoring_enabled': temp_monitor is not None and temp_monitor.temp_available,
                'thresholds': {
                    'warning': TEMP_WARNING_THRESHOLD,
                    'critical': TEMP_CRITICAL_THRESHOLD,
                    'throttle': TEMP_THROTTLE_THRESHOLD
                }
            },
            'system_info': {
                'camera_fov_h': CAMERA_FOV_HORIZONTAL,
                'camera_fov_v': CAMERA_FOV_VERTICAL,
                'servo_mount_height': SERVO_MOUNT_HEIGHT,
                'camera_tilt_offset': CAMERA_TILT_OFFSET,
                'temp_monitoring': temp_monitor is not None
            }
        }
        
        print(f"📊 Enhanced data logging with temperature monitoring to: {self.log_dir}")
    
    def log_temperature_data(self):
        """Log current temperature data to CSV"""
        if not self.temp_monitor or not self.temp_monitor.temp_available:
            return
        
        try:
            temp_info = self.temp_monitor.get_temperature_info()
            
            with open(self.temp_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(),
                    temp_info['temperature'],
                    temp_info['cpu_freq'],
                    temp_info['cpu_usage'],
                    temp_info['status'],
                    json.dumps(temp_info['alerts']),
                    temp_info['min_temp'],
                    temp_info['max_temp']
                ])
            
            # Update statistics
            stats = self.session_data['statistics']
            stats['temp_readings'] += 1
            stats['max_temp'] = max(stats['max_temp'], temp_info['temperature'])
            if stats['min_temp'] == 100.0:  # First reading
                stats['min_temp'] = temp_info['temperature']
            else:
                stats['min_temp'] = min(stats['min_temp'], temp_info['temperature'])
            
            # Count alerts
            if any(temp_info['alerts'].values()):
                stats['temp_alerts'] += 1
                
        except Exception as e:
            print(f"Error logging temperature data: {e}")
    
    def log_calibration_data(self, calibration_system):
        """Log calibration data to file"""
        try:
            calibration_data = {
                'timestamp': datetime.now().isoformat(),
                'calibration_complete': calibration_system.calibration_complete,
                'samples_collected': len(calibration_system.measurements),
                'focal_length_x': calibration_system.focal_length_x,
                'focal_length_y': calibration_system.focal_length_y,
                'measurements': calibration_system.measurements,
                'validation_info': {
                    'theoretical_fx': None,
                    'theoretical_fy': None,
                    'deviation_x': None,
                    'deviation_y': None
                }
            }
            
            # Calculate theoretical values for comparison
            if calibration_system.measurements:
                frame_width = calibration_system.measurements[0]['frame_width']
                frame_height = calibration_system.measurements[0]['frame_height']
                
                theoretical_fx = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
                theoretical_fy = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
                
                calibration_data['validation_info'].update({
                    'theoretical_fx': theoretical_fx,
                    'theoretical_fy': theoretical_fy,
                    'frame_width': frame_width,
                    'frame_height': frame_height
                })
                
                if calibration_system.focal_length_x and calibration_system.focal_length_y:
                    calibration_data['validation_info'].update({
                        'deviation_x': abs(calibration_system.focal_length_x - theoretical_fx) / theoretical_fx,
                        'deviation_y': abs(calibration_system.focal_length_y - theoretical_fy) / theoretical_fy
                    })
            
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2, default=str)
            
            self.session_data['calibration_data'] = calibration_data
            self.session_data['statistics']['calibration_samples'] = len(calibration_system.measurements)
            
            print(f"📸 Calibration data saved to: {self.calibration_file}")
            
        except Exception as e:
            print(f"Error logging calibration data: {e}")
    
    def log_frame_data(self, frame_count, pan_angle, tilt_angle, pan_velocity,
                      tilt_velocity, detection_confidence, person_detected,
                      tracking_active, target_lost_frames, distance_data=None,
                      calibration_system=None):
        try:
            distance = x_pos = y_pos = z_pos = 0.0
            angular_width = angular_height = bbox_width = bbox_height = 0.0
            gps_lat = gps_lon = gps_alt = 0.0
            altitude_source = "N/A"
            calibration_status = "COMPLETE" if calibration_system and calibration_system.calibration_complete else "CALIBRATING"
            focal_length_x = calibration_system.focal_length_x if calibration_system else 0.0
            focal_length_y = calibration_system.focal_length_y if calibration_system else 0.0
            
            # Get temperature data
            cpu_temp = cpu_freq = cpu_usage = 0.0
            temp_status = "N/A"
            
            if self.temp_monitor and self.temp_monitor.temp_available:
                temp_info = self.temp_monitor.get_temperature_info()
                cpu_temp = temp_info['temperature']
                cpu_freq = temp_info['cpu_freq']
                cpu_usage = temp_info['cpu_usage']
                temp_status = temp_info['status']
            
            if distance_data:
                distance = distance_data.get('distance', 0.0)
                x_pos = distance_data.get('x_position', 0.0)
                y_pos = distance_data.get('y_position', 0.0)
                z_pos = distance_data.get('z_position', 0.0)
                angular_width = distance_data.get('angular_width', 0.0)
                angular_height = distance_data.get('angular_height', 0.0)
                bbox_width = distance_data.get('bbox_width', 0.0)
                bbox_height = distance_data.get('bbox_height', 0.0)
                
                if self.gps_handler and distance >= MIN_DISTANCE_FOR_GPS:
                    gps_point = self.gps_handler.add_detection_point(
                        x_pos, y_pos, z_pos, detection_confidence
                    )
                    
                    if gps_point:
                        gps_lat = gps_point['latitude']
                        gps_lon = gps_point['longitude']
                        gps_alt = gps_point['altitude']
                        altitude_source = gps_point['altitude_source']
                        self.session_data['statistics']['gps_points_created'] += 1
                        
                        # Enhanced GPS logging with all altitude sources
                        with open(self.gps_csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                gps_point['timestamp'],
                                gps_point['latitude'],
                                gps_point['longitude'],
                                gps_point['altitude'],
                                gps_point['altitude_source'],
                                gps_point['vehicle_lat'],
                                gps_point['vehicle_lon'],
                                gps_point['vehicle_heading'],
                                gps_point['vehicle_alt_gps'],
                                gps_point['vehicle_alt_relative'],
                                gps_point['vehicle_alt_barometric'],
                                gps_point['relative_x'],
                                gps_point['relative_y'],
                                gps_point['relative_z'],
                                gps_point['confidence'],
                                gps_point['gps_fix_type'],
                                gps_point['satellites'],
                                gps_point['distance']
                            ])
                
                if distance > 0:
                    stats = self.session_data['statistics']
                    stats['min_distance'] = min(stats['min_distance'], distance)
                    stats['max_distance'] = max(stats['max_distance'], distance)
                    stats['distance_samples'] += 1
                    stats['avg_distance'] = (
                        (stats['avg_distance'] * (stats['distance_samples'] - 1) + distance) /
                        stats['distance_samples']
                    )
            
            # Enhanced CSV logging with calibration, altitude, and temperature data
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), frame_count, pan_angle, tilt_angle,
                    pan_velocity, tilt_velocity, detection_confidence,
                    person_detected, tracking_active, target_lost_frames,
                    distance, x_pos, y_pos, z_pos,
                    angular_width, angular_height, bbox_width, bbox_height,
                    gps_lat, gps_lon, gps_alt, altitude_source,
                    calibration_status, focal_length_x, focal_length_y,
                    cpu_temp, cpu_freq, cpu_usage, temp_status
                ])
            
            # Log temperature data periodically
            if frame_count % 60 == 0:  # Every 2 seconds at 30fps
                self.log_temperature_data()
            
            if person_detected:
                self.session_data['statistics']['total_detections'] += 1
            if abs(pan_velocity) > 1 or abs(tilt_velocity) > 1:
                self.session_data['statistics']['total_movements'] += 1
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_event(self, event_type, description, data=None):
        """Enhanced event logging with optional data"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description,
            'data': data
        }
        self.session_data['events'].append(event)
        print(f"📝 {event_type}: {description}")
    
    def finalize_session(self):
        """Finalize session with comprehensive data including temperature"""
        self.session_data['end_time'] = datetime.now().isoformat()
        
        if self.gps_handler:
            self.session_data['gps_status'] = self.gps_handler.get_status()
        
        if self.temp_monitor:
            self.session_data['final_temperature'] = self.temp_monitor.get_temperature_info()
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
            
            stats = self.session_data['statistics']
            print(f"\n📊 Session Complete:")
            print(f"   Total Detections: {stats['total_detections']}")
            print(f"   GPS Waypoints: {stats['gps_points_created']}")
            print(f"   Calibration Samples: {stats['calibration_samples']}")
            if stats['distance_samples'] > 0:
                print(f"   Distance Range: {stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m")
                print(f"   Average Distance: {stats['avg_distance']:.2f}m")
            if stats['temp_readings'] > 0:
                print(f"   Temperature Range: {stats['min_temp']:.1f}°C - {stats['max_temp']:.1f}°C")
                print(f"   Temperature Alerts: {stats['temp_alerts']}")
            print(f"   Log Files: {len(self.session_data['log_files'])} created")
            
        except Exception as e:
            print(f"Session save error: {e}")

# ==============================================================================
# ENHANCED DISTANCE CALCULATOR
# ==============================================================================

class EnhancedDistanceCalculator:
    def __init__(self, frame_width=640, frame_height=480, calibration_system=None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.calibration_system = calibration_system
        self.distance_history = deque(maxlen=7)  # Increased history for better smoothing
        self.confidence_weights = deque(maxlen=7)
        
        # Initialize with default focal lengths
        self.focal_length_x = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        
    def update_frame_size(self, width, height):
        """Update frame size and recalculate focal lengths if needed"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            
            # Update focal lengths if calibration is complete
            if self.calibration_system and self.calibration_system.calibration_complete:
                self.focal_length_x = self.calibration_system.focal_length_x
                self.focal_length_y = self.calibration_system.focal_length_y
            else:
                # Use theoretical values based on FOV
                self.focal_length_x = width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
                self.focal_length_y = height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
    
    def update_calibration(self):
        """Update focal lengths from calibration system"""
        if (self.calibration_system and 
            self.calibration_system.calibration_complete and
            self.calibration_system.focal_length_x and 
            self.calibration_system.focal_length_y):
            
            self.focal_length_x = self.calibration_system.focal_length_x
            self.focal_length_y = self.calibration_system.focal_length_y
            return True
        return False
    
    def calculate_distance_from_bbox(self, bbox, confidence=1.0):
        """Enhanced distance calculation with confidence weighting"""
        bbox_width_pixels = bbox.width() * self.frame_width
        bbox_height_pixels = bbox.height() * self.frame_height
        
        # Calculate distances from both dimensions
        distance_from_height = (AVERAGE_PERSON_HEIGHT * self.focal_length_y) / bbox_height_pixels
        distance_from_width = (AVERAGE_PERSON_WIDTH * self.focal_length_x) / bbox_width_pixels
        
        # Weight the measurements - height is generally more reliable
        height_weight = 0.75
        width_weight = 0.25
        
        # Adjust weights based on aspect ratio quality
        aspect_ratio = bbox_width_pixels / bbox_height_pixels
        ideal_aspect_ratio = AVERAGE_PERSON_WIDTH / AVERAGE_PERSON_HEIGHT
        aspect_error = abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio
        
        if aspect_error > 0.5:  # Poor aspect ratio, rely more on height
            height_weight = 0.9
            width_weight = 0.1
        
        distance = distance_from_height * height_weight + distance_from_width * width_weight
        
        # Apply confidence-based filtering
        if confidence > 0.7:
            weight = 1.0
        elif confidence > 0.5:
            weight = 0.8
        else:
            weight = 0.6
            
        self.distance_history.append(distance)
        self.confidence_weights.append(weight * confidence)
        
        return self._get_smoothed_distance()
    
    def calculate_3d_position(self, bbox, pan_angle, tilt_angle, distance):
        """Calculate 3D position with enhanced accuracy"""
        # Convert angles to radians
        pan_rad = math.radians(pan_angle - 90)  # Adjust for servo mounting
        actual_tilt_angle = tilt_angle + CAMERA_TILT_OFFSET
        tilt_rad = math.radians(90 - actual_tilt_angle)
        
        # Calculate horizontal distance (ground plane projection)
        horizontal_distance = distance * math.cos(tilt_rad)
        
        # Calculate 3D coordinates
        x = horizontal_distance * math.sin(pan_rad)  # East-West
        y = horizontal_distance * math.cos(pan_rad)  # North-South
        z = distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT  # Height above ground
        
        return x, y, z
    
    def calculate_angular_size(self, bbox):
        """Calculate angular size of bounding box"""
        angular_width = bbox.width() * CAMERA_FOV_HORIZONTAL
        angular_height = bbox.height() * CAMERA_FOV_VERTICAL
        return angular_width, angular_height
    
    def _get_smoothed_distance(self):
        """Enhanced distance smoothing with confidence weighting"""
        if not self.distance_history:
            return 0.0
        
        if len(self.distance_history) == 1:
            return self.distance_history[0]
        
        # Use weighted average based on confidence
        distances = list(self.distance_history)
        weights = list(self.confidence_weights)
        
        if len(weights) != len(distances):
            weights = [1.0] * len(distances)
        
        # Remove extreme outliers (beyond 2 standard deviations)
        if len(distances) >= 3:
            mean_dist = sum(distances) / len(distances)
            std_dist = statistics.stdev(distances)
            
            filtered_pairs = []
            for d, w in zip(distances, weights):
                if abs(d - mean_dist) <= 2 * std_dist:
                    filtered_pairs.append((d, w))
            
            if len(filtered_pairs) >= len(distances) * 0.5:  # Keep at least half
                distances, weights = zip(*filtered_pairs)
                distances, weights = list(distances), list(weights)
        
        # Calculate weighted average
        weighted_sum = sum(d * w for d, w in zip(distances, weights))
        weight_sum = sum(weights)
        
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return sum(distances) / len(distances)
    
    def get_calibration_status(self):
        """Get current calibration status"""
        if self.calibration_system:
            return {
                'calibrated': self.calibration_system.calibration_complete,
                'focal_length_x': self.focal_length_x,
                'focal_length_y': self.focal_length_y,
                'samples': len(self.calibration_system.measurements) if hasattr(self.calibration_system, 'measurements') else 0
            }
        return {
            'calibrated': False,
            'focal_length_x': self.focal_length_x,
            'focal_length_y': self.focal_length_y,
            'samples': 0
        }

# ==============================================================================
# ENHANCED SERVO CONTROLLER
# ==============================================================================

class FastServoController:
    def __init__(self, logger=None, temp_monitor=None):
        self.logger = logger
        self.temp_monitor = temp_monitor
        
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = adafruit_pca9685.PCA9685(self.i2c)
            self.pca.frequency = 50
            
            self.pan_servo = servo.Servo(self.pca.channels[0])
            self.tilt_servo = servo.Servo(self.pca.channels[1])
            
            self.servo_initialized = True
            
        except Exception as e:
            print(f"❌ Servo initialization failed: {e}")
            self.servo_initialized = False
            self.pan_servo = None
            self.tilt_servo = None
        
        # Initialize positions
        self.current_pan = 90.0
        self.current_tilt = 90.0 - CAMERA_TILT_OFFSET
        self.velocity_pan = 0.0
        self.velocity_tilt = 0.0
        self.last_update_time = time.time()
        
        # Thermal management
        self.thermal_throttle_active = False
        self.max_movement_speed = MAX_STEP_SIZE
        
        # Set initial positions if servos are available
        if self.servo_initialized:
            try:
                self.pan_servo.angle = self.current_pan
                self.tilt_servo.angle = self.current_tilt
                time.sleep(0.5)  # Allow servos to reach position
            except Exception as e:
                print(f"⚠️  Initial servo positioning failed: {e}")
        
        # Command queue and worker thread
        self.command_queue = Queue(maxsize=5)
        self.running = True
        self.servo_thread = threading.Thread(target=self._servo_worker, daemon=True)
        self.servo_thread.start()
        
        if self.logger:
            self.logger.log_event('servo_init', 
                                'Servo controller initialized', 
                                {'servo_available': self.servo_initialized})
    
    def _check_thermal_throttle(self):
        """Check if thermal throttling should be applied"""
        if not self.temp_monitor or not self.temp_monitor.temp_available:
            return False
        
        should_throttle = self.temp_monitor.should_throttle_performance()
        
        if should_throttle and not self.thermal_throttle_active:
            self.thermal_throttle_active = True
            self.max_movement_speed = MAX_STEP_SIZE * 0.5  # Reduce movement speed
            print("🌡️  Thermal throttling activated - reducing servo speed")
            
            if self.logger:
                self.logger.log_event('thermal_throttle_start', 
                                    'Servo thermal throttling activated',
                                    {'temperature': self.temp_monitor.current_temp})
        
        elif not should_throttle and self.thermal_throttle_active:
            self.thermal_throttle_active = False
            self.max_movement_speed = MAX_STEP_SIZE  # Restore normal speed
            print("🌡️  Thermal throttling deactivated - normal servo speed")
            
            if self.logger:
                self.logger.log_event('thermal_throttle_end', 
                                    'Servo thermal throttling deactivated',
                                    {'temperature': self.temp_monitor.current_temp})
        
        return should_throttle
    
    def _servo_worker(self):
        """Enhanced servo worker with error handling and thermal management"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.05)
                if command is None:
                    break
                
                pan_angle, tilt_angle = command
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                # Check thermal throttling
                self._check_thermal_throttle()
                
                # Calculate velocities with thermal consideration
                if dt > 0:
                    raw_pan_velocity = (pan_angle - self.current_pan) / dt
                    raw_tilt_velocity = (tilt_angle - self.current_tilt) / dt
                    
                    # Apply thermal throttling if needed
                    if self.thermal_throttle_active:
                        # Limit velocity during thermal throttling
                        max_velocity = self.max_movement_speed / dt if dt > 0 else self.max_movement_speed
                        raw_pan_velocity = max(-max_velocity, min(max_velocity, raw_pan_velocity))
                        raw_tilt_velocity = max(-max_velocity, min(max_velocity, raw_tilt_velocity))
                    
                    self.velocity_pan = raw_pan_velocity
                    self.velocity_tilt = raw_tilt_velocity
                
                # Only move if there's a significant change
                if (abs(pan_angle - self.current_pan) > 0.1 or 
                    abs(tilt_angle - self.current_tilt) > 0.1):
                    
                    if self.servo_initialized:
                        try:
                            self.pan_servo.angle = pan_angle
                            self.tilt_servo.angle = tilt_angle
                            consecutive_errors = 0  # Reset error counter
                            
                        except Exception as e:
                            consecutive_errors += 1
                            if consecutive_errors <= max_consecutive_errors:
                                print(f"Servo movement error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                            
                            if consecutive_errors >= max_consecutive_errors:
                                print("❌ Too many servo errors. Disabling servo control.")
                                self.servo_initialized = False
                    
                    self.current_pan = pan_angle
                    self.current_tilt = tilt_angle
                    
                    # Add extra delay during thermal throttling
                    delay = 0.005
                    if self.thermal_throttle_active:
                        delay = 0.01  # Slower updates during thermal throttling
                    
                    time.sleep(delay)
                
                self.last_update_time = current_time
                
            except Empty:
                continue
            except Exception as e:
                print(f"Servo thread error: {e}")
    
    def move_to(self, pan_angle, tilt_angle):
        """Move servos to specified angles with bounds checking and thermal consideration"""
        # Clamp angles to valid range
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        
        # Apply thermal throttling to movement limits
        if self.thermal_throttle_active:
            # Limit movement range during thermal throttling
            pan_change = pan_angle - self.current_pan
            tilt_change = tilt_angle - self.current_tilt
            
            max_change = self.max_movement_speed
            pan_change = max(-max_change, min(max_change, pan_change))
            tilt_change = max(-max_change, min(max_change, tilt_change))
            
            pan_angle = self.current_pan + pan_change
            tilt_angle = self.current_tilt + tilt_change
        
        # Clear old commands and add new one
        try:
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            self.command_queue.put_nowait((pan_angle, tilt_angle))
        except:
            pass  # Queue full, command will be ignored
    
    def get_current_state(self):
        """Get current servo state with thermal information"""
        return {
            'pan': self.current_pan,
            'tilt': self.current_tilt,
            'pan_velocity': self.velocity_pan,
            'tilt_velocity': self.velocity_tilt,
            'thermal_throttle': self.thermal_throttle_active,
            'max_speed': self.max_movement_speed
        }
    
    def center_servos(self):
        """Move servos to center position"""
        print("🎯 Centering servos...")
        self.move_to(90, 90 - CAMERA_TILT_OFFSET)
        time.sleep(1.0)  # Allow time to reach position
    
    def is_servo_available(self):
        """Check if servos are available"""
        return self.servo_initialized
    
    def shutdown(self):
        """Shutdown servo controller gracefully"""
        if self.logger:
            self.logger.log_event('servo_shutdown', 'Servo controller shutting down')
        
        print("🔧 Shutting down servo controller...")
        self.running = False
        self.command_queue.put(None)
        
        if self.servo_thread and self.servo_thread.is_alive():
            self.servo_thread.join(timeout=2.0)
        
        # Return servos to center position
        if self.servo_initialized:
            try:
                self.pan_servo.angle = 90
                self.tilt_servo.angle = 90
                time.sleep(0.5)
            except:
                pass
        
        print("✅ Servo controller shutdown complete")

# ==============================================================================
# ENHANCED TRACKER
# ==============================================================================

class UltraFastTracker:
    def __init__(self, servo_controller, logger=None, calibration_system=None, temp_monitor=None):
        self.servo = servo_controller
        self.logger = logger
        self.calibration_system = calibration_system
        self.temp_monitor = temp_monitor
        
        # Frame properties
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        # Initialize enhanced distance calculator
        self.distance_calculator = EnhancedDistanceCalculator(
            self.frame_width, self.frame_height, calibration_system
        )
        
        # Tracking state
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        self.lock_on_target = False
        self.frame_skip_counter = 0
        self.current_distance = 0.0
        self.current_3d_position = (0.0, 0.0, 0.0)
        
        # Enhanced tracking history
        self.pan_history = deque(maxlen=DETECTION_HISTORY_SIZE * 2)
        self.tilt_history = deque(maxlen=DETECTION_HISTORY_SIZE * 2)
        self.confidence_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        
        # Adaptive parameters
        self.dynamic_dead_zone = DEAD_ZONE
        self.tracking_quality = 0.0
        
        # Thermal performance tracking
        self.thermal_performance_mode = False
        
        if self.logger:
            self.logger.log_event('tracker_init', 'Enhanced tracker with temperature monitoring initialized')
    
    def update_frame_properties(self, width, height):
        """Update frame properties and related calculations"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            
            self.distance_calculator.update_frame_size(width, height)
            
            if self.logger:
                self.logger.log_event('resolution_change', f'Frame size updated: {width}x{height}')
            
            print(f"📷 Frame size updated: {width}x{height}")
    
    def _check_thermal_performance(self):
        """Check and adjust tracking performance based on temperature"""
        if not self.temp_monitor or not self.temp_monitor.temp_available:
            return
        
        temp_info = self.temp_monitor.get_temperature_info()
        should_throttle = temp_info['temperature'] >= TEMP_THROTTLE_THRESHOLD
        
        if should_throttle and not self.thermal_performance_mode:
            self.thermal_performance_mode = True
            print("🌡️  Thermal performance mode activated - reducing tracking sensitivity")
            
            if self.logger:
                self.logger.log_event('tracker_thermal_mode', 
                                    'Tracker thermal performance mode activated',
                                    {'temperature': temp_info['temperature']})
        
        elif not should_throttle and self.thermal_performance_mode:
            self.thermal_performance_mode = False
            print("🌡️  Normal tracking performance restored")
            
            if self.logger:
                self.logger.log_event('tracker_normal_mode', 
                                    'Tracker normal performance restored',
                                    {'temperature': temp_info['temperature']})
    
    def track_person(self, bbox, confidence, frame_count):
        """Enhanced person tracking with calibration integration and thermal management"""
        # Check thermal performance
        self._check_thermal_performance()
        
        # Frame skipping for performance - increase skip count in thermal mode
        skip_count = FRAME_SKIP_COUNT
        if self.thermal_performance_mode:
            skip_count = FRAME_SKIP_COUNT * 2  # Double frame skipping in thermal mode
        
        self.frame_skip_counter += 1
        if self.frame_skip_counter < skip_count:
            return
        self.frame_skip_counter = 0
        
        # Update calibration if still in progress
        if (self.calibration_system and 
            self.calibration_system.is_calibrating and 
            not self.calibration_system.calibration_complete):
            
            calibration_added = self.calibration_system.add_measurement(
                bbox, self.frame_width, self.frame_height, confidence
            )
            
            if calibration_added and self.calibration_system.calibration_complete:
                # Calibration just completed, update distance calculator
                self.distance_calculator.update_calibration()
                
                if self.logger:
                    self.logger.log_calibration_data(self.calibration_system)
                    self.logger.log_event('calibration_complete', 
                                        f'Camera calibration completed with {len(self.calibration_system.measurements)} samples')
        
        # Calculate distance using potentially updated calibration
        self.current_distance = self.distance_calculator.calculate_distance_from_bbox(bbox, confidence)
        
        # Get current servo state
        servo_state = self.servo.get_current_state()
        current_pan = servo_state['pan']
        current_tilt = servo_state['tilt']
        pan_vel = servo_state['pan_velocity']
        tilt_vel = servo_state['tilt_velocity']
        
        # Calculate 3D position
        x, y, z = self.distance_calculator.calculate_3d_position(
            bbox, current_pan, current_tilt, self.current_distance
        )
        self.current_3d_position = (x, y, z)
        
        # Calculate angular sizes
        angular_width, angular_height = self.distance_calculator.calculate_angular_size(bbox)
        
        # Calculate tracking error
        center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
        center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
        
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        
        # Update confidence history for adaptive tracking
        self.confidence_history.append(confidence)
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Dynamic dead zone based on distance, confidence, and thermal state
        distance_factor = min(2.0, max(0.5, self.current_distance / 5.0))
        confidence_factor = min(1.5, max(0.7, avg_confidence))
        thermal_factor = 1.5 if self.thermal_performance_mode else 1.0  # Less sensitive in thermal mode
        
        self.dynamic_dead_zone = DEAD_ZONE * distance_factor / confidence_factor * thermal_factor
        
        # Check if movement is needed
        if abs(error_x) > self.dynamic_dead_zone or abs(error_y) > self.dynamic_dead_zone:
            # Calculate movement adjustments with thermal consideration
            distance_factor = min(2.5, max(0.3, 3.0 / self.current_distance))
            thermal_factor = 0.7 if self.thermal_performance_mode else 1.0  # Reduced sensitivity
            
            pan_adjustment = -error_x * (PAN_SENSITIVITY / self.frame_width) * distance_factor * thermal_factor
            tilt_adjustment = error_y * (TILT_SENSITIVITY / self.frame_height) * distance_factor * thermal_factor
            
            # Apply confidence multiplier
            confidence_multiplier = min(2.0, confidence + 0.5)
            pan_adjustment *= confidence_multiplier
            tilt_adjustment *= confidence_multiplier
            
            # Calculate target angles
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            # Smooth angle changes with thermal consideration
            smoothing_factor = SMOOTHING_FACTOR
            if self.thermal_performance_mode:
                smoothing_factor *= 0.7  # More smoothing in thermal mode
            
            new_pan = self._smooth_angle(current_pan, target_pan, confidence, smoothing_factor)
            new_tilt = self._smooth_angle(current_tilt, target_tilt, confidence, smoothing_factor)
            
            # Add to history
            self.pan_history.append(new_pan)
            self.tilt_history.append(new_tilt)
            
            # Calculate weighted average from history
            if len(self.pan_history) >= 2:
                weights = [1.0, 2.0, 3.0, 4.0, 5.0][:len(self.pan_history)]
                avg_pan = sum(w * angle for w, angle in zip(weights, self.pan_history)) / sum(weights)
                avg_tilt = sum(w * angle for w, angle in zip(weights, self.tilt_history)) / sum(weights)
            else:
                avg_pan, avg_tilt = new_pan, new_tilt
            
            # Send servo command
            self.servo.move_to(avg_pan, avg_tilt)
            
            # Update tracking quality
            error_magnitude = math.sqrt(error_x**2 + error_y**2)
            max_error = math.sqrt(self.frame_width**2 + self.frame_height**2) / 2
            self.tracking_quality = max(0, 1 - (error_magnitude / max_error))
            
            # Lock on target if not already locked (with thermal consideration)
            lock_threshold = 0.7 if not self.thermal_performance_mode else 0.6
            if not self.lock_on_target and self.tracking_quality > lock_threshold:
                self.lock_on_target = True
                if self.logger:
                    self.logger.log_event('target_lock', 
                                        f'Target locked: {self.current_distance:.2f}m, quality: {self.tracking_quality:.2f}')
                print(f"🎯 Target locked: {self.current_distance:.2f}m (quality: {self.tracking_quality:.2f})")
        
        # Update tracking state
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        
        # Log frame data
        if self.logger:
            distance_data = {
                'distance': self.current_distance,
                'x_position': x,
                'y_position': y,
                'z_position': z,
                'angular_width': angular_width,
                'angular_height': angular_height,
                'bbox_width': bbox.width(),
                'bbox_height': bbox.height(),
                'tracking_quality': self.tracking_quality,
                'dynamic_dead_zone': self.dynamic_dead_zone,
                'thermal_mode': self.thermal_performance_mode
            }
            
            self.logger.log_frame_data(
                frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                confidence, True, self.is_tracking_active(), self.target_lost_frames,
                distance_data, self.calibration_system
            )
    
    def handle_lost_target(self, frame_count):
        """Handle lost target with adaptive search behavior and thermal consideration"""
        self.target_lost_frames += 1
        
        # Gradually reduce tracking quality
        self.tracking_quality *= 0.95
        
        lost_threshold = 20 if not self.thermal_performance_mode else 30  # Wait longer in thermal mode
        
        if self.target_lost_frames > lost_threshold and self.lock_on_target:
            self.lock_on_target = False
            self.tracking_quality = 0.0
            
            if self.logger:
                self.logger.log_event('target_lost', 
                                    f'Target lost after {self.target_lost_frames} frames')
            print(f"🔍 Target lost after {self.target_lost_frames} frames - scanning mode")
        
        # Optional: implement search pattern (less aggressive in thermal mode)
        search_threshold = 60 if not self.thermal_performance_mode else 90
        if self.target_lost_frames > search_threshold:
            self._execute_search_pattern()
        
        # Log frame data
        if self.logger:
            servo_state = self.servo.get_current_state()
            self.logger.log_frame_data(
                frame_count, servo_state['pan'], servo_state['tilt'], 
                servo_state['pan_velocity'], servo_state['tilt_velocity'],
                0.0, False, self.is_tracking_active(), self.target_lost_frames,
                None, self.calibration_system
            )
    
    def _execute_search_pattern(self):
        """Execute a simple search pattern when target is lost with thermal consideration"""
        if not self.servo.is_servo_available():
            return
        
        servo_state = self.servo.get_current_state()
        current_pan = servo_state['pan']
        current_tilt = servo_state['tilt']
        
        # Simple oscillating search pattern with thermal adjustment
        search_amplitude = 30 if not self.thermal_performance_mode else 20  # Smaller search in thermal mode
        search_frequency = 0.1 if not self.thermal_performance_mode else 0.07  # Slower search in thermal mode
        
        time_since_lost = (self.target_lost_frames / 30.0)  # Assume 30 fps
        search_offset = search_amplitude * math.sin(2 * math.pi * search_frequency * time_since_lost)
        
        search_pan = 90 + search_offset  # Center around 90 degrees
        search_tilt = current_tilt  # Keep tilt steady
        
        self.servo.move_to(search_pan, search_tilt)
    
    def _smooth_angle(self, current, target, confidence=1.0, smoothing_factor=None):
        """Enhanced angle smoothing with confidence-based adjustment and thermal consideration"""
        if smoothing_factor is None:
            smoothing_factor = SMOOTHING_FACTOR
            
        # Adjust smoothing factor based on confidence
        adaptive_smoothing = smoothing_factor * (0.5 + 0.5 * confidence)
        
        diff = (target - current) * adaptive_smoothing
        
        # Limit maximum step size with thermal consideration
        max_step = MAX_STEP_SIZE * confidence
        if self.thermal_performance_mode:
            max_step *= 0.8  # Smaller steps in thermal mode
        
        diff = max(-max_step, min(max_step, diff))
        
        return current + diff
    
    def is_tracking_active(self):
        """Check if tracking is currently active"""
        return (time.time() - self.last_detection_time) < DETECTION_TIMEOUT
    
    def get_current_distance_info(self):
        """Get comprehensive distance and tracking information including thermal state"""
        return {
            'distance': self.current_distance,
            'position_3d': self.current_3d_position,
            'tracking_quality': self.tracking_quality,
            'lock_on_target': self.lock_on_target,
            'target_lost_frames': self.target_lost_frames,
            'dynamic_dead_zone': self.dynamic_dead_zone,
            'thermal_performance_mode': self.thermal_performance_mode,
            'calibration_status': self.distance_calculator.get_calibration_status()
        }

# ==============================================================================
# ENHANCED CALLBACK CLASS
# ==============================================================================

class OptimizedAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.processing_times = deque(maxlen=30)
        self.last_fps_report = time.time()
    
    def new_function(self):
        return "Enhanced Servo Tracking with Auto-Calibration, GPS Waypoints, and Temperature Monitoring: "
    
    def log_processing_time(self, processing_time):
        """Log processing time for performance monitoring"""
        self.processing_times.append(processing_time)
        
        current_time = time.time()
        if current_time - self.last_fps_report > 5.0:  # Report every 5 seconds
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                
                # Include temperature info in performance report
                temp_info = ""
                if temp_monitor and temp_monitor.temp_available:
                    temp_data = temp_monitor.get_temperature_info()
                    temp_info = f" | CPU: {temp_data['temperature']:.1f}°C"
                
                print(f"📊 Performance: {fps:.1f} FPS (avg: {avg_time*1000:.1f}ms){temp_info}")
            self.last_fps_report = current_time

# ==============================================================================
# MAIN CALLBACK FUNCTION
# ==============================================================================

def enhanced_app_callback(pad, info, user_data):
    """Enhanced main callback function with comprehensive error handling and temperature monitoring"""
    start_time = time.time()
    
    try:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        
        user_data.increment()
        frame_count = user_data.get_count()
        
        # Update frame properties periodically
        if frame_count % 30 == 0:
            try:
                format, width, height = get_caps_from_pad(pad)
                if width and height:
                    tracker.update_frame_properties(width, height)
            except Exception as e:
                print(f"⚠️  Frame property update error: {e}")
        
        # Get frame for visualization if needed
        frame = None
        if user_data.use_frame:
            try:
                format, width, height = get_caps_from_pad(pad)
                if format and width and height:
                    frame = get_numpy_from_buffer(buffer, format, width, height)
            except Exception as e:
                print(f"⚠️  Frame extraction error: {e}")
        
        # Get detections from Hailo
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # Find best person detection
        best_person = None
        best_score = 0
        
        for detection in detections:
            try:
                if detection.get_label() == "person":
                    confidence = detection.get_confidence()
                    
                    if confidence >= MIN_CONFIDENCE:
                        bbox = detection.get_bbox()
                        
                        # Calculate detection quality score
                        area = bbox.width() * bbox.height()
                        aspect_ratio = bbox.width() / bbox.height() if bbox.height() > 0 else 0
                        
                        # Prefer larger detections with good aspect ratios
                        size_score = min(1.0, area * 10)  # Scale area
                        aspect_score = 1.0 - abs(aspect_ratio - 0.4) / 0.4  # Ideal person aspect ratio ~0.4
                        aspect_score = max(0.1, aspect_score)
                        
                        total_score = confidence * size_score * aspect_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_person = {
                                'bbox': bbox,
                                'confidence': confidence,
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'score': total_score
                            }
            except Exception as e:
                print(f"⚠️  Detection processing error: {e}")
                continue
        
        # Process best detection
        if best_person:
            try:
                tracker.track_person(best_person['bbox'], best_person['confidence'], frame_count)
                
                # Performance reporting with temperature info
                if frame_count % 90 == 0:  # Every 3 seconds at 30fps
                    distance_info = tracker.get_current_distance_info()
                    cal_status = distance_info['calibration_status']
                    
                    status_text = "TRACKING" if distance_info['lock_on_target'] else "DETECTING"
                    thermal_indicator = " 🌡️" if distance_info.get('thermal_performance_mode', False) else ""
                    
                    print(f"🎯 {status_text}{thermal_indicator}: Dist: {distance_info['distance']:.2f}m, "
                          f"Quality: {distance_info['tracking_quality']:.2f}, "
                          f"Cal: {'✅' if cal_status['calibrated'] else '📸'}")
                
            except Exception as e:
                print(f"❌ Tracking error: {e}")
                if data_logger:
                    data_logger.log_event('tracking_error', f'Tracking error: {str(e)}')
        else:
            try:
                tracker.handle_lost_target(frame_count)
            except Exception as e:
                print(f"❌ Lost target handling error: {e}")
        
        # Enhanced frame visualization with temperature overlay
        if user_data.use_frame and frame is not None:
            try:
                frame = _draw_enhanced_overlay(frame, best_person, tracker, calibration_system, 
                                             mavlink_handler, temp_monitor)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                user_data.set_frame(frame)
            except Exception as e:
                print(f"⚠️  Frame visualization error: {e}")
        
        # Log processing time
        processing_time = time.time() - start_time
        user_data.log_processing_time(processing_time)
        
    except Exception as e:
        print(f"❌ Critical callback error: {e}")
        if data_logger:
            data_logger.log_event('callback_error', f'Critical callback error: {str(e)}')
    
    return Gst.PadProbeReturn.OK

def _draw_enhanced_overlay(frame, best_person, tracker, calibration_system, mavlink_handler, temp_monitor):
    """Draw enhanced overlay with comprehensive information including temperature"""
    try:
        height, width = frame.shape[:2]
        
        # Draw crosshairs
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        
        # Draw calibration status
        if calibration_system and calibration_system.is_calibrating:
            status_text = calibration_system.get_status_text()
            progress = calibration_system.get_progress()
            
            # Background for calibration text
            cv2.rectangle(frame, (5, 5), (width - 5, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (width - 5, 80), (0, 255, 255), 2)
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Progress bar
            bar_width = width - 20
            bar_x = 10
            bar_y = 45
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress / 100), bar_y + 20), (0, 255, 0), -1)
            
            cv2.putText(frame, f"{progress:.0f}%", (bar_x + bar_width + 5, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw temperature information (top right corner)
        if temp_monitor and temp_monitor.temp_available:
            temp_info = temp_monitor.get_temperature_info()
            temp_color = temp_info['status_color']
            temp_text = f"CPU: {temp_info['temperature']:.1f}°C"
            
            # Temperature background
            text_size = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            temp_bg_x1 = width - text_size[0] - 15
            temp_bg_y1 = 5
            temp_bg_x2 = width - 5
            temp_bg_y2 = 35
            
            # Color background based on temperature status
            bg_color = (0, 0, 0)  # Default black background
            if temp_info['status'] == 'CRITICAL':
                bg_color = (0, 0, 128)  # Dark red background
            elif temp_info['status'] == 'WARNING':
                bg_color = (0, 64, 128)  # Dark orange background
            elif temp_info['status'] == 'HIGH':
                bg_color = (0, 64, 64)  # Dark yellow background
            
            cv2.rectangle(frame, (temp_bg_x1, temp_bg_y1), (temp_bg_x2, temp_bg_y2), bg_color, -1)
            cv2.rectangle(frame, (temp_bg_x1, temp_bg_y1), (temp_bg_x2, temp_bg_y2), temp_color, 2)
            
            cv2.putText(frame, temp_text, (temp_bg_x1 + 5, temp_bg_y1 + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, temp_color, 2)
            
            # Add thermal warning indicators
            if temp_info['status'] in ['WARNING', 'CRITICAL', 'HIGH']:
                warning_text = f"⚠️ {temp_info['status']}"
                cv2.putText(frame, warning_text, (temp_bg_x1 + 5, temp_bg_y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)
        
        # Draw detection and tracking information
        if best_person:
            distance_info = tracker.get_current_distance_info()
            distance = distance_info['distance']
            x, y, z = distance_info['position_3d']
            
            # Main status
            if not (calibration_system and calibration_system.is_calibrating):
                status_color = (0, 255, 0) if distance_info['lock_on_target'] else (255, 255, 0)
                thermal_indicator = " 🌡️" if distance_info.get('thermal_performance_mode', False) else ""
                status_text = f"{'LOCKED' if distance_info['lock_on_target'] else 'TRACKING'}{thermal_indicator}: {distance:.2f}m"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 3D Position
            cv2.putText(frame, f"3D: ({x:.1f}, {y:.1f}, {z:.1f})m", 
                       (10, height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Tracking quality
            quality = distance_info['tracking_quality']
            quality_color = (0, 255, 0) if quality > 0.7 else (255, 255, 0) if quality > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"Quality: {quality:.2f}", 
                       (10, height - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
            
            # Detection confidence and score
            cv2.putText(frame, f"Conf: {best_person['confidence']:.2f} Score: {best_person['score']:.2f}", 
                       (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # System performance indicators
            perf_indicators = []
            if distance_info.get('thermal_performance_mode', False):
                perf_indicators.append("THERMAL")
            
            if perf_indicators:
                perf_text = f"Mode: {' | '.join(perf_indicators)}"
                cv2.putText(frame, perf_text, (10, height - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Bounding box
            bbox = best_person['bbox']
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int((bbox.xmin() + bbox.width()) * width)
            y2 = int((bbox.ymin() + bbox.height()) * height)
            
            # Color based on tracking state
            if distance_info['lock_on_target']:
                box_color = (0, 255, 0)  # Green for locked
            elif quality > 0.5:
                box_color = (255, 255, 0)  # Yellow for good tracking
            else:
                box_color = (255, 0, 0)  # Red for poor tracking
            
            # Add thermal mode indicator to bounding box
            if distance_info.get('thermal_performance_mode', False):
                box_color = tuple(int(c * 0.8) for c in box_color)  # Slightly dimmed colors in thermal mode
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Distance label on bounding box
            label_text = f"{distance:.1f}m"
            if distance_info.get('thermal_performance_mode', False):
                label_text += " 🌡️"
            
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), box_color, -1)
            cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Servo information with thermal state
        servo_state = fast_servo_controller.get_current_state()
        pan = servo_state['pan']
        tilt = servo_state['tilt']
        pan_vel = servo_state['pan_velocity']
        tilt_vel = servo_state['tilt_velocity']
        
        thermal_servo_indicator = " 🌡️" if servo_state.get('thermal_throttle', False) else ""
        servo_text = f"Pan: {pan:.1f}° ({pan_vel:.1f}°/s){thermal_servo_indicator}"
        tilt_text = f"Tilt: {tilt:.1f}° ({tilt_vel:.1f}°/s)"
        
        if not (calibration_system and calibration_system.is_calibrating):
            servo_color = (255, 200, 0) if servo_state.get('thermal_throttle', False) else (255, 255, 0)
            cv2.putText(frame, servo_text, (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, servo_color, 2)
            cv2.putText(frame, tilt_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, servo_color, 2)
        
        # GPS/MAVLink status
        if mavlink_handler:
            status = mavlink_handler.get_status()
            
            # GPS fix indicator
            gps_color = (0, 255, 0) if status['gps_fix'] >= 3 else (0, 0, 255)
            gps_text = f"GPS: {status['satellites']} sats"
            
            if status['connected']:
                cv2.putText(frame, gps_text, (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, gps_color, 2)
                
                # Altitude information
                alt_text = f"Alt: {status['altitude_gps']:.1f}m"
                cv2.putText(frame, alt_text, (width - 200, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Waypoint count
                wp_text = f"WP: {status['points_logged']}"
                cv2.putText(frame, wp_text, (width - 200, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "GPS: DISCONNECTED", (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Calibration focal length info
        if (calibration_system and calibration_system.calibration_complete and
            not (calibration_system and calibration_system.is_calibrating)):
            
            cal_text = f"Focal: X={calibration_system.focal_length_x:.0f} Y={calibration_system.focal_length_y:.0f}"
            cv2.putText(frame, cal_text, (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    except Exception as e:
        print(f"⚠️  Overlay drawing error: {e}")
    
    return frame

# ==============================================================================
# MAIN INITIALIZATION AND STARTUP
# ==============================================================================

def initialize_system():
    """Initialize all system components with proper error handling including temperature monitoring"""
    print("🚀 Initializing Enhanced Servo Tracking System with Temperature Monitoring...")
    
    # Initialize temperature monitoring first
    print("🌡️  Initializing RPi5 temperature monitoring...")
    temp_monitor = RPi5TemperatureMonitor()
    
    # Initialize calibration system
    print("📸 Initializing auto-calibration system...")
    calibration_system = AutoCalibrationSystem()
    
    # Initialize MAVLink connection
    mavlink_handler = None
    try:
        print("🛰️ Initializing MAVLink connection...")
        mavlink_handler = MAVLinkGPSHandler()
        if mavlink_handler.connection_healthy:
            print("✅ MAVLink GPS handler initialized successfully")
        else:
            print("⚠️  MAVLink connection unhealthy, continuing without GPS waypoints")
    except Exception as e:
        print(f"❌ MAVLink initialization failed: {e}")
        print("   Continuing without GPS waypoint functionality")
    
    # Initialize data logger with temperature monitoring
    print("📊 Initializing data logging system with temperature monitoring...")
    data_logger = ServoDataLogger(gps_handler=mavlink_handler, temp_monitor=temp_monitor)
    
    # Initialize servo controller with temperature monitoring
    print("🔧 Initializing servo controller with thermal management...")
    servo_controller = FastServoController(data_logger, temp_monitor)
    
    if servo_controller.is_servo_available():
        print("✅ Servo controller initialized successfully")
        servo_controller.center_servos()
    else:
        print("⚠️  Servo hardware not available, running in simulation mode")
    
    # Initialize tracker with temperature monitoring
    print("🎯 Initializing enhanced tracker with thermal management...")
    tracker = UltraFastTracker(servo_controller, data_logger, calibration_system, temp_monitor)
    
    return calibration_system, mavlink_handler, data_logger, servo_controller, tracker, temp_monitor

def print_system_status(calibration_system, mavlink_handler, servo_controller, temp_monitor):
    """Print comprehensive system status including temperature monitoring"""
    print("\n" + "="*70)
    print("📊 ENHANCED SYSTEM STATUS WITH TEMPERATURE MONITORING")
    print("="*70)
    
    # Temperature monitoring status
    if temp_monitor and temp_monitor.temp_available:
        temp_info = temp_monitor.get_temperature_info()
        status_icon = {
            'NORMAL': '✅',
            'HIGH': '⚠️ ',
            'WARNING': '🔥',
            'CRITICAL': '🚨'
        }.get(temp_info['status'], '❓')
        
        print(f"🌡️  Temperature Monitor: {status_icon} {temp_info['status']}")
        print(f"   Current Temperature: {temp_info['temperature']:.1f}°C")
        print(f"   Temperature Range: {temp_info['min_temp']:.1f}°C - {temp_info['max_temp']:.1f}°C")
        print(f"   CPU Frequency: {temp_info['cpu_freq']:.0f} MHz")
        print(f"   CPU Usage: {temp_info['cpu_usage']:.1f}%")
        print(f"   Thermal Thresholds: Warning={temp_info['thresholds']['warning']}°C, "
              f"Critical={temp_info['thresholds']['critical']}°C")
        
        if any(temp_info['alerts'].values()):
            alert_types = [k for k, v in temp_info['alerts'].items() if v]
            print(f"   🚨 Active Alerts: {', '.join(alert_types)}")
    else:
        print(f"🌡️  Temperature Monitor: ❌ UNAVAILABLE")
    
    # Calibration status
    if calibration_system.is_calibrating:
        print(f"📸 Camera Calibration: IN PROGRESS ({calibration_system.get_progress():.0f}%)")
        print(f"   Samples needed: {CALIBRATION_SAMPLES_REQUIRED}")
        print(f"   Current samples: {len(calibration_system.measurements)}")
        print(f"   Timeout: {CALIBRATION_TIMEOUT}s")
    else:
        print(f"📸 Camera Calibration: ✅ COMPLETE")
        if calibration_system.focal_length_x:
            print(f"   Focal Length X: {calibration_system.focal_length_x:.1f} pixels")
            print(f"   Focal Length Y: {calibration_system.focal_length_y:.1f} pixels")
    
    # Servo status with thermal management
    servo_status = "✅ ACTIVE" if servo_controller.is_servo_available() else "❌ UNAVAILABLE"
    servo_state = servo_controller.get_current_state()
    thermal_info = " (Thermal throttling active)" if servo_state.get('thermal_throttle', False) else ""
    print(f"🔧 Servo Controller: {servo_status}{thermal_info}")
    if servo_controller.is_servo_available():
        print(f"   Max Movement Speed: {servo_state.get('max_speed', MAX_STEP_SIZE):.1f}°/frame")
    
    # GPS status
    if mavlink_handler and mavlink_handler.connection_healthy:
        status = mavlink_handler.get_status()
        fix_status = "✅ 3D FIX" if status['gps_fix'] >= 3 else f"❌ NO FIX ({status['gps_fix']})"
        print(f"🛰️ GPS System: {fix_status}")
        print(f"   Satellites: {status['satellites']}")
        print(f"   Position: {status['latitude']:.6f}, {status['longitude']:.6f}")
        print(f"   Altitude: GPS={status['altitude_gps']:.1f}m, Rel={status['altitude_relative']:.1f}m")
        print(f"   Heading: {status['heading']:.1f}°")
        print(f"   Waypoint Mode: {WAYPOINT_MODE}")
    else:
        print(f"🛰️ GPS System: ❌ UNAVAILABLE")
    
    # System thermal management settings
    print(f"⚙️  Thermal Management:")
    print(f"   Temperature update interval: {TEMP_UPDATE_INTERVAL}s")
    print(f"   Performance throttling: {TEMP_THROTTLE_THRESHOLD}°C")
    print(f"   Warning threshold: {TEMP_WARNING_THRESHOLD}°C")
    print(f"   Critical threshold: {TEMP_CRITICAL_THRESHOLD}°C")
    
    print("="*70)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Set up environment
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    # Global variables for callback access
    calibration_system = None
    mavlink_handler = None
    data_logger = None
    fast_servo_controller = None
    tracker = None
    temp_monitor = None
    
    try:
        # Initialize all systems including temperature monitoring
        (calibration_system, mavlink_handler, data_logger, 
         fast_servo_controller, tracker, temp_monitor) = initialize_system()
        
        # Print comprehensive system status
        print_system_status(calibration_system, mavlink_handler, fast_servo_controller, temp_monitor)
        
        # Wait for calibration to complete if in progress
        if calibration_system.is_calibrating:
            print("\n⏳ Waiting for camera calibration to complete...")
            print("   Please ensure a person is visible in the camera view")
            print("   Person should be 2-4 meters away for best results")
            print("   Calibration will complete automatically\n")
        
        # Temperature monitoring guidance
        if temp_monitor and temp_monitor.temp_available:
            print("🌡️  Temperature monitoring is active")
            print(f"   Current CPU temperature: {temp_monitor.current_temp:.1f}°C")
            print("   System will automatically adjust performance if temperature rises")
            print("   Consider adequate cooling for optimal performance\n")
        
        # Initialize GStreamer app
        user_data = OptimizedAppCallback()
        app = GStreamerDetectionApp(enhanced_app_callback, user_data)
        
        print("\n🚀 Starting Enhanced Servo Tracking System with Temperature Monitoring...")
        print("\nPress Ctrl+C to stop\n")
        
        # Start the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user...")
        
    except Exception as e:
        print(f"\n❌ Critical system error: {e}")
        if data_logger:
            data_logger.log_event('critical_error', f'Critical system error: {str(e)}')
    
    finally:
        print("\n📊 Finalizing system shutdown...")
        
        # Finalize calibration data if available
        if data_logger and calibration_system:
            try:
                data_logger.log_calibration_data(calibration_system)
            except Exception as e:
                print(f"⚠️  Calibration data save error: {e}")
        
        # Finalize session logs
        if data_logger:
            try:
                data_logger.finalize_session()
            except Exception as e:
                print(f"⚠️  Session finalization error: {e}")
        
        # Shutdown servo controller
        if fast_servo_controller:
            try:
                fast_servo_controller.shutdown()
            except Exception as e:
                print(f"⚠️  Servo shutdown error: {e}")
        
        # Shutdown MAVLink connection
        if mavlink_handler:
            try:
                mavlink_handler.shutdown()
            except Exception as e:
                print(f"⚠️  MAVLink shutdown error: {e}")
        
        # Shutdown temperature monitor
        if temp_monitor:
            try:
                temp_monitor.shutdown()
            except Exception as e:
                print(f"⚠️  Temperature monitor shutdown error: {e}")
        
        print("✅ System shutdown complete")
        print("\n📁 Log files saved to: servo_logs/")
        
        # Print final summary including temperature data
        if data_logger:
            try:
                stats = data_logger.session_data['statistics']
                print("\n📊 Final Session Summary:")
                print(f"   Total Detections: {stats['total_detections']}")
                print(f"   GPS Waypoints Created: {stats['gps_points_created']}")
                print(f"   Calibration Samples: {stats['calibration_samples']}")
                if stats['distance_samples'] > 0:
                    print(f"   Distance Range: {stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m")
                    print(f"   Average Distance: {stats['avg_distance']:.2f}m")
                if stats['temp_readings'] > 0:
                    print(f"   Temperature Monitoring:")
                    print(f"     Temperature Range: {stats['min_temp']:.1f}°C - {stats['max_temp']:.1f}°C")
                    print(f"     Temperature Readings: {stats['temp_readings']}")
                    print(f"     Thermal Alerts: {stats['temp_alerts']}")
                print(f"   Total Events Logged: {len(data_logger.session_data['events'])}")
                print(f"   Log Files Created: {len(data_logger.session_data['log_files'])}")
            except:
                pass
        
        print("\nThank you for using the Enhanced Servo Tracking System with Temperature Monitoring! 🎯🌡️")
