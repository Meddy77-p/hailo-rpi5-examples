#!/usr/bin/env python3
"""
Complete Mission Planner Servo Tracking System - FIXED VERSION
Single file with all components and working waypoint uploads
Based on proven File 2 waypoint logic integrated into complete tracking system
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os, sys, time, math, json, csv, threading
from pathlib import Path
from datetime import datetime
from collections import deque
from queue import Queue, Empty
import statistics

# Core imports
import numpy as np
import cv2
import hailo
import board, busio
from adafruit_motor import servo
import adafruit_pca9685
from pymavlink import mavutil
import psutil
import serial

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# ==================== CONFIGURATION ====================
# Tracking parameters
DEAD_ZONE, SMOOTHING_FACTOR, MAX_STEP_SIZE = 15, 0.35, 5
MIN_CONFIDENCE, DETECTION_TIMEOUT = 0.3, 2.0
PAN_SENSITIVITY, TILT_SENSITIVITY = 45, 35

# Camera & physical setup
CAMERA_FOV_H, CAMERA_FOV_V = 79.9, 64.3
PERSON_HEIGHT, PERSON_WIDTH = 1.7, 0.45
SERVO_MOUNT_HEIGHT, CAMERA_TILT_OFFSET = 0.978, 5.0

# MAVLink & GPS - MISSION PLANNER COMPATIBLE (FIXED)
MAVLINK_CONNECTION, MAVLINK_BAUD = '/dev/serial0', 57600
MIN_DISTANCE_FOR_GPS, MAX_WAYPOINTS = 2.0, 25
WAYPOINT_MODE = "ADD"
MAVLINK_TIMEOUT = 10.0
MAVLINK_RETRY_INTERVAL = 3.0  # Reduced from 5.0

# Temperature thresholds
TEMP_WARNING, TEMP_CRITICAL, TEMP_THROTTLE = 70.0, 80.0, 75.0
TEMP_UPDATE_INTERVAL = 2.0

# Calibration
CAL_SAMPLES_REQUIRED, CAL_TIMEOUT = 30, 60
CAL_MIN_CONFIDENCE, CAL_MIN_BBOX_HEIGHT = 0.5, 50

# ==================== TEMPERATURE MONITOR ====================
class TemperatureMonitor:
    def __init__(self):
        self.temp_path = '/sys/class/thermal/thermal_zone0/temp'
        self.current_temp = self.max_temp = 0.0
        self.min_temp = 100.0
        self.running = True
        self.alerts = {'warning': False, 'critical': False, 'throttle': False}
        self.available = self._check_available()
        
        if self.available:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print("✅ Temperature monitoring started")
        else:
            print("⚠️  Temperature monitoring not available")
    
    def _check_available(self):
        try:
            with open(self.temp_path, 'r') as f:
                temp = float(f.read().strip()) / 1000.0
                return 0 < temp < 200
        except Exception as e:
            print(f"Temperature sensor check failed: {e}")
            return False
    
    def _read_temp(self):
        try:
            with open(self.temp_path, 'r') as f:
                return float(f.read().strip()) / 1000.0
        except: 
            return None
    
    def _monitor_loop(self):
        while self.running:
            try:
                temp = self._read_temp()
                if temp:
                    self.current_temp = temp
                    self.max_temp = max(self.max_temp, temp)
                    self.min_temp = min(self.min_temp, temp) if self.min_temp != 100.0 else temp
                    self._check_alerts(temp)
                time.sleep(TEMP_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Temperature monitoring error: {e}")
                time.sleep(1)
    
    def _check_alerts(self, temp):
        # Critical alert
        if temp >= TEMP_CRITICAL and not self.alerts['critical']:
            self.alerts['critical'] = True
            print(f"🚨 CRITICAL: CPU {temp:.1f}°C - System may throttle!")
        
        # Warning alert
        elif temp >= TEMP_WARNING and not self.alerts['warning']:
            self.alerts['warning'] = True
            print(f"⚠️  WARNING: CPU {temp:.1f}°C - Consider cooling")
        
        # Throttle alert
        elif temp >= TEMP_THROTTLE and not self.alerts['throttle']:
            self.alerts['throttle'] = True
            print(f"🐌 THROTTLE: CPU {temp:.1f}°C - Reducing performance")
        
        # Reset alerts when cooled
        if temp < TEMP_WARNING - 5:
            if any(self.alerts.values()):
                self.alerts = {'warning': False, 'critical': False, 'throttle': False}
                print(f"✅ Temperature recovered: {temp:.1f}°C")
    
    def get_info(self):
        status = 'CRITICAL' if self.current_temp >= TEMP_CRITICAL else \
                'WARNING' if self.current_temp >= TEMP_WARNING else \
                'HIGH' if self.current_temp >= TEMP_THROTTLE else 'NORMAL'
        
        return {
            'available': self.available,
            'temperature': self.current_temp,
            'status': status,
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'should_throttle': self.current_temp >= TEMP_THROTTLE,
            'alerts': self.alerts.copy()
        }
    
    def shutdown(self):
        self.running = False

# ==================== AUTO CALIBRATION ====================
class AutoCalibration:
    def __init__(self):
        self.is_calibrating = True
        self.complete = False
        self.measurements = []
        self.start_time = time.time()
        self.focal_x = self.focal_y = None
        print("📸 Auto-calibration initialized")
    
    def add_measurement(self, bbox, frame_w, frame_h, confidence):
        if not self.is_calibrating or self.complete:
            return False
        
        if confidence < CAL_MIN_CONFIDENCE:
            return False
        
        bbox_h = bbox.height() * frame_h
        if bbox_h < CAL_MIN_BBOX_HEIGHT:
            return False
        
        # Validate aspect ratio
        bbox_w = bbox.width() * frame_w
        aspect = bbox_w / bbox_h
        if not (0.2 <= aspect <= 0.8):
            return False
        
        self.measurements.append({
            'bbox_height': bbox_h,
            'bbox_width': bbox_w,
            'confidence': confidence,
            'frame_width': frame_w,
            'frame_height': frame_h
        })
        
        # Complete calibration
        elapsed = time.time() - self.start_time
        if len(self.measurements) >= CAL_SAMPLES_REQUIRED or elapsed > CAL_TIMEOUT:
            return self._complete_calibration()
        return True
    
    def _complete_calibration(self):
        if len(self.measurements) < 10:
            print("❌ Insufficient calibration data. Using defaults.")
            self._use_defaults()
            return False
        
        # Remove outliers and calculate focal lengths
        heights = [m['bbox_height'] for m in self.measurements]
        widths = [m['bbox_width'] for m in self.measurements]
        
        try:
            avg_height = statistics.median(self._remove_outliers(heights))
            avg_width = statistics.median(self._remove_outliers(widths))
        except Exception as e:
            print(f"Calibration calculation error: {e}")
            self._use_defaults()
            return False
        
        # Calculate with 3m reference distance
        self.focal_y = (avg_height * 3.0) / PERSON_HEIGHT
        self.focal_x = (avg_width * 3.0) / PERSON_WIDTH
        
        # Validate against theoretical values
        frame_w = self.measurements[0]['frame_width']
        frame_h = self.measurements[0]['frame_height']
        theo_fx = frame_w / (2 * math.tan(math.radians(CAMERA_FOV_H / 2)))
        theo_fy = frame_h / (2 * math.tan(math.radians(CAMERA_FOV_V / 2)))
        
        # Adjust if too far from theoretical
        if abs(self.focal_x - theo_fx) / theo_fx > 0.3:
            self.focal_x = 0.7 * theo_fx + 0.3 * self.focal_x
        if abs(self.focal_y - theo_fy) / theo_fy > 0.3:
            self.focal_y = 0.7 * theo_fy + 0.3 * self.focal_y
        
        self.complete = True
        self.is_calibrating = False
        
        print(f"📸 CALIBRATION COMPLETE!")
        print(f"   Samples: {len(self.measurements)}, Focal X: {self.focal_x:.1f}, Y: {self.focal_y:.1f}")
        return True
    
    def _remove_outliers(self, data):
        if len(data) < 4: 
            return data
        try:
            q1, q3 = statistics.quantiles(data, n=4)[0], statistics.quantiles(data, n=4)[2]
            iqr = q3 - q1
            filtered = [x for x in data if q1 - 1.5*iqr <= x <= q3 + 1.5*iqr]
            return filtered if len(filtered) >= len(data) * 0.5 else data
        except:
            return data
    
    def _use_defaults(self):
        self.focal_x = 640 / (2 * math.tan(math.radians(CAMERA_FOV_H / 2)))
        self.focal_y = 480 / (2 * math.tan(math.radians(CAMERA_FOV_V / 2)))
        self.complete = True
        self.is_calibrating = False
    
    def get_progress(self):
        if self.complete: 
            return 100
        sample_prog = (len(self.measurements) / CAL_SAMPLES_REQUIRED) * 100
        time_prog = ((time.time() - self.start_time) / CAL_TIMEOUT) * 100
        return min(100, max(sample_prog, time_prog))

# ==================== FIXED MISSION PLANNER GPS HANDLER ====================
class GPSWaypointHandler:
    """FIXED Mission Planner compatible GPS waypoint handler using File 2 logic"""
    
    def __init__(self):
        self.connection = None
        self.lat = self.lon = self.alt = self.heading = 0.0
        self.gps_fix = self.satellites = 0
        self.healthy = False
        self.running = True
        self.waypoints_created = 0
        self.last_waypoint_time = 0
        self.connection_attempts = 0
        self.last_connection_attempt = 0
        self.receiver_thread = None
        
        # Mission management like File 2
        self.current_mission = []
        self.mission_version = 0
        self.last_mission_upload = 0
        
        # FIXED: Add position tracking
        self.position_received = False
        
        # Connect
        self._attempt_connection()
    
    def _attempt_connection(self):
        """Connect to MAVLink with retry logic (FIXED based on File 2)"""
        current_time = time.time()
        if current_time - self.last_connection_attempt < MAVLINK_RETRY_INTERVAL:
            return False
        
        self.last_connection_attempt = current_time
        self.connection_attempts += 1
        
        try:
            print(f"🛰️ MAVLink connection attempt {self.connection_attempts}...")
            
            # Check if serial port exists (same as File 2)
            if not os.path.exists(MAVLINK_CONNECTION):
                print(f"❌ Serial port {MAVLINK_CONNECTION} does not exist")
                return False
            
            # Test serial port availability (same as File 2)
            try:
                test_serial = serial.Serial(MAVLINK_CONNECTION, MAVLINK_BAUD, timeout=1)
                test_serial.close()
            except Exception as e:
                print(f"❌ Serial port test failed: {e}")
                return False
            
            # Create MAVLink connection (same as File 2)
            self.connection = mavutil.mavlink_connection(
                MAVLINK_CONNECTION, 
                baud=MAVLINK_BAUD,
                timeout=MAVLINK_TIMEOUT,
                retries=0
            )
            
            # Wait for heartbeat (same timeout as File 2)
            print("⏳ Waiting for heartbeat...")
            heartbeat = self.connection.wait_heartbeat(timeout=MAVLINK_TIMEOUT)
            
            if heartbeat:
                self.healthy = True
                print(f"✅ MAVLink connected! System ID: {self.connection.target_system}")
                
                # Start receiver thread
                if self.receiver_thread is None or not self.receiver_thread.is_alive():
                    self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
                    self.receiver_thread.start()
                
                # FIXED: Get position immediately like File 2
                self._get_current_position()
                
                # FIXED: Clear mission first like File 2
                self._clear_mission_first()
                return True
            else:
                print("❌ No heartbeat received")
                return False
                
        except Exception as e:
            print(f"❌ MAVLink connection failed: {e}")
            self.healthy = False
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
            return False
    
    def _get_current_position(self):
        """Get current GPS position immediately (FIXED - like File 2)"""
        print("📍 Getting current position...")
        
        # Request position data with same interval as File 2
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            200000,  # 0.2 second interval (same as File 2)
            0, 0, 0, 0, 0
        )
        
        # Wait for position message (same logic as File 2)
        start_time = time.time()
        while time.time() - start_time < 5:
            msg = self.connection.recv_match(
                type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], 
                blocking=False, 
                timeout=0.5
            )
            
            if msg:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.alt / 1000.0
                    self.position_received = True
                    print(f"📍 Position: {self.lat:.6f}, {self.lon:.6f}, {self.alt:.1f}m")
                    return
                elif msg.get_type() == 'GPS_RAW_INT':
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.alt / 1000.0
                    self.position_received = True
                    print(f"📍 Position: {self.lat:.6f}, {self.lon:.6f}, {self.alt:.1f}m")
                    return
        
        print("⚠️  Could not get current position")
    
    def _clear_mission_first(self):
        """Clear existing mission first (FIXED - like File 2)"""
        try:
            print("🗑️ Clearing existing mission...")
            self.connection.mav.mission_clear_all_send(
                self.connection.target_system,
                self.connection.target_component
            )
            
            # Wait for ACK like File 2
            start_time = time.time()
            while time.time() - start_time < 3:
                msg = self.connection.recv_match(type='MISSION_ACK', blocking=False, timeout=0.5)
                if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    print("✅ Mission cleared")
                    break
            
            self.current_mission = []
            time.sleep(0.5)
        except Exception as e:
            print(f"Mission clear error: {e}")
    
    def _receiver_loop(self):
        """GPS data receiver loop (simplified)"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                if not self.connection or not self.healthy:
                    time.sleep(1)
                    continue
                
                msg = self.connection.recv_match(blocking=True, timeout=1.0)
                
                if msg is None:
                    consecutive_errors = 0
                    continue
                
                consecutive_errors = 0
                
                msg_type = msg.get_type()
                if msg_type == 'GPS_RAW_INT':
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.alt / 1000.0
                    self.gps_fix = msg.fix_type
                    self.satellites = msg.satellites_visible
                    self.position_received = True
                    
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self.lat = msg.lat / 1e7
                    self.lon = msg.lon / 1e7
                    self.alt = msg.alt / 1000.0
                    self.heading = msg.hdg / 100.0
                    self.position_received = True
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"⚠️  MAVLink receive error {consecutive_errors}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Too many consecutive errors, marking unhealthy")
                    self.healthy = False
                    consecutive_errors = 0
                
                time.sleep(0.5)
    
    def create_waypoint(self, x_meters, y_meters, confidence):
        """Create waypoint using FIXED File 2 logic"""
        # FIXED: Check position received flag
        if not self.healthy or not self.position_received:
            if not self._attempt_connection():
                return None
        
        if not self.healthy or self.gps_fix < 3:
            return None
        
        distance = math.sqrt(x_meters**2 + y_meters**2)
        
        if distance < MIN_DISTANCE_FOR_GPS:
            return None
        
        # FIXED: Reduce rate limiting (was 8 seconds, now 5)
        current_time = time.time()
        if current_time - self.last_waypoint_time < 5.0:
            return None
        
        print(f"🎯 Creating Mission Planner waypoint: distance={distance:.1f}m")
        
        # Calculate GPS coordinates (same as original)
        try:
            bearing = math.degrees(math.atan2(x_meters, y_meters))
            abs_bearing = math.radians((self.heading + bearing) % 360)
            
            EARTH_RADIUS = 6371000
            lat_rad = math.radians(self.lat)
            lon_rad = math.radians(self.lon)
            
            new_lat_rad = math.asin(
                math.sin(lat_rad) * math.cos(distance / EARTH_RADIUS) +
                math.cos(lat_rad) * math.sin(distance / EARTH_RADIUS) * math.cos(abs_bearing)
            )
            new_lon_rad = lon_rad + math.atan2(
                math.sin(abs_bearing) * math.sin(distance / EARTH_RADIUS) * math.cos(lat_rad),
                math.cos(distance / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad)
            )
            
            wp_lat = math.degrees(new_lat_rad)
            wp_lon = math.degrees(new_lon_rad)
            wp_alt = self.alt + 10.0  # 10m safety offset
            
        except Exception as e:
            print(f"❌ GPS calculation error: {e}")
            return None
        
        # FIXED: Use File 2's waypoint creation and upload logic
        if self._add_waypoint_like_file2(wp_lat, wp_lon, wp_alt, confidence):
            self.waypoints_created += 1
            self.last_waypoint_time = current_time
            print(f"✅ Mission Planner waypoint {self.waypoints_created}: {distance:.1f}m @ {(self.heading + bearing) % 360:.0f}°")
            print(f"📱 Visible in Mission Planner Flight Plan tab")
            
            return {
                'latitude': wp_lat, 'longitude': wp_lon, 'altitude': wp_alt,
                'distance': distance, 'confidence': confidence,
                'vehicle_lat': self.lat, 'vehicle_lon': self.lon
            }
        
        return None
    
    def _add_waypoint_like_file2(self, lat, lon, alt, confidence):
        """Add waypoint using File 2's proven method"""
        try:
            # Create waypoint with same structure as File 2
            waypoint = {
                'seq': len(self.current_mission),
                'frame': mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                'command': mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                'current': 0,
                'autocontinue': 1,
                'param1': 2,      # Hold time (2 seconds) - same as File 2
                'param2': 5,      # Acceptance radius (5 meters) - same as File 2
                'param3': 0,      # Pass through (0 = stop) - same as File 2
                'param4': 0,      # Yaw angle (0 = any) - same as File 2
                'x': lat,
                'y': lon,
                'z': alt,
                'mission_type': mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            }
            
            # Add to mission
            self.current_mission.append(waypoint)
            
            # FIXED: Upload immediately like File 2 (no throttling)
            return self._upload_mission_like_file2()
            
        except Exception as e:
            print(f"❌ Waypoint creation error: {e}")
            return False
    
    def _upload_mission_like_file2(self):
        """Upload mission using File 2's exact logic"""
        if not self.current_mission or not self.connection:
            return False
        
        try:
            print(f"📤 Uploading {len(self.current_mission)} waypoints to Mission Planner...")
            
            # FIXED: Clear pending messages like File 2
            while True:
                msg = self.connection.recv_match(blocking=False, timeout=0.01)
                if not msg:
                    break
            
            # Step 1: Send mission count (same as File 2)
            self.connection.mav.mission_count_send(
                self.connection.target_system,
                self.connection.target_component,
                len(self.current_mission),
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )
            
            print(f"📤 Sent mission count: {len(self.current_mission)}")
            
            # Step 2: Handle mission requests (exact File 2 logic)
            waypoints_sent = 0
            timeout_start = time.time()
            ack_received = False
            
            while (waypoints_sent < len(self.current_mission) or not ack_received) and time.time() - timeout_start < 20:
                msg = self.connection.recv_match(
                    type=['MISSION_REQUEST', 'MISSION_REQUEST_INT', 'MISSION_ACK'],
                    blocking=False,
                    timeout=0.5
                )
                
                if not msg:
                    continue
                
                msg_type = msg.get_type()
                
                if msg_type in ['MISSION_REQUEST', 'MISSION_REQUEST_INT']:
                    seq = msg.seq
                    
                    if seq >= len(self.current_mission):
                        print(f"❌ Requested waypoint {seq} out of range")
                        continue
                    
                    waypoint = self.current_mission[seq]
                    
                    # FIXED: Send waypoint using MISSION_ITEM_INT (same as File 2)
                    self.connection.mav.mission_item_int_send(
                        self.connection.target_system,
                        self.connection.target_component,
                        waypoint['seq'],
                        waypoint['frame'],
                        waypoint['command'],
                        waypoint['current'],
                        waypoint['autocontinue'],
                        waypoint['param1'],
                        waypoint['param2'],
                        waypoint['param3'],
                        waypoint['param4'],
                        int(waypoint['x'] * 1e7),  # Latitude as integer
                        int(waypoint['y'] * 1e7),  # Longitude as integer
                        waypoint['z'],
                        waypoint['mission_type']
                    )
                    
                    if seq >= waypoints_sent:
                        waypoints_sent = seq + 1
                        print(f"📤 Sent waypoint {seq + 1}/{len(self.current_mission)}: {waypoint['x']:.6f}, {waypoint['y']:.6f}")
                
                elif msg_type == 'MISSION_ACK':
                    ack_received = True
                    if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                        print("✅ Mission upload successful!")
                        print("🎯 Waypoints should now be visible in Mission Planner")
                        print("📱 In Mission Planner: Go to 'Flight Plan' tab and click 'Read WPs'")
                        return True
                    else:
                        print(f"❌ Mission rejected: error code {msg.type}")
                        return False
            
            # FIXED: Check completion like File 2
            if waypoints_sent >= len(self.current_mission):
                print("✅ All waypoints sent (ACK timeout, but likely successful)")
                print("🎯 Check Mission Planner - waypoints may still be there")
                print("📱 In Mission Planner: Go to 'Flight Plan' tab and click 'Read WPs'")
                return True
            else:
                print(f"❌ Mission upload incomplete: {waypoints_sent}/{len(self.current_mission)} sent")
                return False
            
        except Exception as e:
            print(f"❌ Mission upload error: {e}")
            return False
    
    def get_status(self):
        return {
            'connected': self.healthy,
            'gps_fix': self.gps_fix,
            'satellites': self.satellites,
            'latitude': self.lat,
            'longitude': self.lon,
            'altitude': self.alt,
            'waypoints_created': self.waypoints_created,
            'mission_waypoints': len(self.current_mission),
            'connection_attempts': self.connection_attempts,
            'position_received': self.position_received  # New status
        }
    
    def shutdown(self):
        print("🛰️ Shutting down Mission Planner GPS handler...")
        self.running = False
        
        # Final mission upload (like File 2)
        if self.current_mission and self.healthy:
            print("📤 Final mission upload...")
            self._upload_mission_like_file2()
        
        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2)
        
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None

# ==================== DATA LOGGER ====================
class DataLogger:
    def __init__(self, gps_handler=None, temp_monitor=None):
        self.gps_handler = gps_handler
        self.temp_monitor = temp_monitor
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path("servo_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Log files
        self.csv_file = self.log_dir / f"tracking_{timestamp}.csv"
        self.gps_file = self.log_dir / f"waypoints_{timestamp}.csv"
        self.session_file = self.log_dir / f"session_{timestamp}.json"
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Session data
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'statistics': {
                'detections': 0, 'waypoints': 0, 'max_distance': 0.0,
                'temp_readings': 0, 'max_temp': 0.0
            },
            'events': []
        }
        
        print(f"📊 Data logging initialized: {self.log_dir}")
    
    def _init_csv_files(self):
        try:
            # Main tracking CSV
            headers = ['timestamp', 'frame', 'pan', 'tilt', 'confidence', 'distance',
                      'x_pos', 'y_pos', 'z_pos', 'gps_lat', 'gps_lon', 'cpu_temp']
            with open(self.csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
            
            # GPS waypoints CSV  
            gps_headers = ['timestamp', 'wp_lat', 'wp_lon', 'wp_alt', 'distance',
                          'vehicle_lat', 'vehicle_lon', 'confidence']
            with open(self.gps_file, 'w', newline='') as f:
                csv.writer(f).writerow(gps_headers)
                
        except Exception as e:
            print(f"CSV initialization error: {e}")
    
    def log_frame(self, frame_count, pan, tilt, confidence, distance, 
                  x_pos, y_pos, z_pos, gps_waypoint=None):
        """Log frame data with waypoint creation"""
        try:
            # Get temperature
            temp = 0.0
            if self.temp_monitor and self.temp_monitor.available:
                temp = self.temp_monitor.current_temp
            
            # Get GPS coordinates if waypoint created
            gps_lat = gps_lon = 0.0
            if gps_waypoint:
                gps_lat = gps_waypoint['latitude']
                gps_lon = gps_waypoint['longitude']
                self.session_data['statistics']['waypoints'] += 1
                
                # Log waypoint to GPS CSV
                with open(self.gps_file, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        time.time(), gps_waypoint['latitude'], gps_waypoint['longitude'],
                        gps_waypoint['altitude'], gps_waypoint['distance'],
                        gps_waypoint['vehicle_lat'], gps_waypoint['vehicle_lon'],
                        gps_waypoint['confidence']
                    ])
            
            # Log frame data
            with open(self.csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    time.time(), frame_count, pan, tilt, confidence, distance,
                    x_pos, y_pos, z_pos, gps_lat, gps_lon, temp
                ])
            
            # Update statistics
            if confidence > 0:
                self.session_data['statistics']['detections'] += 1
                self.session_data['statistics']['max_distance'] = max(
                    self.session_data['statistics']['max_distance'], distance)
            
            if temp > 0:
                self.session_data['statistics']['temp_readings'] += 1
                self.session_data['statistics']['max_temp'] = max(
                    self.session_data['statistics']['max_temp'], temp)
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_event(self, event_type, description):
        try:
            event = {'timestamp': time.time(), 'type': event_type, 'description': description}
            self.session_data['events'].append(event)
            print(f"📝 {event_type}: {description}")
        except Exception as e:
            print(f"Event logging error: {e}")
    
    def finalize(self):
        """Save final session data"""
        try:
            self.session_data['end_time'] = datetime.now().isoformat()
            
            if self.gps_handler:
                self.session_data['gps_status'] = self.gps_handler.get_status()
            
            if self.temp_monitor:
                self.session_data['temp_final'] = self.temp_monitor.get_info()
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
            
            stats = self.session_data['statistics']
            print(f"\n📊 Session Complete:")
            print(f"   Detections: {stats['detections']}, Waypoints: {stats['waypoints']}")
            print(f"   Max Distance: {stats['max_distance']:.1f}m")
            if stats['temp_readings'] > 0:
                print(f"   Max Temperature: {stats['max_temp']:.1f}°C")
                
        except Exception as e:
            print(f"Session finalization error: {e}")

# ==================== SERVO CONTROLLER ====================
class ServoController:
    def __init__(self, temp_monitor=None):
        self.temp_monitor = temp_monitor
        self.current_pan = self.current_tilt = 90.0
        self.available = False
        
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = adafruit_pca9685.PCA9685(self.i2c)
            self.pca.frequency = 50
            self.pan_servo = servo.Servo(self.pca.channels[0])
            self.tilt_servo = servo.Servo(self.pca.channels[1])
            self.available = True
            
            # Center servos
            self.pan_servo.angle = self.tilt_servo.angle = 90.0
            time.sleep(0.5)
            print("✅ Servos initialized and centered")
            
        except Exception as e:
            print(f"❌ Servo init failed: {e}")
            print("⚠️  Continuing without servo control")
    
    def move_to(self, pan, tilt):
        if not self.available: 
            return
        
        try:
            # Apply thermal throttling
            if self.temp_monitor and self.temp_monitor.available:
                temp_info = self.temp_monitor.get_info()
                if temp_info['should_throttle']:
                    # Reduce movement speed during thermal throttling
                    pan_diff = (pan - self.current_pan) * 0.5
                    tilt_diff = (tilt - self.current_tilt) * 0.5
                    pan = self.current_pan + pan_diff
                    tilt = self.current_tilt + tilt_diff
            
            # Clamp angles
            pan = max(0, min(180, pan))
            tilt = max(0, min(180, tilt))
            
            # Move servos only if change is significant
            if abs(pan - self.current_pan) > 0.5:
                self.pan_servo.angle = pan
                self.current_pan = pan
                
            if abs(tilt - self.current_tilt) > 0.5:
                self.tilt_servo.angle = tilt
                self.current_tilt = tilt
                
        except Exception as e:
            print(f"❌ Servo move error: {e}")
    
    def get_state(self):
        return {'pan': self.current_pan, 'tilt': self.current_tilt, 'available': self.available}
    
    def shutdown(self):
        """Safely shutdown servos"""
        try:
            if self.available:
                # Center servos before shutdown
                self.pan_servo.angle = self.tilt_servo.angle = 90.0
                time.sleep(0.5)
                # Disable PWM
                self.pca.deinit()
                print("✅ Servos safely shutdown")
        except Exception as e:
            print(f"Servo shutdown error: {e}")

# ==================== DISTANCE CALCULATOR ====================
class DistanceCalculator:
    def __init__(self, calibration):
        self.calibration = calibration
        self.history = deque(maxlen=5)
    
    def calculate_distance(self, bbox, frame_w, frame_h, confidence):
        try:
            # Use calibration focal lengths if available, otherwise defaults
            if self.calibration.complete:
                focal_x, focal_y = self.calibration.focal_x, self.calibration.focal_y
            else:
                focal_x = frame_w / (2 * math.tan(math.radians(CAMERA_FOV_H / 2)))
                focal_y = frame_h / (2 * math.tan(math.radians(CAMERA_FOV_V / 2)))
            
            bbox_w = bbox.width() * frame_w
            bbox_h = bbox.height() * frame_h
            
            # Validate bbox dimensions
            if bbox_h <= 0 or bbox_w <= 0:
                return 0.0
            
            # Calculate distance from both dimensions
            dist_h = (PERSON_HEIGHT * focal_y) / bbox_h
            dist_w = (PERSON_WIDTH * focal_x) / bbox_w
            
            # Weighted average (height more reliable)
            distance = dist_h * 0.75 + dist_w * 0.25
            
            # Clamp to reasonable range
            distance = max(0.5, min(50.0, distance))
            
            # Smooth with history
            self.history.append(distance * confidence)
            weights = [0.1, 0.2, 0.3, 0.4, 1.0][-len(self.history):]
            smoothed = sum(d * w for d, w in zip(self.history, weights)) / sum(weights)
            
            return smoothed
            
        except Exception as e:
            print(f"Distance calculation error: {e}")
            return 0.0
    
    def calculate_3d_position(self, distance, pan, tilt):
        try:
            pan_rad = math.radians(pan - 90)
            tilt_rad = math.radians(90 - (tilt + CAMERA_TILT_OFFSET))
            
            horizontal_dist = distance * math.cos(tilt_rad)
            x = horizontal_dist * math.sin(pan_rad)
            y = horizontal_dist * math.cos(pan_rad)  
            z = distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT
            
            return x, y, z
            
        except Exception as e:
            print(f"3D position calculation error: {e}")
            return 0.0, 0.0, 0.0

# ==================== TRACKER ====================
class PersonTracker:
    def __init__(self, servo, logger, calibration, gps_handler, temp_monitor):
        self.servo = servo
        self.logger = logger
        self.calibration = calibration
        self.gps_handler = gps_handler
        self.temp_monitor = temp_monitor
        self.distance_calc = DistanceCalculator(calibration)
        
        self.last_detection = time.time()
        self.target_lost_frames = 0
        self.lock_on_target = False
        
        # Tracking parameters
        self.frame_center_x = self.frame_center_y = 240
        self.frame_w = self.frame_h = 640
        
        print("🎯 Person tracker initialized")
    
    def update_frame_size(self, width, height):
        if width != self.frame_w or height != self.frame_h:
            self.frame_w, self.frame_h = width, height
            self.frame_center_x, self.frame_center_y = width // 2, height // 2
    
    def track_person(self, bbox, confidence, frame_count):
        try:
            # Add calibration measurement if still calibrating
            if self.calibration.is_calibrating:
                self.calibration.add_measurement(bbox, self.frame_w, self.frame_h, confidence)
            
            # Calculate distance and 3D position
            distance = self.distance_calc.calculate_distance(bbox, self.frame_w, self.frame_h, confidence)
            servo_state = self.servo.get_state()
            x, y, z = self.distance_calc.calculate_3d_position(distance, servo_state['pan'], servo_state['tilt'])
            
            # Calculate tracking error
            center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_w
            center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_h
            error_x = center_x - self.frame_center_x
            error_y = center_y - self.frame_center_y
            
            # Create GPS waypoint for Mission Planner
            waypoint = None
            if (self.gps_handler and self.gps_handler.healthy and 
                distance >= MIN_DISTANCE_FOR_GPS and confidence > 0.7):
                
                waypoint = self.gps_handler.create_waypoint(x, y, confidence)
                if waypoint:
                    self.logger.log_event('waypoint_created', 
                                        f'Mission Planner waypoint at {distance:.1f}m, confidence {confidence:.2f}')
            
            # Move servos if error exceeds dead zone
            dynamic_dead_zone = DEAD_ZONE * min(2.0, max(0.5, distance / 5.0))
            
            if abs(error_x) > dynamic_dead_zone or abs(error_y) > dynamic_dead_zone:
                # Calculate movement with thermal consideration
                thermal_factor = 0.7 if (self.temp_monitor and 
                                       self.temp_monitor.get_info()['should_throttle']) else 1.0
                
                pan_adj = -error_x * (PAN_SENSITIVITY / self.frame_w) * thermal_factor
                tilt_adj = error_y * (TILT_SENSITIVITY / self.frame_h) * thermal_factor
                
                # Apply smoothing and limits
                pan_adj *= SMOOTHING_FACTOR * confidence
                tilt_adj *= SMOOTHING_FACTOR * confidence
                pan_adj = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, pan_adj))
                tilt_adj = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, tilt_adj))
                
                new_pan = servo_state['pan'] + pan_adj
                new_tilt = servo_state['tilt'] + tilt_adj
                self.servo.move_to(new_pan, new_tilt)
                
                # Update lock status
                error_mag = math.sqrt(error_x**2 + error_y**2)
                tracking_quality = 1.0 - (error_mag / (math.sqrt(self.frame_w**2 + self.frame_h**2) / 2))
                
                if not self.lock_on_target and tracking_quality > 0.7:
                    self.lock_on_target = True
                    self.logger.log_event('target_lock', f'Target locked at {distance:.2f}m')
                    print(f"🎯 Target locked: {distance:.2f}m")
            
            # Update tracking state
            self.last_detection = time.time()
            self.target_lost_frames = 0
            
            # Log frame data with waypoint info
            self.logger.log_frame(frame_count, servo_state['pan'], servo_state['tilt'],
                                 confidence, distance, x, y, z, waypoint)
                                 
        except Exception as e:
            print(f"❌ Tracking error: {e}")
    
    def handle_lost_target(self, frame_count):
        try:
            self.target_lost_frames += 1
            
            if self.target_lost_frames > 30 and self.lock_on_target:
                self.lock_on_target = False
                self.logger.log_event('target_lost', f'Target lost after {self.target_lost_frames} frames')
                print(f"🔍 Target lost - scanning mode")
            
            # Log lost target frame
            servo_state = self.servo.get_state()
            self.logger.log_frame(frame_count, servo_state['pan'], servo_state['tilt'],
                                 0.0, 0.0, 0.0, 0.0, 0.0)
                                 
        except Exception as e:
            print(f"Lost target handling error: {e}")
    
    def is_tracking_active(self):
        return (time.time() - self.last_detection) < DETECTION_TIMEOUT

# ==================== CALLBACK SYSTEM ====================
class AppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        self.last_fps_report = time.time()
    
    def new_function(self):
        return "Mission Planner Servo Tracking: "

def main_callback(pad, info, user_data):
    """Main processing callback - handles detection and tracking"""
    try:
        buffer = info.get_buffer()
        if not buffer: 
            return Gst.PadProbeReturn.OK
        
        user_data.increment()
        frame_count = user_data.get_count()
        
        # Update frame properties periodically
        if frame_count % 30 == 0:
            try:
                format, width, height = get_caps_from_pad(pad)
                if width and height:
                    tracker.update_frame_size(width, height)
            except Exception as e:
                print(f"Frame size update error: {e}")
        
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
                        
                        # Validate bbox
                        if bbox.width() <= 0 or bbox.height() <= 0:
                            continue
                            
                        area = bbox.width() * bbox.height()
                        
                        # Score based on confidence, size, and aspect ratio
                        aspect = bbox.width() / bbox.height() if bbox.height() > 0 else 0
                        size_score = min(1.0, area * 10)
                        aspect_score = max(0.1, 1.0 - abs(aspect - 0.4) / 0.4)
                        total_score = confidence * size_score * aspect_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_person = {'bbox': bbox, 'confidence': confidence}
                            
            except Exception as e:
                print(f"Detection processing error: {e}")
                continue
        
        # Process detection or handle lost target
        if best_person:
            tracker.track_person(best_person['bbox'], best_person['confidence'], frame_count)
        else:
            tracker.handle_lost_target(frame_count)
        
        # Enhanced frame visualization
        if user_data.use_frame:
            try:
                format, width, height = get_caps_from_pad(pad)
                if format and width and height:
                    frame = get_numpy_from_buffer(buffer, format, width, height)
                    frame = draw_overlay(frame, best_person, tracker)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    user_data.set_frame(frame)
            except Exception as e:
                print(f"Frame visualization error: {e}")
        
        # Performance reporting every 3 seconds
        if frame_count % 90 == 0:
            current_time = time.time()
            if current_time - user_data.last_fps_report > 3:
                try:
                    distance_info = tracker.distance_calc.history
                    if distance_info:
                        avg_dist = sum(distance_info) / len(distance_info)
                        status = "LOCKED" if tracker.lock_on_target else "TRACKING"
                        temp_info = ""
                        if temp_monitor and temp_monitor.available:
                            temp = temp_monitor.get_info()
                            temp_info = f" | CPU: {temp['temperature']:.1f}°C"
                        print(f"🎯 {status}: {avg_dist:.2f}m{temp_info}")
                    user_data.last_fps_report = current_time
                except Exception as e:
                    print(f"Performance reporting error: {e}")
        
    except Exception as e:
        print(f"❌ Main callback error: {e}")
    
    return Gst.PadProbeReturn.OK

def draw_overlay(frame, best_person, tracker):
    """Draw comprehensive overlay with system status"""
    try:
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw crosshairs
        cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (0,255,255), 2)
        cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (0,255,255), 2)
        
        # Calibration status
        if calibration.is_calibrating:
            progress = calibration.get_progress()
            cv2.rectangle(frame, (5, 5), (width-5, 60), (0,0,0), -1)
            cv2.rectangle(frame, (5, 5), (width-5, 60), (0,255,255), 2)
            cv2.putText(frame, f"CALIBRATING: {progress:.0f}%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # Progress bar
            bar_w = width - 20
            cv2.rectangle(frame, (10, 35), (10 + bar_w, 50), (100,100,100), -1)
            cv2.rectangle(frame, (10, 35), (10 + int(bar_w * progress/100), 50), (0,255,0), -1)
        
        # Temperature display (top right)
        if temp_monitor and temp_monitor.available:
            temp_info = temp_monitor.get_info()
            temp_color = (0,255,0) if temp_info['status'] == 'NORMAL' else \
                        (0,255,255) if temp_info['status'] == 'HIGH' else \
                        (0,165,255) if temp_info['status'] == 'WARNING' else (0,0,255)
            
            temp_text = f"CPU: {temp_info['temperature']:.1f}C"
            text_size = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            bg_x1 = width - text_size[0] - 10
            
            cv2.rectangle(frame, (bg_x1, 5), (width-5, 30), (0,0,0), -1)
            cv2.rectangle(frame, (bg_x1, 5), (width-5, 30), temp_color, 2)
            cv2.putText(frame, temp_text, (bg_x1+5, 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, temp_color, 2)
        
        # Detection and tracking info
        if best_person and not calibration.is_calibrating:
            # Distance from history
            distance = 0.0
            if tracker.distance_calc.history:
                distance = tracker.distance_calc.history[-1]
            
            # Status text
            status_color = (0,255,0) if tracker.lock_on_target else (255,255,0)
            status_text = f"{'LOCKED' if tracker.lock_on_target else 'TRACKING'}: {distance:.1f}m"
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Servo angles
            servo_state = tracker.servo.get_state()
            cv2.putText(frame, f"Pan: {servo_state['pan']:.1f} Tilt: {servo_state['tilt']:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            # Bounding box
            bbox = best_person['bbox']
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int((bbox.xmin() + bbox.width()) * width)
            y2 = int((bbox.ymin() + bbox.height()) * height)
            
            box_color = (0,255,0) if tracker.lock_on_target else (255,255,0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Distance label
            cv2.rectangle(frame, (x1, y1-25), (x1+60, y1), box_color, -1)
            cv2.putText(frame, f"{distance:.1f}m", (x1+5, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Mission Planner GPS status (FIXED to use new status)
        if gps_handler:
            status = gps_handler.get_status()
            if status['connected'] and status['position_received']:
                gps_color = (0,255,0) if status['gps_fix'] >= 3 else (0,165,255)
                gps_text = f"GPS: {status['satellites']} sats"
                wp_text = f"WP: {status['waypoints_created']}"
                mission_text = f"MP: {status['mission_waypoints']}"
                
                cv2.putText(frame, gps_text, (width-200, height-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 1)
                cv2.putText(frame, wp_text, (width-200, height-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                cv2.putText(frame, mission_text, (width-200, height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            else:
                gps_status_text = "GPS: NO POS" if status['connected'] else f"GPS: CONN {status['connection_attempts']}"
                cv2.putText(frame, gps_status_text, (width-200, height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
    except Exception as e:
        print(f"⚠️ Overlay drawing error: {e}")
    
    return frame

# ==================== MAIN SYSTEM ====================
def initialize_system():
    """Initialize all system components with error handling"""
    print("🚀 Initializing FIXED Mission Planner Servo Tracking System...")
    
    # Initialize components with error handling
    temp_monitor = TemperatureMonitor()
    calibration = AutoCalibration()
    
    # GPS handler initialization (FIXED)
    gps_handler = GPSWaypointHandler()
    
    # Data logger
    logger = DataLogger(gps_handler, temp_monitor)
    
    # Servo controller
    servo_controller = ServoController(temp_monitor)
    
    # Person tracker
    tracker = PersonTracker(servo_controller, logger, calibration, gps_handler, temp_monitor)
    
    print("✅ System initialization complete")
    return temp_monitor, calibration, gps_handler, logger, servo_controller, tracker

def print_system_status():
    """Print enhanced system status"""
    print("\n" + "="*60)
    print("📊 FIXED MISSION PLANNER SERVO TRACKING SYSTEM STATUS")
    print("="*60)
    
    # Temperature
    if temp_monitor and temp_monitor.available:
        temp_info = temp_monitor.get_info()
        print(f"🌡️  Temperature: {temp_info['temperature']:.1f}°C ({temp_info['status']})")
    else:
        print(f"🌡️  Temperature: UNAVAILABLE")
    
    # Calibration
    if calibration.is_calibrating:
        print(f"📸 Calibration: IN PROGRESS ({calibration.get_progress():.0f}%)")
    else:
        print(f"📸 Calibration: COMPLETE")
    
    # GPS (FIXED status reporting)
    if gps_handler:
        status = gps_handler.get_status()
        if status['connected']:
            pos_status = "POSITION OK" if status['position_received'] else "NO POSITION"
            fix_status = "3D FIX" if status['gps_fix'] >= 3 else "NO FIX"
            print(f"🛰️ GPS: {pos_status}, {fix_status} ({status['satellites']} sats)")
            print(f"📱 Mission Planner: {status['mission_waypoints']} waypoints ready")
        else:
            print(f"🛰️ GPS: CONNECTING... (attempt {status['connection_attempts']})")
    else:
        print(f"🛰️ GPS: UNAVAILABLE")
    
    # Servo
    if servo_controller:
        servo_status = "ACTIVE" if servo_controller.available else "UNAVAILABLE"
        print(f"🔧 Servos: {servo_status}")
    else:
        print(f"🔧 Servos: NOT INITIALIZED")
    
    print("="*60)

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Global variables for callback access
    temp_monitor = calibration = gps_handler = data_logger = servo_controller = tracker = None
    
    try:
        # Set up environment
        project_root = Path(__file__).resolve().parent.parent
        env_file = project_root / ".env"
        os.environ["HAILO_ENV_FILE"] = str(env_file)
        
        # Initialize system
        (temp_monitor, calibration, gps_handler, 
         data_logger, servo_controller, tracker) = initialize_system()
        
        print_system_status()
        
        if calibration.is_calibrating:
            print("\n⏳ Camera calibration in progress...")
            print("   Please ensure a person is visible 2-4 meters away")
        
        print("\n🚀 Starting FIXED Mission Planner tracking system...")
        print("📱 Mission Planner Integration Active (FIXED):")
        print("   • Waypoints will appear in Mission Planner Flight Plan tab")
        print("   • Click 'Read WPs' in Mission Planner to see waypoints")
        print("   • Waypoints created every 5 seconds when tracking targets >2m away")
        print("   • System creates complete flight missions for Mission Planner")
        print("   • FIXED: Immediate position getting and upload like File 2")
        print("   • FIXED: Proper mission clearing and upload protocol")
        print("Press Ctrl+C to stop\n")
        
        # Start GStreamer app
        user_data = AppCallback()
        app = GStreamerDetectionApp(main_callback, user_data)
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("📊 Finalizing shutdown...")
        
        # Shutdown all components safely
        try:
            if data_logger:
                data_logger.finalize()
        except Exception as e:
            print(f"Data logger shutdown error: {e}")
        
        try:
            if servo_controller:
                servo_controller.shutdown()
        except Exception as e:
            print(f"Servo shutdown error: {e}")
        
        try:
            if temp_monitor:
                temp_monitor.shutdown()
        except Exception as e:
            print(f"Temperature monitor shutdown error: {e}")
        
        try:
            if gps_handler:
                gps_handler.shutdown()
        except Exception as e:
            print(f"GPS handler shutdown error: {e}")
        
        # Final statistics
        if data_logger:
            try:
                stats = data_logger.session_data['statistics']
                print(f"\n📊 Final Stats:")
                print(f"   Detections: {stats['detections']}")
                print(f"   Waypoints: {stats['waypoints']}")
                print(f"   Max Distance: {stats['max_distance']:.1f}m")
                if stats['temp_readings'] > 0:
                    print(f"   Max Temperature: {stats['max_temp']:.1f}°C")
                
                # Show Mission Planner info
                if gps_handler:
                    gps_status = gps_handler.get_status()
                    print(f"   Mission Planner Waypoints: {gps_status['mission_waypoints']}")
                    
            except Exception as e:
                print(f"Final stats error: {e}")
        
        print("✅ Shutdown complete")
        print("📁 Logs saved to: servo_logs/")
        print("📱 Waypoints remain in Mission Planner - use 'Read WPs' to view")
        print("Thank you for using the FIXED Mission Planner Servo Tracking System! 🎯")