#!/usr/bin/env python3
"""
Enhanced Servo Tracking System with GPS Waypoint Generation
For Raspberry Pi 5 with CubeOrange and Hailo AI
Enhanced with: Live altitude, robust calibration, reduced lag, improved accuracy
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
# ENHANCED CONFIGURATION CONSTANTS
# ==============================================================================

# Servo tracking parameters (optimized for reduced lag)
DEAD_ZONE = 8  # Reduced for more responsive tracking
SMOOTHING_FACTOR = 0.45  # Increased for smoother movement
MAX_STEP_SIZE = 8  # Increased for faster response
MIN_CONFIDENCE = 0.25  # Slightly reduced for better detection
DETECTION_TIMEOUT = 1.5  # Reduced timeout
PAN_SENSITIVITY = 55  # Increased sensitivity
TILT_SENSITIVITY = 45  # Increased sensitivity
FRAME_SKIP_COUNT = 0  # Removed frame skipping for less lag
DETECTION_HISTORY_SIZE = 5  # Increased for better smoothing

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
GPS_UPDATE_INTERVAL = 0.5  # Faster updates
MIN_DISTANCE_FOR_GPS = 2.5  # Reduced minimum distance
MAX_GPS_POINTS = 150  # Increased capacity
WAYPOINT_ALTITUDE_OFFSET = 0.0
WAYPOINT_MODE = "ADD"
WAYPOINT_CLEAR_TIMEOUT = 300
MAX_WAYPOINTS = 25  # Increased waypoint capacity

# Enhanced calibration parameters
CALIBRATION_MODE = True
CALIBRATION_DISTANCE = 2.0
CALIBRATION_SAMPLES_REQUIRED = 20  # More samples for accuracy
CALIBRATION_TIMEOUT = 60.0  # Auto-complete calibration
AUTO_CALIBRATION_ENABLED = True
CALIBRATION_ACCURACY_THRESHOLD = 0.95

# Performance optimization
THREAD_POOL_SIZE = 4
PROCESSING_QUEUE_SIZE = 10
ALTITUDE_UPDATE_RATE = 5.0  # Hz

# ==============================================================================
# ENHANCED MAVLINK GPS HANDLER WITH ALTITUDE AND RELIABILITY
# ==============================================================================

class EnhancedMAVLinkHandler:
    def __init__(self, connection_string=MAVLINK_CONNECTION, baud=MAVLINK_BAUD):
        self.connection_string = connection_string
        self.baud = baud
        self.mavlink_connection = None
        
        # Enhanced position data
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0  # GPS altitude
        self.current_alt_rel = 0.0  # Relative altitude from home
        self.current_alt_terrain = 0.0  # Terrain altitude
        self.barometric_altitude = 0.0  # Barometric altitude
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
            
            # Request enhanced data streams
            self.request_enhanced_data_streams()
            
            # Start threads
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
                time.sleep(0.1)  # Small delay between requests
                
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
                
                # Enhanced GPS handling
                if msg_type == 'GPS_RAW_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    self.gps_fix_type = msg.fix_type
                    self.satellites_visible = msg.satellites_visible
                    self.gps_accuracy = getattr(msg, 'eph', 0) / 100.0  # Horizontal accuracy
                    self.last_gps_update = current_time
                    
                elif msg_type == 'GPS2_RAW':
                    # Secondary GPS data if available
                    pass
                    
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    self.current_alt_rel = msg.relative_alt / 1000.0
                    self.current_heading = msg.hdg / 100.0
                    
                # Enhanced altitude data
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
                
                # Enhanced attitude data
                elif msg_type == 'ATTITUDE':
                    yaw_rad = msg.yaw
                    self.current_heading = math.degrees(yaw_rad) % 360
                    
                # Enhanced home position
                elif msg_type == 'HOME_POSITION':
                    self.home_lat = msg.latitude / 1e7
                    self.home_lon = msg.longitude / 1e7
                    self.home_alt = msg.altitude / 1000.0
                    self.home_set = True
                    print(f"🏠 Home position set: {self.home_lat:.6f}, {self.home_lon:.6f}")
                    
                # Mission management
                elif msg_type == 'MISSION_CURRENT':
                    self.current_wp_seq = msg.seq
                    
                elif msg_type == 'MISSION_ITEM_REACHED':
                    print(f"✅ Reached waypoint {msg.seq}")
                    
                # System status
                elif msg_type == 'HEARTBEAT':
                    self.last_heartbeat = current_time
                    
                elif msg_type == 'GPS_STATUS':
                    # Additional GPS quality metrics
                    pass
                    
            except Exception as e:
                if self.running and self.connected:
                    print(f"MAVLink receive error: {e}")
                    self.check_connection_health()
    
    def _heartbeat_monitor(self):
        """Monitor connection health and attempt reconnection if needed"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if we've lost connection
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
    
    def get_live_altitude_data(self):
        """Get comprehensive live altitude data from CubeOrange"""
        return {
            'gps_altitude': self.current_alt,
            'relative_altitude': self.current_alt_rel,
            'terrain_altitude': self.current_alt_terrain,
            'barometric_altitude': self.barometric_altitude,
            'altitude_history': list(self.altitude_history)[-10:],  # Last 10 readings
            'altitude_accuracy': self.get_altitude_accuracy()
        }
    
    def get_altitude_accuracy(self):
        """Calculate altitude accuracy based on GPS quality"""
        if self.gps_fix_type >= 3 and self.satellites_visible >= 6:
            return "HIGH"
        elif self.gps_fix_type >= 2 and self.satellites_visible >= 4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_enhanced_gps_position(self, x_meters, y_meters, z_meters=0):
        """Enhanced GPS position calculation with altitude integration"""
        if self.gps_fix_type < 3:
            return None, None, None
        
        # Calculate horizontal position
        bearing_to_target = math.degrees(math.atan2(x_meters, y_meters))
        absolute_bearing = (self.current_heading + bearing_to_target) % 360
        horizontal_distance = math.sqrt(x_meters**2 + y_meters**2)
        
        # Enhanced geodetic calculations
        lat_rad = math.radians(self.current_lat)
        lon_rad = math.radians(self.current_lon)
        bearing_rad = math.radians(absolute_bearing)
        
        # More precise earth radius calculation
        a = 6378137.0  # WGS84 semi-major axis
        f = 1/298.257223563  # WGS84 flattening
        e2 = 2*f - f*f  # First eccentricity squared
        
        # Calculate new latitude
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(horizontal_distance / a) +
            math.cos(lat_rad) * math.sin(horizontal_distance / a) * math.cos(bearing_rad)
        )
        
        # Calculate new longitude
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(horizontal_distance / a) * math.cos(lat_rad),
            math.cos(horizontal_distance / a) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        # Calculate target altitude using multiple sources
        target_altitude = self.calculate_target_altitude(z_meters)
        
        return math.degrees(new_lat_rad), math.degrees(new_lon_rad), target_altitude
    
    def calculate_target_altitude(self, z_meters):
        """Calculate target altitude using best available altitude data"""
        # Use relative altitude as primary source
        if self.current_alt_rel > 0:
            base_altitude = self.current_alt_rel
        elif self.current_alt > 0:
            base_altitude = self.current_alt
        else:
            base_altitude = self.barometric_altitude
        
        return base_altitude + z_meters + WAYPOINT_ALTITUDE_OFFSET
    
    def add_enhanced_detection_point(self, x_meters, y_meters, z_meters, confidence):
        """Enhanced detection point with improved accuracy"""
        current_time = time.time()
        
        # Dynamic update interval based on movement
        dynamic_interval = GPS_UPDATE_INTERVAL
        if len(self.gps_points) > 0:
            last_point = self.gps_points[-1]
            movement_speed = math.sqrt(x_meters**2 + y_meters**2) / (current_time - last_point['timestamp'])
            if movement_speed > 2.0:  # Fast movement
                dynamic_interval *= 0.5
        
        if current_time - self.last_point_time < dynamic_interval:
            return None
        
        distance = math.sqrt(x_meters**2 + y_meters**2)
        if distance < MIN_DISTANCE_FOR_GPS:
            return None
        
        lat, lon, alt = self.calculate_enhanced_gps_position(x_meters, y_meters, z_meters)
        if lat is None or lon is None:
            return None
        
        # Enhanced GPS point with more data
        gps_point = {
            'timestamp': current_time,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'relative_x': x_meters,
            'relative_y': y_meters,
            'relative_z': z_meters,
            'distance': distance,
            'confidence': confidence,
            'vehicle_lat': self.current_lat,
            'vehicle_lon': self.current_lon,
            'vehicle_alt': self.current_alt,
            'vehicle_alt_rel': self.current_alt_rel,
            'vehicle_heading': self.current_heading,
            'gps_accuracy': self.gps_accuracy,
            'satellites': self.satellites_visible,
            'fix_type': self.gps_fix_type
        }
        
        self.gps_points.append(gps_point)
        self.last_point_time = current_time
        
        # Enhanced waypoint upload with better error handling
        success = self.upload_enhanced_waypoint(lat, lon, alt, confidence)
        
        if success:
            self.last_waypoint_time = current_time
            bearing = math.degrees(math.atan2(x_meters, y_meters))
            bearing = (self.current_heading + bearing) % 360
            print(f"🎯 Enhanced waypoint: {distance:.1f}m @ {bearing:.0f}° (Alt: {alt:.1f}m)")
        
        return gps_point
    
    def upload_enhanced_waypoint(self, lat, lon, alt, confidence):
        """Enhanced waypoint upload with improved reliability"""
        try:
            # Confidence-based waypoint management
            if confidence < 0.4 and self.waypoint_mode == "ADD":
                return False  # Skip low-confidence waypoints
            
            wp_seq = self.get_current_mission_count()
            
            # Enhanced waypoint type selection based on confidence
            if confidence > 0.8:
                waypoint_type = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
            else:
                waypoint_type = mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM
            
            return self.upload_waypoint_with_retry(lat, lon, alt, wp_seq, waypoint_type)
            
        except Exception as e:
            print(f"Enhanced waypoint upload error: {e}")
            return False
    
    def upload_waypoint_with_retry(self, lat, lon, alt, seq, waypoint_type, max_retries=3):
        """Upload waypoint with retry mechanism"""
        for attempt in range(max_retries):
            try:
                success = self.upload_waypoint(lat, lon, alt, seq, waypoint_type)
                if success:
                    return True
                
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief delay before retry
                    
            except Exception as e:
                print(f"Waypoint upload attempt {attempt + 1} failed: {e}")
        
        return False
    
    def upload_waypoint(self, lat, lon, alt, seq, waypoint_type=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT):
        """Enhanced waypoint upload with better protocol handling"""
        try:
            # Clear any pending messages
            while self.mavlink_connection.recv_match(blocking=False):
                pass
            
            # Send mission count
            self.mavlink_connection.mav.mission_count_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                seq + 1, 0
            )
            
            start_time = time.time()
            items_sent = set()
            
            while time.time() - start_time < 8:  # Increased timeout
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.5)
                
                if msg:
                    msg_type = msg.get_type()
                    
                    if msg_type in ['MISSION_REQUEST', 'MISSION_REQUEST_INT']:
                        requested_seq = msg.seq
                        
                        if requested_seq not in items_sent:
                            if requested_seq == 0:
                                self.send_enhanced_home_position()
                            elif requested_seq == seq:
                                # Enhanced waypoint with better parameters
                                self.mavlink_connection.mav.mission_item_int_send(
                                    self.mavlink_connection.target_system,
                                    self.mavlink_connection.target_component,
                                    seq,
                                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                    waypoint_type,
                                    0, 1,  # current, autocontinue
                                    0,  # param1 (hold time)
                                    3,  # param2 (acceptance radius) 
                                    0,  # param3
                                    float('nan'),  # param4 (yaw angle)
                                    int(lat * 1e7),
                                    int(lon * 1e7),
                                    alt, 0
                                )
                            items_sent.add(requested_seq)
                            
                    elif msg_type == 'MISSION_ACK':
                        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                            print("✅ Enhanced waypoint uploaded successfully!")
                            return True
                        else:
                            print(f"❌ Mission rejected: {msg.type}")
                            return False
            
            print("❌ Timeout waiting for enhanced mission protocol")
            return False
            
        except Exception as e:
            print(f"Error uploading enhanced waypoint: {e}")
            return False
    
    def send_enhanced_home_position(self):
        """Send enhanced home position"""
        if self.home_set and self.home_lat and self.home_lon:
            lat, lon, alt = self.home_lat, self.home_lon, self.home_alt
        else:
            lat, lon, alt = self.current_lat, self.current_lon, self.current_alt
        
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
    
    def get_current_mission_count(self):
        """Enhanced mission count with better error handling"""
        try:
            # Clear pending messages
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
            else:
                return max(1, self.mission_count)  # Use last known count
                
        except Exception as e:
            print(f"Error getting mission count: {e}")
            return max(1, self.mission_count)
    
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
            'altitude_data': self.get_live_altitude_data()
        }
    
    def clear_mission(self):
        """Enhanced mission clearing"""
        try:
            self.mavlink_connection.mav.mission_clear_all_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component
            )
            
            msg = self.mavlink_connection.recv_match(type='MISSION_ACK', blocking=True, timeout=3)
            if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("🗑️ Enhanced mission cleared")
                self.mission_count = 0
                return True
                
        except Exception as e:
            print(f"Error clearing enhanced mission: {e}")
        return False
    
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
# ENHANCED SERVO CONTROLLER WITH REDUCED LAG
# ==============================================================================

class UltraFastServoController:
    def __init__(self, logger=None):
        self.logger = logger
        
        # Enhanced I2C setup with higher frequency
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)  # 1MHz
        self.pca = adafruit_pca9685.PCA9685(self.i2c)
        self.pca.frequency = 60  # Increased frequency for faster response
        
        self.pan_servo = servo.Servo(self.pca.channels[0], min_pulse=500, max_pulse=2500)
        self.tilt_servo = servo.Servo(self.pca.channels[1], min_pulse=500, max_pulse=2500)
        
        # Enhanced position tracking
        self.current_pan = 90.0
        self.current_tilt = 90.0 - CAMERA_TILT_OFFSET
        self.target_pan = self.current_pan
        self.target_tilt = self.current_tilt
        
        # Enhanced velocity tracking
        self.velocity_pan = 0.0
        self.velocity_tilt = 0.0
        self.acceleration_pan = 0.0
        self.acceleration_tilt = 0.0
        self.last_velocity_pan = 0.0
        self.last_velocity_tilt = 0.0
        self.last_update_time = time.time()
        
        # Position smoothing
        self.position_filter_pan = deque(maxlen=3)
        self.position_filter_tilt = deque(maxlen=3)
        
        # Enhanced command processing
        self.command_queue = Queue(maxsize=3)  # Smaller queue for less lag
        self.running = True
        self.servo_thread = threading.Thread(target=self._ultra_fast_servo_worker, daemon=True)
        
        # Initialize servos to center position
        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        time.sleep(0.1)  # Allow servos to reach position
        
        self.servo_thread.start()
        
        if self.logger:
            self.logger.log_event('servo_init', 'Ultra-fast servo controller initialized')
    
    def _ultra_fast_servo_worker(self):
        """Ultra-fast servo worker with reduced latency"""
        last_movement_time = time.time()
        
        while self.running:
            try:
                # Process commands with minimal delay
                command = None
                try:
                    command = self.command_queue.get(timeout=0.01)  # Very short timeout
                except Empty:
                    # Continue with smooth interpolation even without new commands
                    current_time = time.time()
                    dt = current_time - self.last_update_time
                    
                    if dt > 0.002:  # 500Hz maximum update rate
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
                
                # Calculate velocities and accelerations
                if dt > 0:
                    new_velocity_pan = (target_pan - self.current_pan) / dt
                    new_velocity_tilt = (target_tilt - self.current_tilt) / dt
                    
                    self.acceleration_pan = (new_velocity_pan - self.last_velocity_pan) / dt
                    self.acceleration_tilt = (new_velocity_tilt - self.last_velocity_tilt) / dt
                    
                    self.velocity_pan = new_velocity_pan
                    self.velocity_tilt = new_velocity_tilt
                    
                    self.last_velocity_pan = new_velocity_pan
                    self.last_velocity_tilt = new_velocity_tilt
                
                # Ultra-fast movement with predictive positioning
                self._execute_ultra_fast_movement(target_pan, target_tilt, dt)
                
                self.last_update_time = current_time
                last_movement_time = current_time
                
            except Exception as e:
                print(f"Ultra-fast servo thread error: {e}")
                time.sleep(0.001)
    
    def _smooth_interpolate(self, dt):
        """Smooth interpolation between positions for fluid movement"""
        if abs(self.target_pan - self.current_pan) > 0.1 or abs(self.target_tilt - self.current_tilt) > 0.1:
            # Exponential smoothing with dynamic factor
            smoothing_factor = min(0.8, SMOOTHING_FACTOR * (1 + dt * 10))
            
            new_pan = self.current_pan + (self.target_pan - self.current_pan) * smoothing_factor
            new_tilt = self.current_tilt + (self.target_tilt - self.current_tilt) * smoothing_factor
            
            self._execute_ultra_fast_movement(new_pan, new_tilt, dt)
    
    def _execute_ultra_fast_movement(self, pan_angle, tilt_angle, dt):
        """Execute movement with minimal latency"""
        # Apply position filtering for smoothness
        self.position_filter_pan.append(pan_angle)
        self.position_filter_tilt.append(tilt_angle)
        
        # Use median filtering for smooth movement
        if len(self.position_filter_pan) >= 2:
            filtered_pan = statistics.median(self.position_filter_pan)
            filtered_tilt = statistics.median(self.position_filter_tilt)
        else:
            filtered_pan, filtered_tilt = pan_angle, tilt_angle
        
        # Check if movement is significant enough
        pan_diff = abs(filtered_pan - self.current_pan)
        tilt_diff = abs(filtered_tilt - self.current_tilt)
        
        if pan_diff > 0.05 or tilt_diff > 0.05:
            try:
                # Simultaneous servo updates for minimal lag
                if pan_diff > 0.05:
                    self.pan_servo.angle = filtered_pan
                    self.current_pan = filtered_pan
                
                if tilt_diff > 0.05:
                    self.tilt_servo.angle = filtered_tilt
                    self.current_tilt = filtered_tilt
                
                # Minimal delay for servo response
                time.sleep(0.001)  # 1ms delay
                
            except Exception as e:
                print(f"Servo movement error: {e}")
    
    def move_to_ultra_fast(self, pan_angle, tilt_angle):
        """Ultra-fast move command with lag reduction"""
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
            'velocity_pan': self.velocity_pan,
            'velocity_tilt': self.velocity_tilt,
            'acceleration_pan': self.acceleration_pan,
            'acceleration_tilt': self.acceleration_tilt,
            'queue_size': self.command_queue.qsize()
        }
    
    def shutdown(self):
        """Enhanced shutdown procedure"""
        if self.logger:
            self.logger.log_event('servo_shutdown', 'Ultra-fast servo controller shutting down')
        
        self.running = False
        
        # Send stop command
        try:
            self.command_queue.put_nowait(None)
        except:
            pass
        
        # Wait for thread to finish
        if self.servo_thread.is_alive():
            self.servo_thread.join(timeout=2.0)
        
        # Return to center position
        try:
            self.pan_servo.angle = 90
            self.tilt_servo.angle = 90
            time.sleep(0.5)
        except:
            pass

# ==============================================================================
# ROBUST AUTOMATIC CALIBRATION SYSTEM
# ==============================================================================

class RobustAutoCalibration:
    def __init__(self, logger=None):
        self.logger = logger
        self.calibration_active = AUTO_CALIBRATION_ENABLED
        self.calibration_complete = False
        
        # Calibration data collection
        self.measurements = []
        self.distance_measurements = []
        self.angle_measurements = []
        self.confidence_measurements = []
        
        # Calibration parameters
        self.start_time = time.time()
        self.samples_collected = 0
        self.required_samples = CALIBRATION_SAMPLES_REQUIRED
        self.calibration_timeout = CALIBRATION_TIMEOUT
        
        # Statistical analysis
        self.focal_length_estimates = []
        self.height_pixel_ratios = []
        
        # Calibration phases
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
        
        # Check timeout
        if current_time - self.start_time > self.calibration_timeout:
            self._force_complete_calibration()
            return
        
        # Update phase
        if self.current_phase == "INITIALIZATION":
            self._advance_phase("DATA_COLLECTION")
        
        # Quality checks for measurement
        person_height_pixels = bbox.height() * frame_height
        
        # Reject measurements that are too small or large
        if person_height_pixels < 30 or person_height_pixels > frame_height * 0.8:
            return
        
        # Reject low confidence measurements
        if confidence < 0.4:
            return
        
        # Reject measurements at very close or far distances
        if distance < 1.0 or distance > 15.0:
            return
        
        # Store measurement
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
        
        # Calculate focal length estimate for this measurement
        focal_estimate = (person_height_pixels * distance) / AVERAGE_PERSON_HEIGHT
        self.focal_length_estimates.append(focal_estimate)
        
        # Progress update
        if self.samples_collected % 5 == 0:
            progress = (self.samples_collected / self.required_samples) * 100
            print(f"📊 Calibration progress: {self.samples_collected}/{self.required_samples} ({progress:.1f}%)")
        
        # Check if we have enough samples
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
        
        # Statistical analysis of focal length estimates
        focal_estimates = np.array(self.focal_length_estimates)
        
        # Remove outliers using IQR method
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
        
        # Calculate statistics
        final_focal_length = np.median(filtered_estimates)
        focal_std = np.std(filtered_estimates)
        focal_confidence = 1.0 - (focal_std / final_focal_length)
        
        print(f"\n🎯 AUTOMATIC CALIBRATION ANALYSIS:")
        print(f"   Samples analyzed: {len(filtered_estimates)}")
        print(f"   Focal length: {final_focal_length:.1f} pixels")
        print(f"   Standard deviation: {focal_std:.2f}")
        print(f"   Confidence: {focal_confidence:.3f}")
        
        # Validation phase
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
        
        # Update global focal length (in a real implementation, you'd update the DistanceCalculator)
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
# ENHANCED DISTANCE CALCULATOR WITH IMPROVED ACCURACY
# ==============================================================================

class EnhancedDistanceCalculator:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Enhanced focal length calculation
        self.focal_length_x = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        
        # Advanced filtering
        self.distance_history = deque(maxlen=10)  # Increased history
        self.kalman_filter = self._initialize_kalman_filter()
        
        # Multi-method distance estimation
        self.estimation_methods = ['height', 'width', 'area', 'aspect_ratio']
        self.method_weights = {'height': 0.4, 'width': 0.3, 'area': 0.2, 'aspect_ratio': 0.1}
        
        # Accuracy tracking
        self.accuracy_history = deque(maxlen=50)
        self.confidence_threshold = 0.7
        
    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for distance smoothing"""
        try:
            # Simple 1D Kalman filter for distance
            class SimpleKalmanFilter:
                def __init__(self):
                    self.x = 0.0  # state (distance)
                    self.P = 1000.0  # uncertainty
                    self.Q = 0.1  # process noise
                    self.R = 1.0  # measurement noise
                
                def update(self, measurement):
                    # Prediction step
                    self.P = self.P + self.Q
                    
                    # Update step
                    K = self.P / (self.P + self.R)  # Kalman gain
                    self.x = self.x + K * (measurement - self.x)
                    self.P = (1 - K) * self.P
                    
                    return self.x
            
            return SimpleKalmanFilter()
        except:
            return None
    
    def update_frame_size(self, width, height):
        """Update frame size and recalculate focal lengths"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.focal_length_x = width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
            self.focal_length_y = height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
    
    def calculate_enhanced_distance(self, bbox, confidence=1.0):
        """Calculate distance using multiple enhanced methods"""
        bbox_width_pixels = bbox.width() * self.frame_width
        bbox_height_pixels = bbox.height() * self.frame_height
        bbox_area_pixels = bbox_width_pixels * bbox_height_pixels
        
        distances = {}
        
        # Method 1: Height-based distance (most reliable for humans)
        if bbox_height_pixels > 10:
            distances['height'] = (AVERAGE_PERSON_HEIGHT * self.focal_length_y) / bbox_height_pixels
        
        # Method 2: Width-based distance (backup method)
        if bbox_width_pixels > 5:
            distances['width'] = (AVERAGE_PERSON_WIDTH * self.focal_length_x) / bbox_width_pixels
        
        # Method 3: Area-based distance (for very close/far subjects)
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
        
        # Weighted combination of methods
        final_distance = self._combine_distance_estimates(distances, confidence)
        
        # Apply Kalman filtering if available
        if self.kalman_filter:
            final_distance = self.kalman_filter.update(final_distance)
        
        # Add to history
        self.distance_history.append(final_distance)
        
        # Calculate confidence in measurement
        measurement_confidence = self._calculate_measurement_confidence(distances, bbox)
        self.accuracy_history.append(measurement_confidence)
        
        return self._get_smoothed_distance()
    
    def _combine_distance_estimates(self, distances, detection_confidence):
        """Combine multiple distance estimates with adaptive weighting"""
        if not distances:
            return 5.0  # Default distance
        
        # Adaptive weighting based on detection confidence
        adaptive_weights = self.method_weights.copy()
        
        # Increase height weight for high-confidence detections
        if detection_confidence > 0.8:
            adaptive_weights['height'] *= 1.2
            adaptive_weights['width'] *= 0.9
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for method, distance in distances.items():
            if method in adaptive_weights and 0.5 < distance < 50:  # Sanity check
                weight = adaptive_weights[method]
                weighted_sum += distance * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return distances.get('height', distances.get('width', 5.0))
    
    def _calculate_measurement_confidence(self, distances, bbox):
        """Calculate confidence in distance measurement"""
        if len(distances) < 2:
            return 0.5
        
        # Calculate variance between methods
        distance_values = list(distances.values())
        variance = np.var(distance_values)
        mean_distance = np.mean(distance_values)
        
        # Confidence decreases with variance
        confidence = 1.0 / (1.0 + variance / max(mean_distance, 1.0))
        
        # Bonus for reasonable bounding box size
        bbox_area = bbox.width() * bbox.height()
        if 0.001 < bbox_area < 0.3:  # Reasonable size
            confidence *= 1.1
        
        return min(1.0, confidence)
    
    def calculate_enhanced_3d_position(self, bbox, pan_angle, tilt_angle, distance, altitude_data=None):
        """Calculate enhanced 3D position with altitude integration"""
        # Basic 3D calculation
        pan_rad = math.radians(pan_angle - 90)
        actual_tilt_angle = tilt_angle + CAMERA_TILT_OFFSET
        tilt_rad = math.radians(90 - actual_tilt_angle)
        
        horizontal_distance = distance * math.cos(tilt_rad)
        
        x = horizontal_distance * math.sin(pan_rad)
        y = horizontal_distance * math.cos(pan_rad)
        z_relative = distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT
        
        # Enhanced Z calculation with altitude data
        if altitude_data:
            # Use terrain-relative altitude when available
            terrain_offset = altitude_data.get('terrain_altitude', 0)
            z_absolute = z_relative + terrain_offset
        else:
            z_absolute = z_relative
        
        return x, y, z_relative, z_absolute
    
    def calculate_angular_size(self, bbox):
        """Calculate angular size with enhanced precision"""
        angular_width = bbox.width() * CAMERA_FOV_HORIZONTAL
        angular_height = bbox.height() * CAMERA_FOV_VERTICAL
        
        # Calculate angular area
        angular_area = angular_width * angular_height
        
        return angular_width, angular_height, angular_area
    
    def _get_smoothed_distance(self):
        """Get smoothed distance using advanced filtering"""
        if not self.distance_history:
            return 0.0
        
        # Use different smoothing based on history length
        if len(self.distance_history) < 3:
            return self.distance_history[-1]
        
        # Remove outliers using modified Z-score
        distances = list(self.distance_history)
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (distances - median) / mad
            filtered_distances = [d for d, z in zip(distances, modified_z_scores) if abs(z) < 3.5]
        else:
            filtered_distances = distances
        
        if not filtered_distances:
            filtered_distances = distances
        
        # Exponential weighted moving average
        weights = np.exp(np.linspace(-1, 0, len(filtered_distances)))
        weights /= weights.sum()
        
        return np.average(filtered_distances, weights=weights)
    
    def get_measurement_quality(self):
        """Get overall measurement quality metrics"""
        if not self.accuracy_history:
            return {'quality': 'UNKNOWN', 'confidence': 0.0}
        
        avg_confidence = np.mean(self.accuracy_history)
        
        if avg_confidence > 0.8:
            quality = 'EXCELLENT'
        elif avg_confidence > 0.6:
            quality = 'GOOD'
        elif avg_confidence > 0.4:
            quality = 'FAIR'
        else:
            quality = 'POOR'
        
        return {'quality': quality, 'confidence': avg_confidence}

# ==============================================================================
# ULTRA-FAST TRACKER WITH LAG REDUCTION
# ==============================================================================

class UltraFastTracker:
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
        self.distance_calculator = EnhancedDistanceCalculator(self.frame_width, self.frame_height)
        
        # Enhanced tracking state
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        self.lock_on_target = False
        self.tracking_quality = "NONE"
        
        # Enhanced position tracking
        self.current_distance = 0.0
        self.current_3d_position = (0.0, 0.0, 0.0, 0.0)  # x, y, z_rel, z_abs
        self.last_3d_position = (0.0, 0.0, 0.0, 0.0)
        
        # Predictive tracking
        self.velocity_3d = (0.0, 0.0, 0.0)
        self.prediction_history = deque(maxlen=5)
        
        # Enhanced smoothing with multiple filters
        self.pan_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.tilt_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.position_predictor = self._initialize_position_predictor()
        
        # Adaptive parameters
        self.dynamic_sensitivity = {'pan': PAN_SENSITIVITY, 'tilt': TILT_SENSITIVITY}
        self.dynamic_dead_zone = DEAD_ZONE
        
        # Performance metrics
        self.frames_processed = 0
        self.successful_tracks = 0
        self.average_confidence = 0.0
        
        print("🎯 Ultra-Fast Tracker with lag reduction initialized")
    
    def _initialize_position_predictor(self):
        """Initialize position prediction system"""
        class PositionPredictor:
            def __init__(self):
                self.last_positions = deque(maxlen=5)
                self.last_times = deque(maxlen=5)
            
            def add_position(self, x, y, timestamp):
                self.last_positions.append((x, y))
                self.last_times.append(timestamp)
            
            def predict_next_position(self, dt=0.033):  # Assume 30fps
                if len(self.last_positions) < 2:
                    return None
                
                # Simple linear prediction
                pos1 = self.last_positions[-2]
                pos2 = self.last_positions[-1]
                t1 = self.last_times[-2]
                t2 = self.last_times[-1]
                
                if t2 - t1 > 0:
                    vx = (pos2[0] - pos1[0]) / (t2 - t1)
                    vy = (pos2[1] - pos1[1]) / (t2 - t1)
                    
                    pred_x = pos2[0] + vx * dt
                    pred_y = pos2[1] + vy * dt
                    
                    return pred_x, pred_y
                
                return pos2
        
        return PositionPredictor()
    
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
            
            if self.logger:
                self.logger.log_event('resolution_change', 
                                    f'Frame: {width}x{height}, Factor: {resolution_factor:.2f}')
    
    def track_person_ultra_fast(self, bbox, confidence, frame_count):
        """Ultra-fast person tracking with minimal lag"""
        current_time = time.time()
        self.frames_processed += 1
        
        # Enhanced distance calculation
        self.current_distance = self.distance_calculator.calculate_enhanced_distance(bbox, confidence)
        
        # Get current servo state
        servo_state = self.servo.get_enhanced_state()
        current_pan = servo_state['current_pan']
        current_tilt = servo_state['current_tilt']
        
        # Get altitude data from MAVLink
        altitude_data = None
        if self.mavlink_handler:
            status = self.mavlink_handler.get_enhanced_status()
            altitude_data = status.get('altitude_data', {})
        
        # Enhanced 3D position calculation
        x, y, z_rel, z_abs = self.distance_calculator.calculate_enhanced_3d_position(
            bbox, current_pan, current_tilt, self.current_distance, altitude_data
        )
        self.current_3d_position = (x, y, z_rel, z_abs)
        
        # Calculate 3D velocity for prediction
        if self.last_3d_position != (0.0, 0.0, 0.0, 0.0):
            dt = current_time - self.last_detection_time
            if dt > 0:
                self.velocity_3d = (
                    (x - self.last_3d_position[0]) / dt,
                    (y - self.last_3d_position[1]) / dt,
                    (z_rel - self.last_3d_position[2]) / dt
                )
        
        self.last_3d_position = self.current_3d_position
        
        # Enhanced angular calculations
        angular_width, angular_height, angular_area = self.distance_calculator.calculate_angular_size(bbox)
        
        # Calculate target center with sub-pixel precision
        center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
        center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
        
        # Add to position predictor
        self.position_predictor.add_position(center_x, center_y, current_time)
        
        # Predictive targeting for reduced lag
        predicted_position = self.position_predictor.predict_next_position()
        if predicted_position:
            center_x, center_y = predicted_position
        
        # Calculate errors
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        
        # Dynamic dead zone based on distance and confidence
        distance_factor = max(0.5, min(3.0, self.current_distance / 5.0))
        confidence_factor = max(0.5, confidence)
        self.dynamic_dead_zone = DEAD_ZONE * distance_factor / confidence_factor
        
        # Enhanced movement calculation
        if abs(error_x) > self.dynamic_dead_zone or abs(error_y) > self.dynamic_dead_zone:
            # Adaptive sensitivity based on multiple factors
            adaptive_pan_sens = self.dynamic_sensitivity['pan']
            adaptive_tilt_sens = self.dynamic_sensitivity['tilt']
            
            # Distance-based sensitivity adjustment
            distance_multiplier = min(2.0, max(0.3, 2.0 / max(self.current_distance, 0.5)))
            adaptive_pan_sens *= distance_multiplier
            adaptive_tilt_sens *= distance_multiplier
            
            # Confidence-based adjustment
            confidence_multiplier = min(1.5, max(0.7, confidence + 0.3))
            adaptive_pan_sens *= confidence_multiplier
            adaptive_tilt_sens *= confidence_multiplier
            
            # Velocity-based prediction adjustment
            velocity_magnitude = math.sqrt(self.velocity_3d[0]**2 + self.velocity_3d[1]**2)
            if velocity_magnitude > 1.0:  # Fast moving target
                prediction_factor = min(1.5, 1.0 + velocity_magnitude * 0.1)
                adaptive_pan_sens *= prediction_factor
                adaptive_tilt_sens *= prediction_factor
            
            # Calculate adjustments
            pan_adjustment = -error_x * (adaptive_pan_sens / self.frame_width)
            tilt_adjustment = error_y * (adaptive_tilt_sens / self.frame_height)
            
            # Apply movement limits
            pan_adjustment = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, pan_adjustment))
            tilt_adjustment = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, tilt_adjustment))
            
            # Calculate target angles
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            # Enhanced smoothing with adaptive factors
            smoothing_factor = SMOOTHING_FACTOR
            if confidence > 0.8:
                smoothing_factor *= 1.2  # More aggressive for high confidence
            if velocity_magnitude > 2.0:
                smoothing_factor *= 1.3  # More aggressive for fast targets
            
            new_pan = self._enhanced_smooth_angle(current_pan, target_pan, smoothing_factor)
            new_tilt = self._enhanced_smooth_angle(current_tilt, target_tilt, smoothing_factor)
            
            # Add to history for trend analysis
            self.pan_history.append(new_pan)
            self.tilt_history.append(new_tilt)
            
            # Multi-level smoothing
            if len(self.pan_history) >= 3:
                # Use weighted average of recent positions
                weights = np.array([1.0, 2.0, 4.0, 6.0, 8.0][:len(self.pan_history)])
                weights = weights / weights.sum()
                
                avg_pan = np.average(list(self.pan_history), weights=weights)
                avg_tilt = np.average(list(self.tilt_history), weights=weights)
            else:
                avg_pan, avg_tilt = new_pan, new_tilt
            
            # Ultra-fast servo command
            self.servo.move_to_ultra_fast(avg_pan, avg_tilt)
            
            # Update tracking state
            if not self.lock_on_target:
                self.lock_on_target = True
                self.tracking_quality = "LOCKED"
                if self.logger:
                    self.logger.log_event('target_lock', f'Ultra-fast lock at {self.current_distance:.2f}m')
                print(f"🎯 Ultra-fast target lock: {self.current_distance:.2f}m, Conf: {confidence:.3f}")
        
        # Update tracking metrics
        self.last_detection_time = current_time
        self.target_lost_frames = 0
        self.successful_tracks += 1
        self.average_confidence = (self.average_confidence * (self.successful_tracks - 1) + confidence) / self.successful_tracks
        
        # Enhanced GPS waypoint generation
        if self.mavlink_handler and self.current_distance >= MIN_DISTANCE_FOR_GPS:
            gps_point = self.mavlink_handler.add_enhanced_detection_point(
                x, y, z_abs, confidence
            )
        
        # Enhanced logging
        if self.logger:
            distance_data = {
                'distance': self.current_distance,
                'x_position': x,
                'y_position': y,
                'z_position_relative': z_rel,
                'z_position_absolute': z_abs,
                'velocity_3d': self.velocity_3d,
                'angular_width': angular_width,
                'angular_height': angular_height,
                'angular_area': angular_area,
                'bbox_width': bbox.width(),
                'bbox_height': bbox.height(),
                'tracking_quality': self.tracking_quality,
                'prediction_used': predicted_position is not None,
                'dynamic_dead_zone': self.dynamic_dead_zone
            }
            
            self.logger.log_enhanced_frame_data(
                frame_count, servo_state, confidence, True, 
                self.is_tracking_active(), self.target_lost_frames, distance_data
            )
    
    def handle_lost_target_enhanced(self, frame_count):
        """Enhanced lost target handling with smart recovery"""
        self.target_lost_frames += 1
        current_time = time.time()
        
        # Adaptive search based on last known position and velocity
        if self.target_lost_frames == 1 and self.velocity_3d != (0.0, 0.0, 0.0):
            # Predict where target might be
            dt = current_time - self.last_detection_time
            predicted_x = self.last_3d_position[0] + self.velocity_3d[0] * dt
            predicted_y = self.last_3d_position[1] + self.velocity_3d[1] * dt
            
            # Convert back to servo angles for search
            if self.current_distance > 0:
                predicted_pan = math.degrees(math.atan2(predicted_x, predicted_y)) + 90
                # Initiate search around predicted position
                servo_state = self.servo.get_enhanced_state()
                self.servo.move_to_ultra_fast(predicted_pan, servo_state['current_tilt'])
        
        # Progressive search pattern
        elif self.target_lost_frames > 5 and self.target_lost_frames % 10 == 0:
            servo_state = self.servo.get_enhanced_state()
            # Implement spiral search pattern
            search_radius = min(30, self.target_lost_frames * 2)
            search_angle = (self.target_lost_frames // 10) * 45
            
            new_pan = servo_state['current_pan'] + search_radius * math.cos(math.radians(search_angle))
            new_tilt = servo_state['current_tilt'] + search_radius * 0.5 * math.sin(math.radians(search_angle))
            
            new_pan = max(10, min(170, new_pan))
            new_tilt = max(10, min(170, new_tilt))
            
            self.servo.move_to_ultra_fast(new_pan, new_tilt)
        
        # Update tracking state
        if self.target_lost_frames > 15 and self.lock_on_target:
            self.lock_on_target = False
            self.tracking_quality = "SEARCHING"
            if self.logger:
                self.logger.log_event('target_lost', 'Enhanced target lost - intelligent search mode')
            print("🔍 Target lost - intelligent search mode activated")
        
        # Enhanced logging for lost target
        if self.logger:
            servo_state = self.servo.get_enhanced_state()
            self.logger.log_enhanced_frame_data(
                frame_count, servo_state, 0.0, False, 
                self.is_tracking_active(), self.target_lost_frames, None
            )
    
    def _enhanced_smooth_angle(self, current, target, smoothing_factor):
        """Enhanced angle smoothing with adaptive factors"""
        diff = target - current
        
        # Handle angle wrapping
        if abs(diff) > 180:
            if diff > 0:
                diff -= 360
            else:
                diff += 360
        
        # Apply smoothing with limits
        smoothed_diff = diff * smoothing_factor
        smoothed_diff = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, smoothed_diff))
        
        result = current + smoothed_diff
        
        # Ensure result is in valid range
        return max(10, min(170, result))
    
    def is_tracking_active(self):
        """Enhanced tracking active check"""
        time_since_detection = time.time() - self.last_detection_time
        return time_since_detection < DETECTION_TIMEOUT and self.lock_on_target
    
    def get_enhanced_tracking_info(self):
        """Get comprehensive tracking information"""
        quality_metrics = self.distance_calculator.get_measurement_quality()
        
        return {
            'distance': self.current_distance,
            'position_3d': self.current_3d_position,
            'velocity_3d': self.velocity_3d,
            'tracking_quality': self.tracking_quality,
            'lock_on_target': self.lock_on_target,
            'frames_processed': self.frames_processed,
            'successful_tracks': self.successful_tracks,
            'average_confidence': self.average_confidence,
            'measurement_quality': quality_metrics,
            'dynamic_dead_zone': self.dynamic_dead_zone,
            'target_lost_frames': self.target_lost_frames
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
        self.csv_file = self.log_dir / f"enhanced_servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"enhanced_session_{timestamp}.json"
        self.gps_csv_file = self.log_dir / f"enhanced_gps_points_{timestamp}.csv"
        self.altitude_csv_file = self.log_dir / f"altitude_data_{timestamp}.csv"
        
        # Enhanced CSV headers
        self.csv_headers = [
            'timestamp', 'frame_count', 'pan_angle', 'tilt_angle', 'target_pan', 'target_tilt',
            'pan_velocity', 'tilt_velocity', 'pan_acceleration', 'tilt_acceleration',
            'detection_confidence', 'person_detected', 'tracking_active', 'tracking_quality',
            'target_lost_frames', 'distance_meters', 'x_position', 'y_position', 
            'z_position_relative', 'z_position_absolute', 'velocity_x', 'velocity_y', 'velocity_z',
            'angular_width', 'angular_height', 'angular_area', 'bbox_width', 'bbox_height',
            'gps_latitude', 'gps_longitude', 'gps_altitude', 'prediction_used', 'dynamic_dead_zone',
            'measurement_quality', 'servo_queue_size'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_headers)
        
        # Enhanced GPS headers
        self.gps_headers = [
            'timestamp', 'detection_lat', 'detection_lon', 'detection_alt',
            'vehicle_lat', 'vehicle_lon', 'vehicle_alt', 'vehicle_alt_rel', 'vehicle_heading',
            'relative_x', 'relative_y', 'relative_z', 'confidence', 'gps_accuracy',
            'satellites', 'fix_type'
        ]
        
        with open(self.gps_csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.gps_headers)
        
        # Altitude data headers
        self.altitude_headers = [
            'timestamp', 'gps_altitude', 'relative_altitude', 'terrain_altitude',
            'barometric_altitude', 'altitude_accuracy'
        ]
        
        with open(self.altitude_csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.altitude_headers)
        
        # Enhanced session data
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'enhanced_features': [
                'ultra_fast_servos', 'enhanced_distance_calculation', 'predictive_tracking',
                'automatic_calibration', 'live_altitude', 'robust_mavlink_connection'
            ],
            'log_files': {
                'csv': str(self.csv_file),
                'json': str(self.json_file),
                'gps_csv': str(self.gps_csv_file),
                'altitude_csv': str(self.altitude_csv_file)
            },
            'statistics': {
                'total_detections': 0,
                'total_movements': 0,
                'min_distance': float('inf'),
                'max_distance': 0.0,
                'avg_distance': 0.0,
                'distance_samples': 0,
                'gps_points_created': 0,
                'altitude_samples': 0,
                'tracking_quality_distribution': {},
                'prediction_usage': 0,
                'calibration_samples': 0
            },
            'events': [],
            'performance_metrics': {
                'avg_processing_time': 0.0,
                'max_processing_time': 0.0,
                'servo_response_times': [],
                'gps_update_intervals': []
            }
        }
        
        print(f"📊 Enhanced data logging to: {self.log_dir}")
    
    def log_enhanced_frame_data(self, frame_count, servo_state, detection_confidence, 
                               person_detected, tracking_active, target_lost_frames, distance_data=None):
        """Enhanced frame data logging"""
        try:
            # Extract servo data
            pan_angle = servo_state.get('current_pan', 0)
            tilt_angle = servo_state.get('current_tilt', 0)
            target_pan = servo_state.get('target_pan', 0)
            target_tilt = servo_state.get('target_tilt', 0)
            pan_velocity = servo_state.get('velocity_pan', 0)
            tilt_velocity = servo_state.get('velocity_tilt', 0)
            pan_acceleration = servo_state.get('acceleration_pan', 0)
            tilt_acceleration = servo_state.get('acceleration_tilt', 0)
            servo_queue_size = servo_state.get('queue_size', 0)
            
            # Initialize default values
            distance = x_pos = y_pos = z_pos_rel = z_pos_abs = 0.0
            vel_x = vel_y = vel_z = 0.0
            angular_width = angular_height = angular_area = 0.0
            bbox_width = bbox_height = 0.0
            gps_lat = gps_lon = gps_alt = 0.0
            prediction_used = False
            dynamic_dead_zone = 0.0
            measurement_quality = 'UNKNOWN'
            tracking_quality = 'NONE'
            
            if distance_data:
                distance = distance_data.get('distance', 0.0)
                x_pos = distance_data.get('x_position', 0.0)
                y_pos = distance_data.get('y_position', 0.0)
                z_pos_rel = distance_data.get('z_position_relative', 0.0)
                z_pos_abs = distance_data.get('z_position_absolute', 0.0)
                
                velocity_3d = distance_data.get('velocity_3d', (0, 0, 0))
                vel_x, vel_y, vel_z = velocity_3d
                
                angular_width = distance_data.get('angular_width', 0.0)
                angular_height = distance_data.get('angular_height', 0.0)
                angular_area = distance_data.get('angular_area', 0.0)
                bbox_width = distance_data.get('bbox_width', 0.0)
                bbox_height = distance_data.get('bbox_height', 0.0)
                
                prediction_used = distance_data.get('prediction_used', False)
                dynamic_dead_zone = distance_data.get('dynamic_dead_zone', 0.0)
                tracking_quality = distance_data.get('tracking_quality', 'NONE')
                
                # Handle GPS waypoint creation
                if self.gps_handler and distance >= MIN_DISTANCE_FOR_GPS:
                    status = self.gps_handler.get_enhanced_status()
                    if status['connected'] and status['gps_fix'] >= 3:
                        gps_lat = status['latitude']
                        gps_lon = status['longitude']
                        gps_alt = status['altitude']
                        
                        self.session_data['statistics']['gps_points_created'] += 1
            
            # Log altitude data separately
            if self.gps_handler:
                self._log_altitude_data()
            
            # Update statistics
            stats = self.session_data['statistics']
            if person_detected:
                stats['total_detections'] += 1
                
            if abs(pan_velocity) > 1 or abs(tilt_velocity) > 1:
                stats['total_movements'] += 1
                
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
                    pan_velocity, tilt_velocity, pan_acceleration, tilt_acceleration,
                    detection_confidence, person_detected, tracking_active, tracking_quality,
                    target_lost_frames, distance, x_pos, y_pos, z_pos_rel, z_pos_abs,
                    vel_x, vel_y, vel_z, angular_width, angular_height, angular_area,
                    bbox_width, bbox_height, gps_lat, gps_lon, gps_alt,
                    prediction_used, dynamic_dead_zone, measurement_quality, servo_queue_size
                ])
                
        except Exception as e:
            print(f"Enhanced logging error: {e}")
    
    def _log_altitude_data(self):
        """Log altitude data from MAVLink"""
        try:
            status = self.gps_handler.get_enhanced_status()
            altitude_data = status.get('altitude_data', {})
            
            if altitude_data:
                with open(self.altitude_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.time(),
                        altitude_data.get('gps_altitude', 0),
                        altitude_data.get('relative_altitude', 0),
                        altitude_data.get('terrain_altitude', 0),
                        altitude_data.get('barometric_altitude', 0),
                        altitude_data.get('altitude_accuracy', 'UNKNOWN')
                    ])
                
                self.session_data['statistics']['altitude_samples'] += 1
                
        except Exception as e:
            print(f"Altitude logging error: {e}")
    
    def log_calibration_event(self, event_type, data):
        """Log calibration events"""
        self.log_event(f'calibration_{event_type}', str(data))
        self.session_data['statistics']['calibration_samples'] += 1
    
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
        
        # Add comprehensive system status
        if self.gps_handler:
            self.session_data['final_gps_status'] = self.gps_handler.get_enhanced_status()
        
        # Calculate performance metrics
        stats = self.session_data['statistics']
        if stats['total_detections'] > 0:
            self.session_data['performance_summary'] = {
                'detection_rate': stats['total_detections'] / max(1, stats['total_movements']),
                'gps_success_rate': stats['gps_points_created'] / max(1, stats['total_detections']),
                'prediction_usage_rate': stats['prediction_usage'] / max(1, stats['total_detections']),
                'average_distance': stats['avg_distance'],
                'distance_range': f"{stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m"
            }
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            # Print comprehensive summary
            print(f"\n📊 Enhanced Session Complete:")
            print(f"   Total Detections: {stats['total_detections']}")
            print(f"   GPS Points Created: {stats['gps_points_created']}")
            print(f"   Altitude Samples: {stats['altitude_samples']}")
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

class EnhancedAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.processing_times = deque(maxlen=100)
        self.last_fps_update = time.time()
        self.fps = 0.0
    
    def new_function(self):
        return "Enhanced Ultra-Fast Tracking with Cube Orange Integration: "
    
    def get_fps(self):
        return self.fps
    
    def update_fps(self):
        current_time = time.time()
        if current_time - self.last_fps_update > 1.0:
            self.fps = self.frame_counter / max(1, current_time - self.last_fps_update)
            self.frame_counter = 0
            self.last_fps_update = current_time

# ==============================================================================
# ENHANCED MAIN CALLBACK FUNCTION
# ==============================================================================

def enhanced_ultra_fast_app_callback(pad, info, user_data):
    """Enhanced callback with all improvements"""
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
            tracker.update_frame_properties(width, height)
    
    # Enhanced frame processing
    frame = None
    if user_data.use_frame:
        format, width, height = get_caps_from_pad(pad)
        if format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Enhanced detection processing
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Advanced detection filtering and selection
    valid_detections = []
    for detection in detections:
        if detection.get_label() == "person":
            confidence = detection.get_confidence()
            if confidence >= MIN_CONFIDENCE:
                bbox = detection.get_bbox()
                area = bbox.width() * bbox.height()
                
                # Enhanced quality metrics
                aspect_ratio = bbox.width() / max(bbox.height(), 0.001)
                reasonable_aspect = 0.2 < aspect_ratio < 2.0
                reasonable_size = 0.001 < area < 0.5
                
                if reasonable_aspect and reasonable_size:
                    valid_detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'quality_score': confidence * area * (1.0 if 0.3 < aspect_ratio < 0.8 else 0.7)
                    })
    
    # Select best detection using enhanced criteria
    best_detection = None
    if valid_detections:
        # Sort by quality score
        valid_detections.sort(key=lambda x: x['quality_score'], reverse=True)
        best_detection = valid_detections[0]
    
    # Enhanced tracking
    if best_detection:
        tracker.track_person_ultra_fast(
            best_detection['bbox'], 
            best_detection['confidence'], 
            frame_count
        )
        
        # Enhanced calibration
        if auto_calibration.calibration_active:
            distance_info = tracker.get_enhanced_tracking_info()
            auto_calibration.add_calibration_measurement(
                best_detection['bbox'],
                distance_info['distance'],
                best_detection['confidence'],
                tracker.frame_height
            )
        
        # Enhanced progress reporting
        if frame_count % 60 == 0:
            tracking_info = tracker.get_enhanced_tracking_info()
            print(f"🏃 Enhanced Tracking: Conf {best_detection['confidence']:.3f}, "
                  f"Dist: {tracking_info['distance']:.2f}m, "
                  f"Quality: {tracking_info['tracking_quality']}, "
                  f"FPS: {user_data.get_fps():.1f}")
    else:
        tracker.handle_lost_target_enhanced(frame_count)
    
    # Enhanced frame visualization
    if user_data.use_frame and frame is not None:
        frame = enhanced_draw_overlay(frame, best_detection, frame_count)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Performance tracking
    processing_time = time.time() - start_time
    user_data.processing_times.append(processing_time)
    
    return Gst.PadProbeReturn.OK

def enhanced_draw_overlay(frame, best_detection, frame_count):
    """Enhanced overlay drawing with comprehensive information"""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Draw enhanced crosshair
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
    
    # Enhanced status display
    if best_detection:
        tracking_info = tracker.get_enhanced_tracking_info()
        distance = tracking_info['distance']
        x, y, z_rel, z_abs = tracking_info['position_3d']
        quality = tracking_info['tracking_quality']
        
        # Calibration status
        if auto_calibration.calibration_active:
            cal_status = auto_calibration.get_calibration_status()
            cv2.putText(frame, f"CALIBRATION: {cal_status['phase']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Progress: {cal_status['progress']:.1f}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"ENHANCED TRACKING: {quality}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Enhanced 3D position display
        cv2.putText(frame, f"3D: ({x:.1f}, {y:.1f}, {z_rel:.1f})m", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Abs Alt: {z_abs:.1f}m", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Velocity display
        vel_x, vel_y, vel_z = tracking_info['velocity_3d']
        velocity_mag = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        cv2.putText(frame, f"Velocity: {velocity_mag:.2f}m/s", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Enhanced bounding box
        bbox = best_detection['bbox']
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int((bbox.xmin() + bbox.width()) * width)
        y2 = int((bbox.ymin() + bbox.height()) * height)
        
        # Color based on quality
        if quality == "LOCKED":
            box_color = (0, 255, 0)
        elif quality == "SEARCHING":
            box_color = (255, 255, 0)
        else:
            box_color = (255, 0, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Enhanced info box
        info_text = f"{distance:.1f}m | {best_detection['confidence']:.3f}"
        cv2.putText(frame, info_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Target prediction indicator
        if tracking_info.get('prediction_used', False):
            cv2.circle(frame, (x1 + (x2-x1)//2, y1 + (y2-y1)//2), 8, (255, 0, 255), 2)
    
    # Enhanced servo state display
    servo_state = fast_servo_controller.get_enhanced_state()
    cv2.putText(frame, f"Pan: {servo_state['current_pan']:.1f}° → {servo_state['target_pan']:.1f}°", 
               (10, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Tilt: {servo_state['current_tilt']:.1f}° → {servo_state['target_tilt']:.1f}°", 
               (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Vel: {servo_state['velocity_pan']:.1f}°/s, {servo_state['velocity_tilt']:.1f}°/s", 
               (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(frame, f"Queue: {servo_state['queue_size']}", 
               (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Enhanced GPS/MAVLink status
    if mavlink_handler:
        status = mavlink_handler.get_enhanced_status()
        altitude_data = status.get('altitude_data', {})
        
        # Connection status
        if status['connected']:
            connection_color = (0, 255, 0)
            conn_text = f"MAVLink: CONNECTED ({status['connection_uptime']:.0f}s)"
        else:
            connection_color = (0, 0, 255)
            conn_text = "MAVLink: DISCONNECTED"
        
        cv2.putText(frame, conn_text, (width - 400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, connection_color, 2)
        
        # GPS status with enhanced info
        if status['gps_fix'] >= 3:
            gps_color = (0, 255, 0)
            gps_text = f"GPS: {status['satellites']} sats (FIX)"
        elif status['gps_fix'] >= 2:
            gps_color = (255, 255, 0)
            gps_text = f"GPS: {status['satellites']} sats (2D)"
        else:
            gps_color = (0, 0, 255)
            gps_text = f"GPS: {status['satellites']} sats (NO FIX)"
        
        cv2.putText(frame, gps_text, (width - 400, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 2)
        
        # Live altitude display
        if altitude_data:
            alt_text = f"Alt: GPS {status['altitude']:.1f}m"
            cv2.putText(frame, alt_text, (width - 400, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if status['altitude_relative'] > 0:
                rel_alt_text = f"Rel: {status['altitude_relative']:.1f}m"
                cv2.putText(frame, rel_alt_text, (width - 400, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if altitude_data.get('barometric_altitude', 0) > 0:
                baro_text = f"Baro: {altitude_data['barometric_altitude']:.1f}m"
                cv2.putText(frame, baro_text, (width - 400, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Mission status
        if status['mission_count'] > 0:
            mission_text = f"Waypoints: {status['mission_count']}"
            cv2.putText(frame, mission_text, (width - 400, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Performance metrics
    fps_text = f"FPS: {user_data.get_fps():.1f}"
    cv2.putText(frame, fps_text, (width - 150, height - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

# ==============================================================================
# ENHANCED INITIALIZATION AND MAIN EXECUTION
# ==============================================================================

def initialize_enhanced_system():
    """Initialize all enhanced system components"""
    global mavlink_handler, data_logger, fast_servo_controller, tracker, auto_calibration
    
    print("🚀 Initializing ENHANCED servo system with Cube Orange integration...")
    
    # Initialize enhanced MAVLink connection with retry
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
    data_logger.log_event('system_init', 'Enhanced servo tracking system starting')
    
    # Initialize ultra-fast servo controller
    fast_servo_controller = UltraFastServoController(data_logger)
    data_logger.log_event('servo_init', 'Ultra-fast servo controller initialized')
    
    # Initialize robust auto-calibration
    auto_calibration = RobustAutoCalibration(data_logger)
    data_logger.log_event('calibration_init', 'Robust auto-calibration system initialized')
    
    # Initialize ultra-fast tracker
    tracker = UltraFastTracker(fast_servo_controller, data_logger, mavlink_handler)
    data_logger.log_event('tracker_init', 'Ultra-fast tracker with lag reduction initialized')
    
    return True

def print_enhanced_system_status():
    """Print comprehensive system status"""
    print("\n" + "="*80)
    print("🎯 ENHANCED SERVO TRACKING SYSTEM STATUS")
    print("="*80)
    
    # MAVLink status
    if mavlink_handler:
        status = mavlink_handler.get_enhanced_status()
        print(f"\n🛰️ Enhanced MAVLink Status:")
        print(f"   Connection: {'✅ CONNECTED' if status['connected'] else '❌ DISCONNECTED'}")
        print(f"   Uptime: {status['connection_uptime']:.1f}s")
        print(f"   Messages: {status['total_messages']}")
        print(f"   GPS Fix: {status['gps_fix']} ({'3D' if status['gps_fix'] >= 3 else '2D' if status['gps_fix'] >= 2 else 'NO FIX'})")
        print(f"   Satellites: {status['satellites']}")
        print(f"   Position: {status['latitude']:.6f}, {status['longitude']:.6f}")
        print(f"   GPS Altitude: {status['altitude']:.1f}m")
        print(f"   Relative Altitude: {status['altitude_relative']:.1f}m")
        print(f"   Heading: {status['heading']:.1f}°")
        print(f"   Home Set: {'✅' if status['home_set'] else '❌'}")
        print(f"   Mission Count: {status['mission_count']}")
        
        # Altitude data
        altitude_data = status.get('altitude_data', {})
        if altitude_data:
            print(f"   Barometric Alt: {altitude_data.get('barometric_altitude', 0):.1f}m")
            print(f"   Terrain Alt: {altitude_data.get('terrain_altitude', 0):.1f}m")
            print(f"   Altitude Accuracy: {altitude_data.get('altitude_accuracy', 'UNKNOWN')}")
    else:
        print(f"\n🛰️ MAVLink Status: ❌ NOT AVAILABLE")
    
    # Servo status
    servo_state = fast_servo_controller.get_enhanced_state()
    print(f"\n🔧 Ultra-Fast Servo Status:")
    print(f"   Pan: {servo_state['current_pan']:.1f}° (Target: {servo_state['target_pan']:.1f}°)")
    print(f"   Tilt: {servo_state['current_tilt']:.1f}° (Target: {servo_state['target_tilt']:.1f}°)")
    print(f"   Velocities: {servo_state['velocity_pan']:.1f}°/s, {servo_state['velocity_tilt']:.1f}°/s")
    print(f"   Accelerations: {servo_state['acceleration_pan']:.1f}°/s², {servo_state['acceleration_tilt']:.1f}°/s²")
    print(f"   Queue Size: {servo_state['queue_size']}")
    
    # Calibration status
    if auto_calibration:
        cal_status = auto_calibration.get_calibration_status()
        print(f"\n📸 Auto-Calibration Status:")
        print(f"   Active: {'✅' if cal_status['active'] else '❌'}")
        print(f"   Complete: {'✅' if cal_status['complete'] else '❌'}")
        print(f"   Phase: {cal_status['phase']}")
        print(f"   Progress: {cal_status['progress']:.1f}% ({cal_status['samples_collected']}/{cal_status['required_samples']})")
        print(f"   Time Elapsed: {cal_status['time_elapsed']:.1f}s")
        if cal_status['active']:
            print(f"   Timeout Remaining: {cal_status['timeout_remaining']:.1f}s")
    
    # Tracking status
    tracking_info = tracker.get_enhanced_tracking_info()
    print(f"\n🎯 Enhanced Tracking Status:")
    print(f"   Quality: {tracking_info['tracking_quality']}")
    print(f"   Lock: {'✅' if tracking_info['lock_on_target'] else '❌'}")
    print(f"   Frames Processed: {tracking_info['frames_processed']}")
    print(f"   Successful Tracks: {tracking_info['successful_tracks']}")
    print(f"   Average Confidence: {tracking_info['average_confidence']:.3f}")
    print(f"   Current Distance: {tracking_info['distance']:.2f}m")
    print(f"   Dynamic Dead Zone: {tracking_info['dynamic_dead_zone']:.1f}")
    
    measurement_quality = tracking_info['measurement_quality']
    print(f"   Measurement Quality: {measurement_quality['quality']} ({measurement_quality['confidence']:.3f})")
    
    # Data logging status
    print(f"\n📊 Enhanced Data Logging:")
    print(f"   Location: {data_logger.log_dir}")
    print(f"   CSV File: {data_logger.csv_file.name}")
    print(f"   GPS File: {data_logger.gps_csv_file.name}")
    print(f"   Altitude File: {data_logger.altitude_csv_file.name}")
    
    print("\n" + "="*80)

def enhanced_shutdown_sequence():
    """Enhanced shutdown with proper cleanup"""
    print("\n🛑 Enhanced shutdown sequence initiated...")
    
    try:
        # Stop calibration
        if auto_calibration and auto_calibration.calibration_active:
            print("📸 Stopping auto-calibration...")
            auto_calibration.calibration_active = False
        
        # Finalize data logging
        print("📊 Finalizing enhanced logs...")
        data_logger.finalize_enhanced_session()
        
        # Shutdown servo controller
        print("🔧 Shutting down ultra-fast servo controller...")
        fast_servo_controller.shutdown()
        
        # Shutdown MAVLink
        if mavlink_handler:
            print("🛰️ Closing enhanced MAVLink connection...")
            mavlink_handler.shutdown()
        
        print("✅ Enhanced shutdown complete")
        
    except Exception as e:
        print(f"❌ Error during enhanced shutdown: {e}")

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
        # Initialize enhanced system
        if not initialize_enhanced_system():
            print("❌ Enhanced system initialization failed")
            exit(1)
        
        # Print system status
        print_enhanced_system_status()
        
        # Initialize enhanced callback
        user_data = EnhancedAppCallback()
        app = GStreamerDetectionApp(enhanced_ultra_fast_app_callback, user_data)
        
        print(f"\n🚀 Starting ENHANCED ultra-fast tracking with Cube Orange integration...")
        print(f"📊 Enhanced data output: {data_logger.log_dir}")
        print("\nPress Ctrl+C to stop")
        print("\nENHANCED FEATURES ACTIVE:")
        print("  ✅ Ultra-fast servo control with reduced lag")
        print("  ✅ Live altitude readings from Cube Orange")
        print("  ✅ Robust automatic calibration")
        print("  ✅ Enhanced camera feed processing")
        print("  ✅ Improved waypoint generation accuracy")
        print("  ✅ Reliable MAVLink/GPS connection with auto-reconnect")
        print("  ✅ Predictive tracking with lag compensation")
        print("  ✅ Enhanced distance calculation with multiple methods")
        print("  ✅ Comprehensive data logging and analysis")
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 User requested shutdown...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        if data_logger:
            data_logger.log_event('error', f'Application error: {str(e)}')
    finally:
        enhanced_shutdown_sequence()

# Global variables (initialized in main)
mavlink_handler = None
data_logger = None
fast_servo_controller = None
tracker = None
auto_calibration = None
user_data = None
