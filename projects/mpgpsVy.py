from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import sys
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
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import signal
import atexit

try:
    from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
    from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
    from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp
except ImportError as e:
    print(f"Error importing Hailo modules: {e}")
    print("Please ensure Hailo AI SDK is properly installed")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class SystemConfig:
    """Centralized configuration for the tracking system"""
    
    # Servo tracking parameters
    DEAD_ZONE: float = 15.0
    SMOOTHING_FACTOR: float = 0.35
    MAX_STEP_SIZE: float = 5.0
    MIN_CONFIDENCE: float = 0.3
    DETECTION_TIMEOUT: float = 2.0
    PAN_SENSITIVITY: float = 45.0
    TILT_SENSITIVITY: float = 35.0
    FRAME_SKIP_COUNT: int = 1
    DETECTION_HISTORY_SIZE: int = 3

    # Camera parameters
    CAMERA_FOV_HORIZONTAL: float = 79.9
    CAMERA_FOV_VERTICAL: float = 64.3
    AVERAGE_PERSON_HEIGHT: float = 1.7
    AVERAGE_PERSON_WIDTH: float = 0.45
    FOCAL_LENGTH_PIXELS: float = 382.0

    # Physical setup
    SERVO_MOUNT_HEIGHT: float = 1.3
    CAMERA_TILT_OFFSET: float = 5.0

    # MAVLink configuration
    MAVLINK_CONNECTION: str = '/dev/serial0'
    MAVLINK_BAUD: int = 57600
    MAVLINK_SYSTEM_ID: int = 255
    MAVLINK_COMPONENT_ID: int = 190

    # GPS waypoint parameters
    GPS_UPDATE_INTERVAL: float = 1.0
    MIN_DISTANCE_FOR_GPS: float = 3.0
    MAX_GPS_POINTS: int = 100
    WAYPOINT_MODE: str = "ADD"  # "ADD", "REPLACE", "CLEAR_OLD"
    WAYPOINT_CLEAR_TIMEOUT: float = 300.0
    MAX_WAYPOINTS: int = 15

    # Altitude configuration
    USE_RELATIVE_ALTITUDE: bool = True
    TARGET_HEIGHT_ABOVE_GROUND: float = 2.0
    MIN_ALTITUDE_AGL: float = 5.0
    MAX_ALTITUDE_AGL: float = 100.0

    # Calibration
    CALIBRATION_MODE: bool = False
    CALIBRATION_DISTANCE: float = 2.0

    # Performance settings
    LOG_LEVEL: str = "INFO"
    MAX_LOG_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    SERVO_UPDATE_RATE: float = 50.0  # Hz

# Global configuration instance
config = SystemConfig()

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging():
    """Setup logging configuration with rotating file handler"""
    try:
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Create rotating file handler
        log_file = Path(__file__).parent / 'servo_tracking.log'
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=config.MAX_LOG_FILE_SIZE, 
            backupCount=3
        )
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Reduce verbosity of some modules
        logging.getLogger('pymavlink').setLevel(logging.WARNING)
        logging.getLogger('adafruit_pca9685').setLevel(logging.WARNING)
        
        return logging.getLogger(__name__)
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

logger = setup_logging()

# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class Position3D:
    """3D position representation"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class AltitudeInfo:
    """Comprehensive altitude information"""
    current_alt: float = 0.0
    alt_msl: float = 0.0
    alt_relative: float = 0.0
    alt_agl: Optional[float] = None
    altitude_type: str = "unknown"
    rangefinder_available: bool = False

@dataclass
class GPSPoint:
    """GPS waypoint information"""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    altitude_type: str
    relative_position: Position3D
    distance_2d: float
    distance_3d: float
    confidence: float
    vehicle_position: Dict[str, Any]

@dataclass
class TrackingState:
    """Current tracking state"""
    is_active: bool = False
    target_locked: bool = False
    last_detection_time: float = 0.0
    lost_frames: int = 0
    current_distance: float = 0.0
    current_position: Position3D = None
    
    def __post_init__(self):
        if self.current_position is None:
            self.current_position = Position3D()

# ==============================================================================
# MAVLINK GPS HANDLER
# ==============================================================================

class MAVLinkGPSHandler:
    """Optimized MAVLink GPS handler with thread safety"""
    
    def __init__(self, connection_string: str = None, baud: int = None):
        self.connection_string = connection_string or config.MAVLINK_CONNECTION
        self.baud = baud or config.MAVLINK_BAUD
        
        # Connection state
        self.mavlink_connection: Optional[mavutil.mavlink_connection] = None
        self.running = True
        self.connected = False
        self.mavlink_thread: Optional[threading.Thread] = None
        
        # Thread-safe GPS state
        self._gps_lock = threading.RLock()
        self._current_state = {
            'lat': 0.0, 'lon': 0.0, 'alt': 0.0,
            'alt_msl': 0.0, 'alt_relative': 0.0, 'alt_agl': 0.0,
            'heading': 0.0, 'gps_fix_type': 0, 'satellites_visible': 0,
            'last_update': 0.0
        }
        
        # Home position
        self.home_position = {'lat': None, 'lon': None, 'alt': None, 'alt_msl': None}
        
        # Mission state
        self.mission_count = 0
        self.current_wp_seq = 0
        self.last_waypoint_time = 0.0
        
        # GPS logging
        self.gps_points = deque(maxlen=config.MAX_GPS_POINTS)
        self.last_point_time = 0.0
        
        # Constants
        self.EARTH_RADIUS = 6371000.0
        
        # Rangefinder state
        self.rangefinder_available = False
        
        # Initialize connection
        if not self.connect():
            logger.warning("MAVLink connection failed - GPS functionality disabled")
    
    def connect(self) -> bool:
        """Establish MAVLink connection with comprehensive error handling"""
        try:
            logger.info(f"Connecting to CubeOrange at {self.connection_string}...")
            
            self.mavlink_connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
                source_system=config.MAVLINK_SYSTEM_ID,
                source_component=config.MAVLINK_COMPONENT_ID
            )
            
            logger.info("Waiting for heartbeat...")
            self.mavlink_connection.wait_heartbeat(timeout=10)
            
            self.connected = True
            logger.info("MAVLink connection established!")
            
            self.request_data_streams()
            self.mavlink_thread = threading.Thread(target=self._mavlink_receiver, daemon=True)
            self.mavlink_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"MAVLink connection failed: {e}")
            self.connected = False
            return False
    
    def request_data_streams(self):
        """Request required data streams"""
        if not self.connected or not self.mavlink_connection:
            return
            
        try:
            # Request all data streams at 10Hz
            self.mavlink_connection.mav.request_data_stream_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10, 1
            )
            
            # Request specific altitude messages
            messages_to_request = [
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE
            ]
            
            for msg_id in messages_to_request:
                self.mavlink_connection.mav.command_long_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0, msg_id, 100000, 0, 0, 0, 0, 0  # 10Hz
                )
            
            logger.info("Data streams requested successfully")
            
        except Exception as e:
            logger.error(f"Error requesting data streams: {e}")
    
    def _mavlink_receiver(self):
        """Main MAVLink message receiver loop"""
        msg_handlers = {
            'GPS_RAW_INT': self._handle_gps_raw,
            'GLOBAL_POSITION_INT': self._handle_global_position,
            'ATTITUDE': self._handle_attitude,
            'HOME_POSITION': self._handle_home_position,
            'DISTANCE_SENSOR': self._handle_distance_sensor,
            'RANGEFINDER': self._handle_distance_sensor,
            'MISSION_CURRENT': self._handle_mission_current,
            'MISSION_ITEM_REACHED': self._handle_mission_reached
        }
        
        logger.info("MAVLink receiver thread started")
        
        while self.running and self.connected:
            try:
                if not self.mavlink_connection:
                    break
                    
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                handler = msg_handlers.get(msg_type)
                if handler:
                    handler(msg)
                    
            except Exception as e:
                if self.running:
                    logger.error(f"MAVLink receive error: {e}")
                    time.sleep(0.1)  # Prevent rapid error loops
        
        logger.info("MAVLink receiver thread stopped")
    
    def _handle_gps_raw(self, msg):
        """Handle GPS_RAW_INT messages"""
        with self._gps_lock:
            self._current_state.update({
                'lat': msg.lat / 1e7,
                'lon': msg.lon / 1e7,
                'alt_msl': msg.alt / 1000.0,
                'gps_fix_type': msg.fix_type,
                'satellites_visible': msg.satellites_visible,
                'last_update': time.time()
            })
    
    def _handle_global_position(self, msg):
        """Handle GLOBAL_POSITION_INT messages"""
        with self._gps_lock:
            self._current_state.update({
                'lat': msg.lat / 1e7,
                'lon': msg.lon / 1e7,
                'alt_msl': msg.alt / 1000.0,
                'alt_relative': msg.relative_alt / 1000.0,
                'heading': msg.hdg / 100.0
            })
            
            # Set primary altitude based on configuration
            if config.USE_RELATIVE_ALTITUDE:
                self._current_state['alt'] = self._current_state['alt_relative']
            else:
                self._current_state['alt'] = self._current_state['alt_msl']
    
    def _handle_attitude(self, msg):
        """Handle ATTITUDE messages"""
        with self._gps_lock:
            self._current_state['heading'] = math.degrees(msg.yaw) % 360
    
    def _handle_home_position(self, msg):
        """Handle HOME_POSITION messages"""
        self.home_position.update({
            'lat': msg.latitude / 1e7,
            'lon': msg.longitude / 1e7,
            'alt': msg.altitude / 1000.0,
            'alt_msl': msg.altitude / 1000.0
        })
        logger.info(f"Home position: {self.home_position['lat']:.6f}, "
                   f"{self.home_position['lon']:.6f} @ {self.home_position['alt']:.1f}m")
    
    def _handle_distance_sensor(self, msg):
        """Handle rangefinder/distance sensor messages"""
        if hasattr(msg, 'current_distance') and msg.current_distance > 0:
            with self._gps_lock:
                self._current_state['alt_agl'] = msg.current_distance / 100.0  # cm to m
                self.rangefinder_available = True
    
    def _handle_mission_current(self, msg):
        """Handle MISSION_CURRENT messages"""
        self.current_wp_seq = msg.seq
    
    def _handle_mission_reached(self, msg):
        """Handle MISSION_ITEM_REACHED messages"""
        logger.info(f"Waypoint {msg.seq} reached")
    
    @property
    def current_position(self) -> Tuple[float, float, float]:
        """Thread-safe access to current position"""
        with self._gps_lock:
            return (self._current_state['lat'], 
                   self._current_state['lon'], 
                   self._current_state['alt'])
    
    @property
    def altitude_info(self) -> AltitudeInfo:
        """Get comprehensive altitude information"""
        with self._gps_lock:
            return AltitudeInfo(
                current_alt=self._current_state['alt'],
                alt_msl=self._current_state['alt_msl'],
                alt_relative=self._current_state['alt_relative'],
                alt_agl=self._current_state['alt_agl'] if self.rangefinder_available else None,
                altitude_type='relative' if config.USE_RELATIVE_ALTITUDE else 'msl',
                rangefinder_available=self.rangefinder_available
            )
    
    def calculate_waypoint_altitude(self, target_distance_3d: float) -> float:
        """Calculate appropriate waypoint altitude with safety constraints"""
        try:
            alt_info = self.altitude_info
            base_altitude = alt_info.current_alt
            target_altitude = base_altitude + config.TARGET_HEIGHT_ABOVE_GROUND
            
            # Apply safety constraints if rangefinder available
            if alt_info.rangefinder_available and alt_info.alt_agl is not None:
                min_alt_required = alt_info.alt_agl + config.MIN_ALTITUDE_AGL
                max_alt_allowed = alt_info.alt_agl + config.MAX_ALTITUDE_AGL
                target_altitude = max(min_alt_required, min(target_altitude, max_alt_allowed))
            
            # Ensure positive altitude for relative mode
            if config.USE_RELATIVE_ALTITUDE:
                target_altitude = max(target_altitude, config.MIN_ALTITUDE_AGL)
            
            return target_altitude
            
        except Exception as e:
            logger.error(f"Error calculating waypoint altitude: {e}")
            return base_altitude + config.TARGET_HEIGHT_ABOVE_GROUND
    
    def calculate_gps_position(self, x_meters: float, y_meters: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate GPS coordinates from relative position"""
        with self._gps_lock:
            if self._current_state['gps_fix_type'] < 3:
                return None, None
            
            try:
                # Calculate bearing and distance
                bearing_to_target = math.degrees(math.atan2(x_meters, y_meters))
                absolute_bearing = (self._current_state['heading'] + bearing_to_target) % 360
                distance = math.sqrt(x_meters**2 + y_meters**2)
                
                # Convert to GPS coordinates using haversine formula
                lat_rad = math.radians(self._current_state['lat'])
                lon_rad = math.radians(self._current_state['lon'])
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
                
            except Exception as e:
                logger.error(f"Error calculating GPS position: {e}")
                return None, None
    
    def add_detection_point(self, x_meters: float, y_meters: float, z_meters: float, 
                           confidence: float) -> Optional[GPSPoint]:
        """Add detection point and create waypoint if appropriate"""
        if not self.connected:
            return None
            
        current_time = time.time()
        if current_time - self.last_point_time < config.GPS_UPDATE_INTERVAL:
            return None
        
        distance_2d = math.sqrt(x_meters**2 + y_meters**2)
        distance_3d = math.sqrt(x_meters**2 + y_meters**2 + z_meters**2)
        
        if distance_2d < config.MIN_DISTANCE_FOR_GPS:
            return None
        
        lat, lon = self.calculate_gps_position(x_meters, y_meters)
        if lat is None or lon is None:
            return None
        
        waypoint_alt = self.calculate_waypoint_altitude(distance_3d)
        
        # Create GPS point
        with self._gps_lock:
            vehicle_pos = self._current_state.copy()
        
        gps_point = GPSPoint(
            timestamp=current_time,
            latitude=lat,
            longitude=lon,
            altitude=waypoint_alt,
            altitude_type='relative' if config.USE_RELATIVE_ALTITUDE else 'msl',
            relative_position=Position3D(x_meters, y_meters, z_meters),
            distance_2d=distance_2d,
            distance_3d=distance_3d,
            confidence=confidence,
            vehicle_position=vehicle_pos
        )
        
        self.gps_points.append(gps_point)
        self.last_point_time = current_time
        
        # Upload waypoint based on mode
        success = self._upload_waypoint_by_mode(lat, lon, waypoint_alt)
        
        if success:
            bearing = math.degrees(math.atan2(x_meters, y_meters))
            with self._gps_lock:
                bearing = (self._current_state['heading'] + bearing) % 360
            
            alt_type = "REL" if config.USE_RELATIVE_ALTITUDE else "MSL"
            logger.info(f"Waypoint: {distance_2d:.1f}m @ {bearing:.0f}°, alt: {waypoint_alt:.1f}m ({alt_type})")
            self.last_waypoint_time = current_time
        
        return gps_point
    
    def _upload_waypoint_by_mode(self, lat: float, lon: float, alt: float) -> bool:
        """Upload waypoint according to configured mode"""
        try:
            if config.WAYPOINT_MODE == "REPLACE":
                self.clear_mission()
                return self.upload_waypoint(lat, lon, alt, 1)
            elif config.WAYPOINT_MODE == "ADD":
                if len(self.gps_points) <= config.MAX_WAYPOINTS:
                    wp_seq = self.get_current_mission_count()
                    return self.upload_waypoint(lat, lon, alt, wp_seq)
            elif config.WAYPOINT_MODE == "CLEAR_OLD":
                if time.time() - self.last_waypoint_time > config.WAYPOINT_CLEAR_TIMEOUT:
                    self.clear_mission()
                wp_seq = self.get_current_mission_count()
                return self.upload_waypoint(lat, lon, alt, wp_seq)
            
            return False
        except Exception as e:
            logger.error(f"Error uploading waypoint: {e}")
            return False
    
    def get_current_mission_count(self) -> int:
        """Get current mission count with timeout"""
        if not self.connected or not self.mavlink_connection:
            return 1
            
        try:
            # Clear pending messages
            while self.mavlink_connection.recv_match(blocking=False):
                pass
            
            self.mavlink_connection.mav.mission_request_list_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component
            )
            
            msg = self.mavlink_connection.recv_match(type='MISSION_COUNT', blocking=True, timeout=2)
            if msg:
                self.mission_count = msg.count
                return msg.count
            return 1
        except Exception as e:
            logger.error(f"Error getting mission count: {e}")
            return 1
    
    def upload_waypoint(self, lat: float, lon: float, alt: float, seq: int) -> bool:
        """Upload waypoint with improved error handling"""
        if not self.connected or not self.mavlink_connection:
            return False
            
        try:
            self.mavlink_connection.mav.mission_count_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                seq + 1, 0
            )
            
            start_time = time.time()
            items_sent = set()
            
            while time.time() - start_time < 5:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.5)
                
                if msg:
                    msg_type = msg.get_type()
                    
                    if msg_type in ['MISSION_REQUEST', 'MISSION_REQUEST_INT']:
                        requested_seq = msg.seq
                        
                        if requested_seq not in items_sent:
                            if requested_seq == 0:
                                self._send_home_position()
                            elif requested_seq == seq:
                                self._send_waypoint_item(lat, lon, alt, seq)
                            items_sent.add(requested_seq)
                            
                    elif msg_type == 'MISSION_ACK':
                        if msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                            return True
                        else:
                            logger.error(f"Mission rejected: {msg.type}")
                            return False
            
            logger.error("Timeout waiting for mission protocol")
            return False
            
        except Exception as e:
            logger.error(f"Error uploading waypoint: {e}")
            return False
    
    def _send_waypoint_item(self, lat: float, lon: float, alt: float, seq: int):
        """Send waypoint item with proper frame"""
        frame = (mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT 
                if config.USE_RELATIVE_ALTITUDE 
                else mavutil.mavlink.MAV_FRAME_GLOBAL_INT)
        
        self.mavlink_connection.mav.mission_item_int_send(
            self.mavlink_connection.target_system,
            self.mavlink_connection.target_component,
            seq, frame,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 1, 0, 5, 0, float('nan'),
            int(lat * 1e7), int(lon * 1e7), alt, 0
        )
    
    def _send_home_position(self):
        """Send home position waypoint"""
        if self.home_position['lat'] and self.home_position['lon']:
            lat, lon, alt = (self.home_position['lat'], 
                           self.home_position['lon'], 
                           self.home_position['alt'])
        else:
            with self._gps_lock:
                lat, lon, alt = (self._current_state['lat'], 
                               self._current_state['lon'], 
                               self._current_state['alt'])
        
        frame = (mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT 
                if config.USE_RELATIVE_ALTITUDE 
                else mavutil.mavlink.MAV_FRAME_GLOBAL_INT)
        
        self.mavlink_connection.mav.mission_item_int_send(
            self.mavlink_connection.target_system,
            self.mavlink_connection.target_component,
            0, frame,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            1, 1, 0, 0, 0, 0,
            int(lat * 1e7), int(lon * 1e7), 0, 0
        )
    
    def clear_mission(self) -> bool:
        """Clear all waypoints"""
        if not self.connected or not self.mavlink_connection:
            return False
            
        try:
            self.mavlink_connection.mav.mission_clear_all_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component
            )
            msg = self.mavlink_connection.recv_match(type='MISSION_ACK', blocking=True, timeout=2)
            if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                logger.info("Mission cleared")
                self.mission_count = 0
                return True
        except Exception as e:
            logger.error(f"Error clearing mission: {e}")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        with self._gps_lock:
            state = self._current_state.copy()
        
        return {
            'connected': self.connected,
            'gps_fix': state['gps_fix_type'],
            'satellites': state['satellites_visible'],
            'latitude': state['lat'],
            'longitude': state['lon'],
            'altitude': state['alt'],
            'altitude_info': asdict(self.altitude_info),
            'heading': state['heading'],
            'last_update': time.time() - state['last_update'],
            'points_logged': len(self.gps_points),
            'mission_count': self.mission_count
        }
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down MAVLink handler...")
        self.running = False
        self.connected = False
        
        if self.mavlink_thread and self.mavlink_thread.is_alive():
            self.mavlink_thread.join(timeout=2.0)
        
        if self.mavlink_connection:
            try:
                self.mavlink_connection.close()
            except:
                pass

# ==============================================================================
# DISTANCE CALCULATOR (FIXED)
# ==============================================================================

class DistanceCalculator:
    """Optimized distance calculation with improved algorithms"""
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Pre-calculate constants FIRST
        self._half_fov_h_rad = math.radians(config.CAMERA_FOV_HORIZONTAL / 2)
        self._half_fov_v_rad = math.radians(config.CAMERA_FOV_VERTICAL / 2)
        
        # Now calculate focal lengths
        self._update_focal_lengths()
        
        # Smoothing history
        self.distance_history = deque(maxlen=5)
    
    def _update_focal_lengths(self):
        """Update focal lengths when frame size changes"""
        self.focal_length_x = self.frame_width / (2 * math.tan(self._half_fov_h_rad))
        self.focal_length_y = self.frame_height / (2 * math.tan(self._half_fov_v_rad))
    
    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions and recalculate focal lengths"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self._update_focal_lengths()
    
    def calculate_distance_from_bbox(self, bbox) -> float:
        """Calculate distance using weighted average of height and width"""
        try:
            bbox_width_pixels = bbox.width() * self.frame_width
            bbox_height_pixels = bbox.height() * self.frame_height
            
            # Avoid division by zero
            if bbox_height_pixels <= 0 or bbox_width_pixels <= 0:
                return 0.0
            
            distance_from_height = (config.AVERAGE_PERSON_HEIGHT * self.focal_length_y) / bbox_height_pixels
            distance_from_width = (config.AVERAGE_PERSON_WIDTH * self.focal_length_x) / bbox_width_pixels
            
            # Weight height more heavily as it's typically more reliable
            distance = distance_from_height * 0.7 + distance_from_width * 0.3
            
            # Add to history for smoothing
            self.distance_history.append(distance)
            
            return self._get_smoothed_distance()
            
        except Exception as e:
            logger.error(f"Distance calculation error: {e}")
            return 0.0
    
    def calculate_3d_position(self, bbox, pan_angle: float, tilt_angle: float, distance: float) -> Position3D:
        """Calculate 3D position with improved trigonometry"""
        try:
            # Convert angles to radians for calculation
            pan_rad = math.radians(pan_angle - 90)
            actual_tilt_angle = tilt_angle + config.CAMERA_TILT_OFFSET
            tilt_rad = math.radians(90 - actual_tilt_angle)
            
            # Calculate horizontal distance
            horizontal_distance = distance * math.cos(tilt_rad)
            
            # Calculate 3D coordinates
            x = horizontal_distance * math.sin(pan_rad)
            y = horizontal_distance * math.cos(pan_rad)
            z = distance * math.sin(tilt_rad) + config.SERVO_MOUNT_HEIGHT
            
            return Position3D(x, y, z)
            
        except Exception as e:
            logger.error(f"3D position calculation error: {e}")
            return Position3D(0.0, 0.0, 0.0)
    
    def calculate_angular_size(self, bbox) -> Tuple[float, float]:
        """Calculate angular size of detection"""
        try:
            angular_width = bbox.width() * config.CAMERA_FOV_HORIZONTAL
            angular_height = bbox.height() * config.CAMERA_FOV_VERTICAL
            return angular_width, angular_height
        except Exception as e:
            logger.error(f"Angular size calculation error: {e}")
            return 0.0, 0.0
    
    def _get_smoothed_distance(self) -> float:
        """Get smoothed distance using median filtering"""
        if not self.distance_history:
            return 0.0
        
        try:
            # Use median of history for better noise rejection
            sorted_distances = sorted(self.distance_history)
            n = len(sorted_distances)
            
            if n >= 3:
                # Remove outliers and average the middle values
                filtered = sorted_distances[1:-1]
                return sum(filtered) / len(filtered)
            elif n == 2:
                return sum(sorted_distances) / 2
            else:
                return sorted_distances[0]
                
        except Exception as e:
            logger.error(f"Distance smoothing error: {e}")
            return 0.0

# ==============================================================================
# SERVO CONTROLLER
# ==============================================================================

class ServoController:
    """Thread-safe servo controller with improved performance"""
    
    def __init__(self, data_logger=None):
        self.data_logger = data_logger
        
        # Initialize hardware with error handling
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = adafruit_pca9685.PCA9685(self.i2c)
            self.pca.frequency = 50
            
            self.pan_servo = servo.Servo(self.pca.channels[0])
            self.tilt_servo = servo.Servo(self.pca.channels[1])
            
            logger.info("Servo hardware initialized successfully")
            
        except Exception as e:
            logger.error(f"Servo initialization failed: {e}")
            raise
        
        # State management with thread safety
        self._state_lock = threading.RLock()
        self._current_state = {
            'pan': 90.0,
            'tilt': 90.0 - config.CAMERA_TILT_OFFSET,
            'pan_velocity': 0.0,
            'tilt_velocity': 0.0,
            'last_update': time.time()
        }
        
        # Command processing
        self.command_queue = Queue(maxsize=10)
        self.running = True
        self.servo_thread = threading.Thread(target=self._servo_worker, daemon=True)
        
        # Initialize servo positions safely
        self._safe_move_servos(self._current_state['pan'], self._current_state['tilt'])
        
        # Start worker thread
        self.servo_thread.start()
        
        if self.data_logger:
            self.data_logger.log_event('servo_init', 'Servo controller initialized')
        
        logger.info("Servo controller ready")
    
    def _servo_worker(self):
        """Optimized servo worker with better error handling"""
        logger.info("Servo worker thread started")
        
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.05)
                if command is None:
                    break
                
                pan_angle, tilt_angle = command
                self._process_movement(pan_angle, tilt_angle)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Servo thread error: {e}")
                time.sleep(0.1)  # Prevent rapid error loops
        
        logger.info("Servo worker thread stopped")
    
    def _process_movement(self, pan_angle: float, tilt_angle: float):
        """Process servo movement with velocity calculation"""
        current_time = time.time()
        
        with self._state_lock:
            dt = current_time - self._current_state['last_update']
            
            if dt > 0:
                self._current_state['pan_velocity'] = (pan_angle - self._current_state['pan']) / dt
                self._current_state['tilt_velocity'] = (tilt_angle - self._current_state['tilt']) / dt
            
            # Only move if significant change
            if (abs(pan_angle - self._current_state['pan']) > 0.1 or 
                abs(tilt_angle - self._current_state['tilt']) > 0.1):
                
                if self._safe_move_servos(pan_angle, tilt_angle):
                    self._current_state['pan'] = pan_angle
                    self._current_state['tilt'] = tilt_angle
            
            self._current_state['last_update'] = current_time
    
    def _safe_move_servos(self, pan_angle: float, tilt_angle: float) -> bool:
        """Safely move servos with error handling and angle clamping"""
        try:
            # Clamp angles to safe ranges
            pan_angle = max(0.0, min(180.0, pan_angle))
            tilt_angle = max(0.0, min(180.0, tilt_angle))
            
            # Move servos
            self.pan_servo.angle = pan_angle
            self.tilt_servo.angle = tilt_angle
            
            # Small delay for servo response
            time.sleep(0.005)
            return True
            
        except Exception as e:
            logger.error(f"Servo movement error: {e}")
            return False
    
    def move_to(self, pan_angle: float, tilt_angle: float):
        """Queue movement command with overflow protection"""
        try:
            # Clear old commands if queue is getting full
            while self.command_queue.qsize() > 5:
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            
            self.command_queue.put_nowait((pan_angle, tilt_angle))
            
        except Exception as e:
            logger.debug(f"Command queue error: {e}")
    
    def get_current_state(self) -> Tuple[float, float, float, float]:
        """Thread-safe state access"""
        with self._state_lock:
            return (self._current_state['pan'], 
                   self._current_state['tilt'],
                   self._current_state['pan_velocity'], 
                   self._current_state['tilt_velocity'])
    
    def reset_position(self):
        """Reset servos to center position"""
        logger.info("Resetting servo positions to center")
        self.move_to(90.0, 90.0)
        time.sleep(1.0)  # Allow time for movement
    
    def shutdown(self):
        """Clean shutdown with servo reset"""
        logger.info("Shutting down servo controller...")
        
        if self.data_logger:
            self.data_logger.log_event('servo_shutdown', 'Servo controller shutting down')
        
        # Stop worker thread
        self.running = False
        try:
            self.command_queue.put_nowait(None)
        except:
            pass
        
        if self.servo_thread.is_alive():
            self.servo_thread.join(timeout=2.0)
        
        # Reset to center position
        try:
            self._safe_move_servos(90.0, 90.0)
            logger.info("Servos reset to center position")
        except Exception as e:
            logger.error(f"Error resetting servos: {e}")

# ==============================================================================
# TRACKER
# ==============================================================================

class PersonTracker:
    """Optimized person tracking with improved algorithms"""
    
    def __init__(self, servo_controller: ServoController, data_logger=None):
        self.servo = servo_controller
        self.data_logger = data_logger
        
        # Frame properties
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        # Distance calculator
        self.distance_calculator = DistanceCalculator(self.frame_width, self.frame_height)
        
        # Tracking state
        self.tracking_state = TrackingState()
        self.frame_skip_counter = 0
        
        # Smoothing history
        self.pan_history = deque(maxlen=config.DETECTION_HISTORY_SIZE)
        self.tilt_history = deque(maxlen=config.DETECTION_HISTORY_SIZE)
        
        # Pre-calculate weights for smoothing
        self._weights = [1.0, 2.0, 3.0]
        
        logger.info("Person tracker initialized")
    
    def update_frame_properties(self, width: int, height: int):
        """Update frame properties with change detection"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            self.distance_calculator.update_frame_size(width, height)
            
            if self.data_logger:
                self.data_logger.log_event('resolution_change', f'Frame: {width}x{height}')
            
            logger.info(f"Frame size updated: {width}x{height}")
    
    def track_person(self, bbox, confidence: float, frame_count: int):
        """Main person tracking function with improved algorithms"""
        try:
            # Frame skipping for performance
            self.frame_skip_counter += 1
            if self.frame_skip_counter < config.FRAME_SKIP_COUNT:
                return
            self.frame_skip_counter = 0
            
            # Calculate distance and 3D position
            self.tracking_state.current_distance = self.distance_calculator.calculate_distance_from_bbox(bbox)
            
            # Get current servo state
            current_pan, current_tilt, pan_vel, tilt_vel = self.servo.get_current_state()
            
            # Calculate 3D position
            self.tracking_state.current_position = self.distance_calculator.calculate_3d_position(
                bbox, current_pan, current_tilt, self.tracking_state.current_distance
            )
            
            # Calculate angular size for logging
            angular_width, angular_height = self.distance_calculator.calculate_angular_size(bbox)
            
            # Calculate target center in frame coordinates
            center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
            center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
            
            # Calculate tracking errors
            error_x = center_x - self.frame_center_x
            error_y = center_y - self.frame_center_y
            
            # Dynamic dead zone based on distance
            dynamic_dead_zone = config.DEAD_ZONE * (1 + self.tracking_state.current_distance / 10.0)
            
            # Only move if outside dead zone
            if abs(error_x) > dynamic_dead_zone or abs(error_y) > dynamic_dead_zone:
                # Calculate servo adjustments
                target_pan, target_tilt = self._calculate_servo_adjustments(
                    error_x, error_y, current_pan, current_tilt, confidence
                )
                
                # Apply smoothing
                smoothed_pan, smoothed_tilt = self._apply_smoothing(target_pan, target_tilt)
                
                # Send movement command
                self.servo.move_to(smoothed_pan, smoothed_tilt)
                
                # Update lock status
                if not self.tracking_state.target_locked:
                    self.tracking_state.target_locked = True
                    if self.data_logger:
                        self.data_logger.log_event('target_lock', 
                                                 f'Target locked at {self.tracking_state.current_distance:.2f}m')
                    logger.info(f"Target locked at {self.tracking_state.current_distance:.2f}m")
            
            # Update tracking state
            self.tracking_state.last_detection_time = time.time()
            self.tracking_state.lost_frames = 0
            self.tracking_state.is_active = True
            
            # Log data
            if self.data_logger:
                self._log_tracking_data(
                    frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                    confidence, angular_width, angular_height, bbox
                )
                
        except Exception as e:
            logger.error(f"Tracking error: {e}")
    
    def _calculate_servo_adjustments(self, error_x: float, error_y: float, 
                                   current_pan: float, current_tilt: float, 
                                   confidence: float) -> Tuple[float, float]:
        """Calculate servo adjustments with distance and confidence weighting"""
        try:
            # Distance-based adjustment factor
            distance_factor = min(2.0, max(0.5, 2.0 / max(self.tracking_state.current_distance, 0.1)))
            
            # Base adjustments
            pan_adjustment = -error_x * (config.PAN_SENSITIVITY / self.frame_width) * distance_factor
            tilt_adjustment = error_y * (config.TILT_SENSITIVITY / self.frame_height) * distance_factor
            
            # Confidence multiplier
            confidence_multiplier = min(2.0, confidence + 0.5)
            pan_adjustment *= confidence_multiplier
            tilt_adjustment *= confidence_multiplier
            
            # Calculate target angles
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            return target_pan, target_tilt
            
        except Exception as e:
            logger.error(f"Servo adjustment calculation error: {e}")
            return current_pan, current_tilt
    
    def _apply_smoothing(self, target_pan: float, target_tilt: float) -> Tuple[float, float]:
        """Apply smoothing with weighted history"""
        try:
            current_pan, current_tilt, _, _ = self.servo.get_current_state()
            
            # Apply smoothing factor
            new_pan = self._smooth_angle(current_pan, target_pan)
            new_tilt = self._smooth_angle(current_tilt, target_tilt)
            
            # Add to history
            self.pan_history.append(new_pan)
            self.tilt_history.append(new_tilt)
            
            # Calculate weighted average if we have enough history
            if len(self.pan_history) >= 2:
                weights = self._weights[:len(self.pan_history)]
                weight_sum = sum(weights)
                
                avg_pan = sum(w * angle for w, angle in zip(weights, self.pan_history)) / weight_sum
                avg_tilt = sum(w * angle for w, angle in zip(weights, self.tilt_history)) / weight_sum
                
                return avg_pan, avg_tilt
            
            return new_pan, new_tilt
            
        except Exception as e:
            logger.error(f"Smoothing error: {e}")
            current_pan, current_tilt, _, _ = self.servo.get_current_state()
            return current_pan, current_tilt
    
    def _smooth_angle(self, current: float, target: float) -> float:
        """Apply smoothing to angle changes"""
        try:
            diff = (target - current) * config.SMOOTHING_FACTOR
            diff = max(-config.MAX_STEP_SIZE, min(config.MAX_STEP_SIZE, diff))
            return current + diff
        except Exception as e:
            logger.error(f"Angle smoothing error: {e}")
            return current
    
    def _log_tracking_data(self, frame_count: int, current_pan: float, current_tilt: float,
                          pan_vel: float, tilt_vel: float, confidence: float,
                          angular_width: float, angular_height: float, bbox):
        """Log comprehensive tracking data"""
        try:
            distance_data = {
                'distance': self.tracking_state.current_distance,
                'x_position': self.tracking_state.current_position.x,
                'y_position': self.tracking_state.current_position.y,
                'z_position': self.tracking_state.current_position.z,
                'angular_width': angular_width,
                'angular_height': angular_height,
                'bbox_width': bbox.width(),
                'bbox_height': bbox.height()
            }
            
            self.data_logger.log_frame_data(
                frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                confidence, True, self.is_tracking_active(), self.tracking_state.lost_frames,
                distance_data
            )
            
        except Exception as e:
            logger.error(f"Tracking data logging error: {e}")
    
    def handle_lost_target(self, frame_count: int):
        """Handle lost target with improved logic"""
        try:
            self.tracking_state.lost_frames += 1
            
            # Switch to scanning mode after losing target
            if self.tracking_state.lost_frames > 10 and self.tracking_state.target_locked:
                self.tracking_state.target_locked = False
                self.tracking_state.is_active = False
                
                if self.data_logger:
                    self.data_logger.log_event('target_lost', 'Target lost - scanning mode')
                logger.info("Target lost - scanning mode")
            
            # Log lost target data
            if self.data_logger:
                current_pan, current_tilt, pan_vel, tilt_vel = self.servo.get_current_state()
                self.data_logger.log_frame_data(
                    frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                    0.0, False, self.is_tracking_active(), self.tracking_state.lost_frames, None
                )
                
        except Exception as e:
            logger.error(f"Lost target handling error: {e}")
    
    def is_tracking_active(self) -> bool:
        """Check if tracking is currently active"""
        return (time.time() - self.tracking_state.last_detection_time) < config.DETECTION_TIMEOUT
    
    def get_current_distance_info(self) -> Dict[str, Any]:
        """Get current distance and position information"""
        return {
            'distance': self.tracking_state.current_distance,
            'position_3d': self.tracking_state.current_position,
            'is_active': self.is_tracking_active(),
            'target_locked': self.tracking_state.target_locked
        }
    
    def get_tracking_state(self) -> TrackingState:
        """Get current tracking state"""
        return self.tracking_state

# ==============================================================================
# DATA LOGGER
# ==============================================================================

class DataLogger:
    """Comprehensive data logging with GPS integration"""
    
    def __init__(self, log_dir: str = "servo_logs", gps_handler: MAVLinkGPSHandler = None):
        self.gps_handler = gps_handler
        
        # Setup log directory
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"session_{timestamp}.json"
        self.gps_csv_file = self.log_dir / f"gps_points_{timestamp}.csv"
        
        # Initialize files
        self._initialize_csv_files()
        self.session_data = self._initialize_session_data()
        
        logger.info(f"Data logging initialized: {self.log_dir}")
        logger.info(f"Altitude mode: {'relative' if config.USE_RELATIVE_ALTITUDE else 'MSL'}")
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        try:
            # Main data CSV headers
            csv_headers = [
                'timestamp', 'frame_count', 'pan_angle', 'tilt_angle',
                'pan_velocity', 'tilt_velocity', 'detection_confidence',
                'person_detected', 'tracking_active', 'target_lost_frames',
                'distance_meters', 'x_position', 'y_position', 'z_position',
                'angular_width', 'angular_height', 'bbox_width', 'bbox_height',
                'gps_latitude', 'gps_longitude', 'gps_altitude', 'gps_altitude_type'
            ]
            
            with open(self.csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(csv_headers)
            
            # GPS CSV headers
            gps_headers = [
                'timestamp', 'detection_lat', 'detection_lon', 'detection_alt', 'altitude_type',
                'vehicle_lat', 'vehicle_lon', 'vehicle_alt_msl', 'vehicle_alt_relative', 
                'vehicle_alt_agl', 'vehicle_heading', 'relative_x', 'relative_y', 
                'relative_z', 'distance_2d', 'distance_3d', 'confidence'
            ]
            
            with open(self.gps_csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(gps_headers)
                
        except Exception as e:
            logger.error(f"Error initializing CSV files: {e}")
            raise
    
    def _initialize_session_data(self) -> Dict[str, Any]:
        """Initialize session data structure"""
        return {
            'start_time': datetime.now().isoformat(),
            'configuration': asdict(config),
            'log_files': {
                'csv': str(self.csv_file),
                'json': str(self.json_file),
                'gps_csv': str(self.gps_csv_file)
            },
            'statistics': {
                'total_detections': 0,
                'total_movements': 0,
                'min_distance': float('inf'),
                'max_distance': 0.0,
                'avg_distance': 0.0,
                'distance_samples': 0,
                'gps_points_created': 0,
                'altitude_range': {'min': float('inf'), 'max': 0.0}
            },
            'events': []
        }
    
    def log_frame_data(self, frame_count: int, pan_angle: float, tilt_angle: float, 
                      pan_velocity: float, tilt_velocity: float, detection_confidence: float,
                      person_detected: bool, tracking_active: bool, target_lost_frames: int,
                      distance_data: Optional[Dict[str, Any]] = None):
        """Log frame data with GPS integration"""
        try:
            # Initialize default values
            distance = x_pos = y_pos = z_pos = 0.0
            angular_width = angular_height = bbox_width = bbox_height = 0.0
            gps_lat = gps_lon = gps_alt = 0.0
            gps_alt_type = ''
            
            # Process distance data and GPS waypoints
            if distance_data:
                distance = distance_data.get('distance', 0.0)
                x_pos = distance_data.get('x_position', 0.0)
                y_pos = distance_data.get('y_position', 0.0)
                z_pos = distance_data.get('z_position', 0.0)
                angular_width = distance_data.get('angular_width', 0.0)
                angular_height = distance_data.get('angular_height', 0.0)
                bbox_width = distance_data.get('bbox_width', 0.0)
                bbox_height = distance_data.get('bbox_height', 0.0)
                
                # Handle GPS waypoint creation
                gps_point = self._handle_gps_waypoint(x_pos, y_pos, z_pos, detection_confidence)
                if gps_point:
                    gps_lat = gps_point.latitude
                    gps_lon = gps_point.longitude
                    gps_alt = gps_point.altitude
                    gps_alt_type = gps_point.altitude_type
                
                # Update distance statistics
                self._update_distance_statistics(distance)
            
            # Write main CSV data
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), frame_count, pan_angle, tilt_angle,
                    pan_velocity, tilt_velocity, detection_confidence,
                    person_detected, tracking_active, target_lost_frames,
                    distance, x_pos, y_pos, z_pos,
                    angular_width, angular_height, bbox_width, bbox_height,
                    gps_lat, gps_lon, gps_alt, gps_alt_type
                ])
            
            # Update session statistics
            self._update_session_statistics(person_detected, pan_velocity, tilt_velocity)
            
        except Exception as e:
            logger.error(f"Frame data logging error: {e}")
    
    def _handle_gps_waypoint(self, x_pos: float, y_pos: float, z_pos: float, 
                           confidence: float) -> Optional[GPSPoint]:
        """Handle GPS waypoint creation and logging"""
        if not self.gps_handler or not self.gps_handler.connected:
            return None
        
        try:
            distance_2d = math.sqrt(x_pos**2 + y_pos**2)
            
            if distance_2d >= config.MIN_DISTANCE_FOR_GPS:
                gps_point = self.gps_handler.add_detection_point(x_pos, y_pos, z_pos, confidence)
                
                if gps_point:
                    self.session_data['statistics']['gps_points_created'] += 1
                    
                    # Track altitude range
                    alt = gps_point.altitude
                    stats = self.session_data['statistics']['altitude_range']
                    stats['min'] = min(stats['min'], alt)
                    stats['max'] = max(stats['max'], alt)
                    
                    # Write GPS CSV data
                    self._write_gps_csv_data(gps_point)
                    
                return gps_point
                
        except Exception as e:
            logger.error(f"GPS waypoint handling error: {e}")
        
        return None
    
    def _write_gps_csv_data(self, gps_point: GPSPoint):
        """Write GPS CSV data"""
        try:
            with open(self.gps_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                vehicle_pos = gps_point.vehicle_position
                writer.writerow([
                    gps_point.timestamp, gps_point.latitude, gps_point.longitude,
                    gps_point.altitude, gps_point.altitude_type,
                    vehicle_pos['lat'], vehicle_pos['lon'], vehicle_pos['alt_msl'],
                    vehicle_pos['alt_relative'], vehicle_pos.get('alt_agl'),
                    vehicle_pos['heading'], gps_point.relative_position.x,
                    gps_point.relative_position.y, gps_point.relative_position.z,
                    gps_point.distance_2d, gps_point.distance_3d, gps_point.confidence
                ])
        except Exception as e:
            logger.error(f"GPS CSV writing error: {e}")
    
    def _update_distance_statistics(self, distance: float):
        """Update distance statistics"""
        if distance > 0:
            try:
                stats = self.session_data['statistics']
                stats['min_distance'] = min(stats['min_distance'], distance)
                stats['max_distance'] = max(stats['max_distance'], distance)
                stats['distance_samples'] += 1
                
                # Update running average
                current_avg = stats['avg_distance']
                samples = stats['distance_samples']
                stats['avg_distance'] = (current_avg * (samples - 1) + distance) / samples
                
            except Exception as e:
                logger.error(f"Distance statistics update error: {e}")
    
    def _update_session_statistics(self, person_detected: bool, pan_velocity: float, tilt_velocity: float):
        """Update session statistics"""
        try:
            if person_detected:
                self.session_data['statistics']['total_detections'] += 1
            
            if abs(pan_velocity) > 1 or abs(tilt_velocity) > 1:
                self.session_data['statistics']['total_movements'] += 1
                
        except Exception as e:
            logger.error(f"Session statistics update error: {e}")
    
    def log_event(self, event_type: str, description: str):
        """Log system events"""
        try:
            event = {
                'timestamp': time.time(),
                'iso_time': datetime.now().isoformat(),
                'type': event_type,
                'description': description
            }
            self.session_data['events'].append(event)
            logger.info(f"{event_type}: {description}")
            
        except Exception as e:
            logger.error(f"Event logging error: {e}")
    
    def finalize_session(self):
        """Finalize and save session data"""
        try:
            self.session_data['end_time'] = datetime.now().isoformat()
            
            if self.gps_handler:
                self.session_data['gps_status'] = self.gps_handler.get_status()
            
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
            
            self._print_session_summary()
            
        except Exception as e:
            logger.error(f"Session finalization error: {e}")
    
    def _print_session_summary(self):
        """Print session summary"""
        try:
            stats = self.session_data['statistics']
            
            print(f"\n📊 Session Summary:")
            print(f"   Total Detections: {stats['total_detections']}")
            print(f"   GPS Points Created: {stats['gps_points_created']}")
            print(f"   Total Movements: {stats['total_movements']}")
            
            if stats['distance_samples'] > 0:
                print(f"   Distance Range: {stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m")
                print(f"   Average Distance: {stats['avg_distance']:.2f}m")
            
            if stats['gps_points_created'] > 0:
                alt_range = stats['altitude_range']
                alt_type = 'REL' if config.USE_RELATIVE_ALTITUDE else 'MSL'
                print(f"   Altitude Range ({alt_type}): {alt_range['min']:.1f}m - {alt_range['max']:.1f}m")
            
            print(f"   Log files saved to: {self.log_dir}")
            
        except Exception as e:
            logger.error(f"Session summary error: {e}")

# ==============================================================================
# CALIBRATION HELPER
# ==============================================================================

class CalibrationHelper:
    """Camera calibration helper for focal length calculation"""
    
    def __init__(self):
        self.measurements = []
        self.calibrating = config.CALIBRATION_MODE
        self.min_measurements = 15
        self.min_pixel_height = 50
        
        if self.calibrating:
            logger.info("Calibration mode enabled - stand 2m from camera")
    
    def add_measurement(self, bbox, frame_height: int):
        """Add calibration measurement with validation"""
        if not self.calibrating:
            return
        
        try:
            person_height_pixels = bbox.height() * frame_height
            
            if person_height_pixels > self.min_pixel_height:
                self.measurements.append(person_height_pixels)
                
                if len(self.measurements) % 5 == 0:
                    logger.info(f"Calibration progress: {len(self.measurements)}/{self.min_measurements}")
                
                if len(self.measurements) >= self.min_measurements:
                    self._complete_calibration()
                    
        except Exception as e:
            logger.error(f"Calibration measurement error: {e}")
    
    def _complete_calibration(self):
        """Complete calibration process with statistical analysis"""
        try:
            # Remove outliers and calculate average
            sorted_measurements = sorted(self.measurements)
            n = len(sorted_measurements)
            
            # Remove top and bottom 15% as outliers
            start_idx = max(0, int(n * 0.15))
            end_idx = min(n, int(n * 0.85))
            filtered_measurements = sorted_measurements[start_idx:end_idx]
            
            avg_height = sum(filtered_measurements) / len(filtered_measurements)
            std_dev = math.sqrt(sum((x - avg_height) ** 2 for x in filtered_measurements) / len(filtered_measurements))
            
            focal_length = (avg_height * config.CALIBRATION_DISTANCE) / config.AVERAGE_PERSON_HEIGHT
            
            print(f"\n📸 CALIBRATION COMPLETE!")
            print(f"   Measurements taken: {len(self.measurements)}")
            print(f"   Valid measurements: {len(filtered_measurements)}")
            print(f"   Average person height: {avg_height:.1f} ± {std_dev:.1f} pixels")
            print(f"   Calculated focal length: {focal_length:.1f} pixels")
            print(f"   Current focal length: {config.FOCAL_LENGTH_PIXELS:.1f} pixels")
            print(f"   Recommendation: Update FOCAL_LENGTH_PIXELS = {focal_length:.1f}")
            
            # Calculate accuracy estimate
            accuracy = abs(focal_length - config.FOCAL_LENGTH_PIXELS) / config.FOCAL_LENGTH_PIXELS * 100
            print(f"   Accuracy difference: {accuracy:.1f}%")
            
            self.calibrating = False
            self.measurements = []
            
            logger.info("Calibration completed successfully")
            
        except Exception as e:
            logger.error(f"Calibration completion error: {e}")

# ==============================================================================
# CALLBACK SYSTEM
# ==============================================================================

class AppCallback(app_callback_class):
    """Enhanced callback class with FPS monitoring"""
    
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0
        self.fps_update_interval = 1.0
    
    def new_function(self):
        return "Optimized Ultra-Fast Tracking with Real Altitude GPS Waypoint Guidance: "
    
    def increment(self):
        """Increment with FPS calculation"""
        super().increment()
        self.fps_counter += 1
        
        current_time = time.time()
        if current_time - self.last_fps_time >= self.fps_update_interval:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time

# ==============================================================================
# MAIN CALLBACK FUNCTION
# ==============================================================================

def main_app_callback(pad, info, user_data):
    """Main application callback with comprehensive error handling"""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    try:
        user_data.increment()
        frame_count = user_data.get_count()
        
        # Update frame properties periodically
        if frame_count % 30 == 0:
            format, width, height = get_caps_from_pad(pad)
            if width and height:
                tracker.update_frame_properties(width, height)
        
        # Get frame for display if enabled
        frame = None
        if user_data.use_frame:
            format, width, height = get_caps_from_pad(pad)
            if format and width and height:
                frame = get_numpy_from_buffer(buffer, format, width, height)
        
        # Get detections
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        # Find best person detection
        best_person = _find_best_person_detection(detections)
        
        # Process tracking
        if best_person:
            tracker.track_person(best_person['bbox'], best_person['confidence'], frame_count)
            
            # Handle calibration
            if calibration_helper.calibrating:
                calibration_helper.add_measurement(best_person['bbox'], tracker.frame_height)
            
            # Periodic status updates
            if frame_count % 60 == 0:
                distance_info = tracker.get_current_distance_info()
                logger.info(f"Tracking: Conf {best_person['confidence']:.2f}, "
                           f"Distance: {distance_info['distance']:.2f}m, "
                           f"FPS: {user_data.current_fps:.1f}")
        else:
            tracker.handle_lost_target(frame_count)
        
        # Update frame display
        if user_data.use_frame and frame is not None:
            _update_frame_display(frame, tracker, best_person, mavlink_handler, user_data)
        
        return Gst.PadProbeReturn.OK
        
    except Exception as e:
        logger.error(f"Callback error: {e}")
        return Gst.PadProbeReturn.OK

def _find_best_person_detection(detections):
    """Find the best person detection based on confidence and size"""
    best_person = None
    best_score = 0
    
    try:
        for detection in detections:
            if detection.get_label() == "person":
                confidence = detection.get_confidence()
                if confidence >= config.MIN_CONFIDENCE:
                    bbox = detection.get_bbox()
                    area = bbox.width() * bbox.height()
                    
                    # Score combines confidence and size (larger targets preferred)
                    score = confidence * 0.7 + min(area * 10, 1.0) * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_person = {'bbox': bbox, 'confidence': confidence}
    
    except Exception as e:
        logger.error(f"Detection processing error: {e}")
    
    return best_person

def _update_frame_display(frame, tracker: PersonTracker, best_person, 
                         mavlink_handler: MAVLinkGPSHandler, user_data):
    """Update frame display with tracking information"""
    try:
        center_x, center_y = tracker.frame_center_x, tracker.frame_center_y
        
        # Draw crosshairs
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 1)
        
        # Display tracking information
        if best_person:
            distance_info = tracker.get_current_distance_info()
            distance = distance_info['distance']
            pos_3d = distance_info['position_3d']
            
            # Status text
            if calibration_helper.calibrating:
                progress = f"{len(calibration_helper.measurements)}/{calibration_helper.min_measurements}"
                cv2.putText(frame, f"CALIBRATION: {progress}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                status_color = (0, 255, 0) if distance_info['target_locked'] else (255, 255, 0)
                cv2.putText(frame, f"TRACKING: {distance:.2f}m", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # 3D position
            cv2.putText(frame, f"3D: ({pos_3d.x:.1f}, {pos_3d.y:.1f}, {pos_3d.z:.1f})m", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Bounding box
            bbox = best_person['bbox']
            x1 = int(bbox.xmin() * frame.shape[1])
            y1 = int(bbox.ymin() * frame.shape[0])
            x2 = int((bbox.xmin() + bbox.width()) * frame.shape[1])
            y2 = int((bbox.ymin() + bbox.height()) * frame.shape[0])
            
            box_color = (0, 255, 0) if distance_info['target_locked'] else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{distance:.1f}m", 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # Servo information
        pan, tilt, pan_vel, tilt_vel = servo_controller.get_current_state()
        cv2.putText(frame, f"Pan: {pan:.1f}° ({pan_vel:.1f}°/s)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"Tilt: {tilt:.1f}° ({tilt_vel:.1f}°/s)", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # FPS display
        cv2.putText(frame, f"FPS: {user_data.current_fps:.1f}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # GPS and altitude information
        if mavlink_handler and mavlink_handler.connected:
            _display_gps_info(frame, mavlink_handler)
        
        # Convert color space for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
        
    except Exception as e:
        logger.error(f"Frame display error: {e}")

def _display_gps_info(frame, mavlink_handler: MAVLinkGPSHandler):
    """Display GPS and altitude information on frame"""
    try:
        status = mavlink_handler.get_status()
        alt_info = status['altitude_info']
        
        # Altitude display
        alt_text = f"Alt: {status['altitude']:.1f}m"
        if config.USE_RELATIVE_ALTITUDE:
            alt_text += " (REL)"
        else:
            alt_text += " (MSL)"
        
        # GPS status
        gps_text = f"GPS: {status['satellites']} sats"
        
        # Color based on GPS fix quality
        if status['gps_fix'] >= 3:
            color = (0, 255, 0)  # Green for good fix
        else:
            color = (0, 0, 255)  # Red for poor fix
        
        cv2.putText(frame, gps_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, alt_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Rangefinder data if available
        if alt_info['rangefinder_available']:
            agl_text = f"AGL: {alt_info['alt_agl']:.1f}m"
            cv2.putText(frame, agl_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Waypoint count
        if status['mission_count'] > 0:
            wp_text = f"WP: {status['mission_count']}"
            cv2.putText(frame, wp_text, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Connection status
        cv2.putText(frame, "MAVLINK OK", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
    except Exception as e:
        logger.error(f"GPS display error: {e}")

# ==============================================================================
# SYSTEM MANAGEMENT
# ==============================================================================

# Global system components
mavlink_handler: Optional[MAVLinkGPSHandler] = None
data_logger: Optional[DataLogger] = None
servo_controller: Optional[ServoController] = None
tracker: Optional[PersonTracker] = None
calibration_helper: Optional[CalibrationHelper] = None

def initialize_system():
    """Initialize all system components with proper error handling"""
    global mavlink_handler, data_logger, servo_controller, tracker, calibration_helper
    
    logger.info("="*50)
    logger.info("Initializing Optimized Servo Tracking System")
    logger.info("="*50)
    logger.info(f"Altitude mode: {'RELATIVE to home' if config.USE_RELATIVE_ALTITUDE else 'Mean Sea Level (MSL)'}")
    logger.info(f"Target height above ground: {config.TARGET_HEIGHT_ABOVE_GROUND}m")
    logger.info(f"Altitude limits: {config.MIN_ALTITUDE_AGL}m - {config.MAX_ALTITUDE_AGL}m AGL")
    
    # Initialize MAVLink connection
    try:
        logger.info("Initializing MAVLink GPS handler...")
        mavlink_handler = MAVLinkGPSHandler()
        if mavlink_handler.connected:
            logger.info("✅ MAVLink GPS handler initialized successfully")
        else:
            logger.warning("⚠️  MAVLink initialization failed - continuing without GPS")
    except Exception as e:
        logger.warning(f"⚠️  MAVLink initialization error: {e}")
        logger.warning("Continuing without GPS waypoint functionality")
        mavlink_handler = None
    
    # Initialize data logger
    try:
        logger.info("Initializing data logger...")
        data_logger = DataLogger(gps_handler=mavlink_handler)
        logger.info("✅ Data logger initialized successfully")
    except Exception as e:
        logger.error(f"❌ Data logger initialization failed: {e}")
        raise
    
    # Initialize servo controller
    try:
        logger.info("Initializing servo controller...")
        servo_controller = ServoController(data_logger)
        logger.info("✅ Servo controller initialized successfully")
    except Exception as e:
        logger.error(f"❌ Servo controller initialization failed: {e}")
        raise
    
    # Initialize tracker
    try:
        logger.info("Initializing person tracker...")
        tracker = PersonTracker(servo_controller, data_logger)
        logger.info("✅ Person tracker initialized successfully")
    except Exception as e:
        logger.error(f"❌ Tracker initialization failed: {e}")
        raise
    
    # Initialize calibration helper
    try:
        logger.info("Initializing calibration helper...")
        calibration_helper = CalibrationHelper()
        logger.info("✅ Calibration helper initialized successfully")
    except Exception as e:
        logger.error(f"❌ Calibration helper initialization failed: {e}")
        raise
    
    logger.info("✅ All system components initialized successfully")
    return True

def print_system_status():
    """Print comprehensive system status"""
    logger.info("System Status Check:")
    
    if mavlink_handler and mavlink_handler.connected:
        # Wait for initial data
        logger.info("Waiting for GPS data...")
        time.sleep(3)
        
        status = mavlink_handler.get_status()
        alt_info = status['altitude_info']
        
        print(f"\n🛰️ GPS Status:")
        print(f"   Fix Type: {status['gps_fix']} (3=3D fix required)")
        print(f"   Satellites: {status['satellites']}")
        print(f"   Position: {status['latitude']:.6f}, {status['longitude']:.6f}")
        print(f"   Last Update: {status['last_update']:.1f}s ago")
        
        print(f"\n🏔️  Altitude Information:")
        print(f"   Primary: {status['altitude']:.1f}m ({'REL' if config.USE_RELATIVE_ALTITUDE else 'MSL'})")
        print(f"   MSL: {alt_info['alt_msl']:.1f}m")
        print(f"   Relative: {alt_info['alt_relative']:.1f}m")
        
        if alt_info['rangefinder_available']:
            print(f"   AGL: {alt_info['alt_agl']:.1f}m (rangefinder)")
        else:
            print("   AGL: Not available (no rangefinder)")
            
        print(f"   Waypoint Mode: {config.WAYPOINT_MODE}")
        print(f"   Mission Count: {status['mission_count']}")
    else:
        print(f"\n🛰️ GPS Status: ❌ Not connected")
    
    print(f"\n📊 Data Output: {data_logger.log_dir}")
    print(f"\n⚙️  Configuration:")
    print(f"   Calibration Mode: {'✅ ON' if config.CALIBRATION_MODE else '❌ OFF'}")
    print(f"   Frame Skip: {config.FRAME_SKIP_COUNT}")
    print(f"   Min Confidence: {config.MIN_CONFIDENCE}")
    print(f"   Dead Zone: {config.DEAD_ZONE} pixels")
    
    print(f"\n🎮 Controls:")
    print("   Press Ctrl+C to stop system")
    if config.CALIBRATION_MODE:
        print("   📸 CALIBRATION MODE: Stand exactly 2m from camera")
        print("   📸 Move around to gather measurements")

def cleanup_system():
    """Clean shutdown of all system components"""
    logger.info("="*50)
    logger.info("Shutting Down System")
    logger.info("="*50)
    
    components = [
        ('data_logger', data_logger, 'finalize_session'),
        ('servo_controller', servo_controller, 'shutdown'),
        ('mavlink_handler', mavlink_handler, 'shutdown')
    ]
    
    for name, component, method in components:
        if component:
            try:
                logger.info(f"Shutting down {name}...")
                getattr(component, method)()
                logger.info(f"✅ {name} shutdown complete")
            except Exception as e:
                logger.error(f"❌ Error shutting down {name}: {e}")
    
    logger.info("✅ System shutdown complete")

def signal_handler(signum, frame):
    """Handle system signals for clean shutdown"""
    logger.info(f"Received signal {signum}")
    cleanup_system()
    sys.exit(0)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main application entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_system)
    
    # Set up environment
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)
    
    try:
        # Initialize system
        logger.info("Starting system initialization...")
        initialize_system()
        print_system_status()
        
        # Create and configure the application
        user_data = AppCallback()
        app = GStreamerDetectionApp(main_app_callback, user_data)
        
        logger.info("🚀 Starting optimized tracking system...")
        print("\n" + "="*50)
        print("🚀 SYSTEM RUNNING - Tracking Active")
        print("="*50)
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        if data_logger:
            data_logger.log_event('error', f'Application error: {str(e)}')
    finally:
        cleanup_system()

if __name__ == "__main__":
    main()