#!/usr/bin/env python3
"""
Servo Tracking System with GPS Waypoint Generation
For Raspberry Pi 5 with CubeOrange and Hailo AI
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
FOCAL_LENGTH_PIXELS = 382

# Physical setup
SERVO_MOUNT_HEIGHT = 1.3
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

# Calibration
CALIBRATION_MODE = True
CALIBRATION_DISTANCE = 2.0

# ==============================================================================
# MAVLINK GPS HANDLER
# ==============================================================================

class MAVLinkGPSHandler:
    def __init__(self, connection_string=MAVLINK_CONNECTION, baud=MAVLINK_BAUD):
        self.connection_string = connection_string
        self.baud = baud
        self.mavlink_connection = None
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
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
            self.request_data_streams()
            self.mavlink_thread = threading.Thread(target=self._mavlink_receiver, daemon=True)
            self.mavlink_thread.start()
            return True
        except Exception as e:
            print(f"❌ MAVLink connection failed: {e}")
            return False
    
    def request_data_streams(self):
        try:
            self.mavlink_connection.mav.request_data_stream_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10, 1
            )
        except Exception as e:
            print(f"Error requesting data streams: {e}")
    
    def _mavlink_receiver(self):
        while self.running:
            try:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=0.1)
                if msg is None:
                    continue
                
                msg_type = msg.get_type()
                
                if msg_type == 'GPS_RAW_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    self.gps_fix_type = msg.fix_type
                    self.satellites_visible = msg.satellites_visible
                    self.last_gps_update = time.time()
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    self.current_heading = msg.hdg / 100.0
                elif msg_type == 'ATTITUDE':
                    yaw_rad = msg.yaw
                    self.current_heading = math.degrees(yaw_rad) % 360
                elif msg_type == 'HOME_POSITION':
                    self.home_lat = msg.latitude / 1e7
                    self.home_lon = msg.longitude / 1e7
                    self.home_alt = msg.altitude / 1000.0
                elif msg_type == 'MISSION_CURRENT':
                    self.current_wp_seq = msg.seq
                elif msg_type == 'MISSION_ITEM_REACHED':
                    print(f"✅ Reached waypoint {msg.seq}")
            except Exception as e:
                if self.running:
                    print(f"MAVLink receive error: {e}")
    
    def calculate_gps_position(self, x_meters, y_meters):
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
        current_time = time.time()
        if current_time - self.last_point_time < GPS_UPDATE_INTERVAL:
            return None
        
        distance = math.sqrt(x_meters**2 + y_meters**2)
        if distance < MIN_DISTANCE_FOR_GPS:
            return None
        
        lat, lon = self.calculate_gps_position(x_meters, y_meters)
        if lat is None or lon is None:
            return None
        
        waypoint_alt = self.current_alt + WAYPOINT_ALTITUDE_OFFSET
        
        gps_point = {
            'timestamp': current_time,
            'latitude': lat,
            'longitude': lon,
            'altitude': waypoint_alt,
            'relative_x': x_meters,
            'relative_y': y_meters,
            'relative_z': z_meters,
            'distance': distance,
            'confidence': confidence,
            'vehicle_lat': self.current_lat,
            'vehicle_lon': self.current_lon,
            'vehicle_heading': self.current_heading
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
            print(f"🎯 New waypoint sent: {distance:.1f}m away at bearing {bearing:.0f}°")
            self.notify_mission_changed()
        
        return gps_point
    
    def get_current_mission_count(self):
        try:
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
            else:
                return 1
        except Exception as e:
            print(f"Error getting mission count: {e}")
            return 1
    
    def upload_waypoint(self, lat, lon, alt, seq):
        try:
            print(f"📤 Uploading waypoint {seq}...")
            
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
        if self.home_lat and self.home_lon:
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
    
    def notify_mission_changed(self):
        try:
            self.mavlink_connection.mav.mission_current_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                self.current_wp_seq
            )
        except Exception as e:
            print(f"Error notifying mission change: {e}")
    
    def clear_mission(self):
        try:
            self.mavlink_connection.mav.mission_clear_all_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component
            )
            msg = self.mavlink_connection.recv_match(type='MISSION_ACK', blocking=True, timeout=2)
            if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                print("🗑️  Mission cleared")
                self.mission_count = 0
                return True
        except Exception as e:
            print(f"Error clearing mission: {e}")
        return False
    
    def get_status(self):
        return {
            'connected': self.mavlink_connection is not None,
            'gps_fix': self.gps_fix_type,
            'satellites': self.satellites_visible,
            'latitude': self.current_lat,
            'longitude': self.current_lon,
            'altitude': self.current_alt,
            'heading': self.current_heading,
            'last_update': time.time() - self.last_gps_update,
            'points_logged': len(self.gps_points),
            'mission_count': self.mission_count
        }
    
    def shutdown(self):
        self.running = False
        if self.mavlink_thread:
            self.mavlink_thread.join(timeout=2.0)
        if self.mavlink_connection:
            self.mavlink_connection.close()

# ==============================================================================
# DATA LOGGER
# ==============================================================================

class ServoDataLogger:
    def __init__(self, log_dir="servo_logs", gps_handler=None):
        self.gps_handler = gps_handler
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"session_{timestamp}.json"
        self.gps_csv_file = self.log_dir / f"gps_points_{timestamp}.csv"
        
        self.csv_headers = [
            'timestamp', 'frame_count', 'pan_angle', 'tilt_angle',
            'pan_velocity', 'tilt_velocity', 'detection_confidence',
            'person_detected', 'tracking_active', 'target_lost_frames',
            'distance_meters', 'x_position', 'y_position', 'z_position',
            'angular_width', 'angular_height', 'bbox_width', 'bbox_height',
            'gps_latitude', 'gps_longitude', 'gps_altitude'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_headers)
        
        self.gps_headers = [
            'timestamp', 'detection_lat', 'detection_lon', 'detection_alt',
            'vehicle_lat', 'vehicle_lon', 'vehicle_heading',
            'relative_x', 'relative_y', 'relative_z', 'confidence'
        ]
        
        with open(self.gps_csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.gps_headers)
        
        self.session_data = {
            'start_time': datetime.now().isoformat(),
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
                'gps_points_created': 0
            },
            'events': []
        }
        
        print(f"📊 Data logging to: {self.log_dir}")
    
    def log_frame_data(self, frame_count, pan_angle, tilt_angle, pan_velocity,
                      tilt_velocity, detection_confidence, person_detected,
                      tracking_active, target_lost_frames, distance_data=None):
        try:
            distance = x_pos = y_pos = z_pos = 0.0
            angular_width = angular_height = bbox_width = bbox_height = 0.0
            gps_lat = gps_lon = gps_alt = 0.0
            
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
                        self.session_data['statistics']['gps_points_created'] += 1
                        
                        with open(self.gps_csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                gps_point['timestamp'],
                                gps_point['latitude'],
                                gps_point['longitude'],
                                gps_point['altitude'],
                                gps_point['vehicle_lat'],
                                gps_point['vehicle_lon'],
                                gps_point['vehicle_heading'],
                                gps_point['relative_x'],
                                gps_point['relative_y'],
                                gps_point['relative_z'],
                                gps_point['confidence']
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
            
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.time(), frame_count, pan_angle, tilt_angle,
                    pan_velocity, tilt_velocity, detection_confidence,
                    person_detected, tracking_active, target_lost_frames,
                    distance, x_pos, y_pos, z_pos,
                    angular_width, angular_height, bbox_width, bbox_height,
                    gps_lat, gps_lon, gps_alt
                ])
            
            if person_detected:
                self.session_data['statistics']['total_detections'] += 1
            if abs(pan_velocity) > 1 or abs(tilt_velocity) > 1:
                self.session_data['statistics']['total_movements'] += 1
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_event(self, event_type, description):
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description
        }
        self.session_data['events'].append(event)
        print(f"📝 {event_type}: {description}")
    
    def finalize_session(self):
        self.session_data['end_time'] = datetime.now().isoformat()
        
        if self.gps_handler:
            self.session_data['gps_status'] = self.gps_handler.get_status()
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            stats = self.session_data['statistics']
            print(f"\n📊 Session Complete:")
            print(f"   Detections: {stats['total_detections']}")
            print(f"   GPS Points: {stats['gps_points_created']}")
            if stats['distance_samples'] > 0:
                print(f"   Distance Range: {stats['min_distance']:.2f}m - {stats['max_distance']:.2f}m")
        except Exception as e:
            print(f"Session save error: {e}")

# ==============================================================================
# DISTANCE CALCULATOR
# ==============================================================================

class DistanceCalculator:
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length_x = frame_width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = frame_height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
        self.distance_history = deque(maxlen=5)
    
    def update_frame_size(self, width, height):
        self.frame_width = width
        self.frame_height = height
        self.focal_length_x = width / (2 * math.tan(math.radians(CAMERA_FOV_HORIZONTAL / 2)))
        self.focal_length_y = height / (2 * math.tan(math.radians(CAMERA_FOV_VERTICAL / 2)))
    
    def calculate_distance_from_bbox(self, bbox):
        bbox_width_pixels = bbox.width() * self.frame_width
        bbox_height_pixels = bbox.height() * self.frame_height
        
        distance_from_height = (AVERAGE_PERSON_HEIGHT * self.focal_length_y) / bbox_height_pixels
        distance_from_width = (AVERAGE_PERSON_WIDTH * self.focal_length_x) / bbox_width_pixels
        
        distance = (distance_from_height * 0.7 + distance_from_width * 0.3)
        self.distance_history.append(distance)
        
        return self._get_smoothed_distance()
    
    def calculate_3d_position(self, bbox, pan_angle, tilt_angle, distance):
        pan_rad = math.radians(pan_angle - 90)
        actual_tilt_angle = tilt_angle + CAMERA_TILT_OFFSET
        tilt_rad = math.radians(90 - actual_tilt_angle)
        
        horizontal_distance = distance * math.cos(tilt_rad)
        
        x = horizontal_distance * math.sin(pan_rad)
        y = horizontal_distance * math.cos(pan_rad)
        z = distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT
        
        return x, y, z
    
    def calculate_angular_size(self, bbox):
        angular_width = bbox.width() * CAMERA_FOV_HORIZONTAL
        angular_height = bbox.height() * CAMERA_FOV_VERTICAL
        return angular_width, angular_height
    
    def _get_smoothed_distance(self):
        if not self.distance_history:
            return 0.0
        
        sorted_distances = sorted(self.distance_history)
        if len(sorted_distances) >= 3:
            filtered = sorted_distances[1:-1]
        else:
            filtered = sorted_distances
        
        return sum(filtered) / len(filtered) if filtered else sorted_distances[0]

# ==============================================================================
# SERVO CONTROLLER
# ==============================================================================

class FastServoController:
    def __init__(self, logger=None):
        self.logger = logger
        
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = adafruit_pca9685.PCA9685(self.i2c)
        self.pca.frequency = 50
        
        self.pan_servo = servo.Servo(self.pca.channels[0])
        self.tilt_servo = servo.Servo(self.pca.channels[1])
        
        self.current_pan = 90.0
        self.current_tilt = 90.0 - CAMERA_TILT_OFFSET
        self.velocity_pan = 0.0
        self.velocity_tilt = 0.0
        self.last_update_time = time.time()
        
        self.pan_servo.angle = self.current_pan
        self.tilt_servo.angle = self.current_tilt
        
        self.command_queue = Queue(maxsize=5)
        self.running = True
        self.servo_thread = threading.Thread(target=self._servo_worker, daemon=True)
        self.servo_thread.start()
        
        if self.logger:
            self.logger.log_event('servo_init', 'Servo controller initialized')
    
    def _servo_worker(self):
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.05)
                if command is None:
                    break
                
                pan_angle, tilt_angle = command
                current_time = time.time()
                dt = current_time - self.last_update_time
                
                if dt > 0:
                    self.velocity_pan = (pan_angle - self.current_pan) / dt
                    self.velocity_tilt = (tilt_angle - self.current_tilt) / dt
                
                if (abs(pan_angle - self.current_pan) > 0.1 or 
                    abs(tilt_angle - self.current_tilt) > 0.1):
                    try:
                        self.pan_servo.angle = pan_angle
                        self.tilt_servo.angle = tilt_angle
                        self.current_pan = pan_angle
                        self.current_tilt = tilt_angle
                        time.sleep(0.005)
                    except Exception as e:
                        print(f"Servo movement error: {e}")
                
                self.last_update_time = current_time
                
            except Empty:
                continue
            except Exception as e:
                print(f"Servo thread error: {e}")
    
    def move_to(self, pan_angle, tilt_angle):
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        
        try:
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    break
            self.command_queue.put_nowait((pan_angle, tilt_angle))
        except:
            pass
    
    def get_current_state(self):
        return self.current_pan, self.current_tilt, self.velocity_pan, self.velocity_tilt
    
    def shutdown(self):
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

# ==============================================================================
# TRACKER
# ==============================================================================

class UltraFastTracker:
    def __init__(self, servo_controller, logger=None):
        self.servo = servo_controller
        self.logger = logger
        
        self.frame_center_x = 320
        self.frame_center_y = 240
        self.frame_width = 640
        self.frame_height = 480
        
        self.distance_calculator = DistanceCalculator(self.frame_width, self.frame_height)
        
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        self.lock_on_target = False
        self.frame_skip_counter = 0
        self.current_distance = 0.0
        self.current_3d_position = (0.0, 0.0, 0.0)
        
        self.pan_history = deque(maxlen=DETECTION_HISTORY_SIZE)
        self.tilt_history = deque(maxlen=DETECTION_HISTORY_SIZE)
    
    def update_frame_properties(self, width, height):
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            self.distance_calculator.update_frame_size(width, height)
            
            if self.logger:
                self.logger.log_event('resolution_change', f'Frame: {width}x{height}')
    
    def track_person(self, bbox, confidence, frame_count):
        self.frame_skip_counter += 1
        if self.frame_skip_counter < FRAME_SKIP_COUNT:
            return
        self.frame_skip_counter = 0
        
        self.current_distance = self.distance_calculator.calculate_distance_from_bbox(bbox)
        
        current_pan, current_tilt, pan_vel, tilt_vel = self.servo.get_current_state()
        
        x, y, z = self.distance_calculator.calculate_3d_position(
            bbox, current_pan, current_tilt, self.current_distance
        )
        self.current_3d_position = (x, y, z)
        
        angular_width, angular_height = self.distance_calculator.calculate_angular_size(bbox)
        
        center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
        center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
        
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        
        dynamic_dead_zone = DEAD_ZONE * (1 + self.current_distance / 10.0)
        
        if abs(error_x) > dynamic_dead_zone or abs(error_y) > dynamic_dead_zone:
            distance_factor = min(2.0, max(0.5, 2.0 / self.current_distance))
            
            pan_adjustment = -error_x * (PAN_SENSITIVITY / self.frame_width) * distance_factor
            tilt_adjustment = error_y * (TILT_SENSITIVITY / self.frame_height) * distance_factor
            
            confidence_multiplier = min(2.0, confidence + 0.5)
            pan_adjustment *= confidence_multiplier
            tilt_adjustment *= confidence_multiplier
            
            target_pan = current_pan + pan_adjustment
            target_tilt = current_tilt + tilt_adjustment
            
            new_pan = self._smooth_angle(current_pan, target_pan)
            new_tilt = self._smooth_angle(current_tilt, target_tilt)
            
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
                    self.logger.log_event('target_lock', f'Target locked at {self.current_distance:.2f}m')
                print(f"🎯 Target locked at {self.current_distance:.2f}m")
        
        self.last_detection_time = time.time()
        self.target_lost_frames = 0
        
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
        self.target_lost_frames += 1
        
        if self.target_lost_frames > 10 and self.lock_on_target:
            self.lock_on_target = False
            if self.logger:
                self.logger.log_event('target_lost', 'Target lost - scanning mode')
            print("🔍 Target lost - scanning mode")
        
        if self.logger:
            current_pan, current_tilt, pan_vel, tilt_vel = self.servo.get_current_state()
            self.logger.log_frame_data(
                frame_count, current_pan, current_tilt, pan_vel, tilt_vel,
                0.0, False, self.is_tracking_active(), self.target_lost_frames,
                None
            )
    
    def _smooth_angle(self, current, target):
        diff = (target - current) * SMOOTHING_FACTOR
        diff = max(-MAX_STEP_SIZE, min(MAX_STEP_SIZE, diff))
        return current + diff
    
    def is_tracking_active(self):
        return (time.time() - self.last_detection_time) < DETECTION_TIMEOUT
    
    def get_current_distance_info(self):
        return {
            'distance': self.current_distance,
            'position_3d': self.current_3d_position
        }

# ==============================================================================
# CALIBRATION HELPER
# ==============================================================================

class CalibrationHelper:
    def __init__(self):
        self.measurements = []
        self.calibrating = CALIBRATION_MODE
    
    def add_measurement(self, bbox, frame_height):
        if not self.calibrating:
            return
        
        person_height_pixels = bbox.height() * frame_height
        
        if person_height_pixels > 50:
            self.measurements.append(person_height_pixels)
            
            if len(self.measurements) >= 10:
                avg_height = sum(self.measurements) / len(self.measurements)
                focal_length = (avg_height * CALIBRATION_DISTANCE) / AVERAGE_PERSON_HEIGHT
                
                print(f"\n📸 CALIBRATION COMPLETE!")
                print(f"   Average person height: {avg_height:.1f} pixels")
                print(f"   Calculated focal length: {focal_length:.1f} pixels")
                print(f"   Update FOCAL_LENGTH_PIXELS = {focal_length:.1f}")
                
                self.calibrating = False
                self.measurements = []

# ==============================================================================
# CALLBACK CLASS
# ==============================================================================

class OptimizedAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
    
    def new_function(self):
        return "Ultra-Fast Tracking with GPS Waypoint Guidance: "

# ==============================================================================
# MAIN CALLBACK FUNCTION
# ==============================================================================

def ultra_fast_app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    frame_count = user_data.get_count()
    
    if frame_count % 30 == 0:
        format, width, height = get_caps_from_pad(pad)
        if width and height:
            tracker.update_frame_properties(width, height)
    
    frame = None
    if user_data.use_frame:
        format, width, height = get_caps_from_pad(pad)
        if format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)
    
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
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
    
    if best_person:
        tracker.track_person(best_person['bbox'], best_person['confidence'], frame_count)
        
        if calibration_helper.calibrating:
            calibration_helper.add_measurement(best_person['bbox'], tracker.frame_height)
        
        if frame_count % 60 == 0:
            distance_info = tracker.get_current_distance_info()
            print(f"🏃 Tracking: Conf {best_person['confidence']:.2f}, "
                  f"Distance: {distance_info['distance']:.2f}m")
    else:
        tracker.handle_lost_target(frame_count)
    
    if user_data.use_frame and frame is not None:
        center_x, center_y = tracker.frame_center_x, tracker.frame_center_y
        
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 1)
        
        if best_person:
            distance_info = tracker.get_current_distance_info()
            distance = distance_info['distance']
            x, y, z = distance_info['position_3d']
            
            if calibration_helper.calibrating:
                cv2.putText(frame, "CALIBRATION MODE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"TRACKING: {distance:.2f}m", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"3D Pos: ({x:.1f}, {y:.1f}, {z:.1f})m", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if best_person['bbox']:
                bbox = best_person['bbox']
                x1 = int(bbox.xmin() * frame.shape[1])
                y1 = int(bbox.ymin() * frame.shape[0])
                x2 = int((bbox.xmin() + bbox.width()) * frame.shape[1])
                y2 = int((bbox.ymin() + bbox.height()) * frame.shape[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{distance:.1f}m", 
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        pan, tilt, pan_vel, tilt_vel = fast_servo_controller.get_current_state()
        cv2.putText(frame, f"Pan: {pan:.1f}° ({pan_vel:.1f}°/s)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Tilt: {tilt:.1f}° ({tilt_vel:.1f}°/s)", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if mavlink_handler:
            status = mavlink_handler.get_status()
            gps_text = f"GPS: {status['satellites']} sats"
            if status['gps_fix'] >= 3:
                cv2.putText(frame, gps_text, (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, gps_text, (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    return Gst.PadProbeReturn.OK

# ==============================================================================
# MAIN INITIALIZATION AND EXECUTION
# ==============================================================================

print("Initializing ULTRA-FAST servo system with GPS waypoint guidance...")

# Initialize MAVLink connection
mavlink_handler = None
try:
    mavlink_handler = MAVLinkGPSHandler()
except Exception as e:
    print(f"⚠️  MAVLink initialization failed: {e}")
    print("   Continuing without GPS waypoint functionality")

# Initialize data logger
data_logger = ServoDataLogger(gps_handler=mavlink_handler)

# Initialize servo controller
fast_servo_controller = FastServoController(data_logger)

# Initialize tracker
tracker = UltraFastTracker(fast_servo_controller, data_logger)

# Initialize calibration helper
calibration_helper = CalibrationHelper()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    user_data = OptimizedAppCallback()
    app = GStreamerDetectionApp(ultra_fast_app_callback, user_data)
    
    print("🚀 Starting ULTRA-FAST tracking with GPS WAYPOINT GUIDANCE...")
    
    if mavlink_handler:
        status = mavlink_handler.get_status()
        print(f"\n🛰️ GPS Status:")
        print(f"   Fix Type: {status['gps_fix']} (3=3D fix)")
        print(f"   Satellites: {status['satellites']}")
        print(f"   Current Position: {status['latitude']:.6f}, {status['longitude']:.6f}")
        print(f"   Altitude: {status['altitude']:.1f}m")
    
    print(f"\n📊 Data output location: {data_logger.log_dir}")
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
        
        if mavlink_handler:
            print("🛰️ Closing MAVLink connection...")
            mavlink_handler.shutdown()
        
        print("✅ Shutdown complete")
