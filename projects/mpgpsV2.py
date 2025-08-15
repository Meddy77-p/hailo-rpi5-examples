#!/usr/bin/env python3
"""
ENHANCED Servo Tracking System v3.0 - FULLY OPTIMIZED VERSION
For Raspberry Pi 5 with CubeOrange and Hailo AI
Optimized for performance, reliability, and GPS waypoint generation
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
import subprocess
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# ==============================================================================
# OPTIMIZED CONFIGURATION CONSTANTS
# ==============================================================================

# Servo tracking parameters (optimized for Raspberry Pi 5)
DEAD_ZONE = 15
SMOOTHING_FACTOR = 0.35
MAX_STEP_SIZE = 6
MIN_CONFIDENCE = 0.35
DETECTION_TIMEOUT = 1.5
PAN_SENSITIVITY = 50
TILT_SENSITIVITY = 40
FRAME_SKIP_COUNT = 1  # Process every 2nd frame to reduce CPU load
DETECTION_HISTORY_SIZE = 3
SERVO_UPDATE_RATE = 60  # Hz (reduced from 100Hz for stability)

# Camera parameters (will be auto-calibrated)
CAMERA_FOV_HORIZONTAL = 79.9
CAMERA_FOV_VERTICAL = 64.3
AVERAGE_PERSON_HEIGHT = 1.7
AVERAGE_PERSON_WIDTH = 0.45

# Physical setup
SERVO_MOUNT_HEIGHT = 1.3
CAMERA_TILT_OFFSET = 5.0

# MAVLink configuration (optimized)
MAVLINK_CONNECTION = '/dev/serial0'
MAVLINK_BAUD = 921600
MAVLINK_SYSTEM_ID = 255
MAVLINK_COMPONENT_ID = 190
MAVLINK_TIMEOUT = 5
MAVLINK_RETRY_INTERVAL = 3
MAVLINK_HEARTBEAT_TIMEOUT = 20

# GPS waypoint parameters
GPS_UPDATE_INTERVAL = 0.5
MIN_DISTANCE_FOR_GPS = 2.0
MAX_GPS_POINTS = 100
WAYPOINT_ALTITUDE_OFFSET = 2.0
GPS_ACCURACY_THRESHOLD = 3.0

# Auto-calibration parameters (improved)
AUTO_CALIBRATION_SAMPLES = 25
CALIBRATION_TIMEOUT = 90
CALIBRATION_MIN_BBOX_SIZE = 0.08
CALIBRATION_MAX_BBOX_SIZE = 0.6
CALIBRATION_DISTANCE_RANGE = (2.0, 8.0)
CALIBRATION_MIN_CONFIDENCE = 0.6

# Performance optimization
BUFFER_SIZE = 1
QUEUE_TIMEOUT = 0.01
DEQUE_SIZE = 50

# ==============================================================================
# ENHANCED DATA STRUCTURES
# ==============================================================================

@dataclass
class CalibrationData:
    focal_length_pixels: float
    samples_collected: int
    confidence_sum: float
    completed: bool
    start_time: float
    quality_score: float = 0.0

@dataclass
class GPSPosition:
    latitude: float
    longitude: float
    altitude: float
    timestamp: float
    fix_type: int
    satellites: int
    hdop: float = 0.0
    vdop: float = 0.0

@dataclass
class Detection3D:
    x: float
    y: float
    z: float
    distance: float
    confidence: float
    bbox_area: float
    timestamp: float

# ==============================================================================
# FIXED MAVLINK GPS HANDLER
# ==============================================================================

class FixedMAVLinkGPSHandler:
    def __init__(self, connection_string=MAVLINK_CONNECTION, baud=MAVLINK_BAUD):
        self.connection_string = connection_string
        self.baud = baud
        self.mavlink_connection = None
        
        # GPS state
        self.current_position = GPSPosition(0.0, 0.0, 0.0, 0.0, 0, 0)
        self.current_heading = 0.0
        self.barometric_altitude = 0.0
        self.relative_altitude = 0.0
        self.terrain_altitude = 0.0
        
        # Home position
        self.home_position = None
        
        # Connection management
        self.running = True
        self.connected = False
        self.last_heartbeat = 0
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Threading
        self.mavlink_thread = None
        self.data_lock = threading.RLock()
        
        # GPS logging
        self.gps_points = deque(maxlen=MAX_GPS_POINTS)
        self.last_point_time = 0
        self.EARTH_RADIUS = 6371000
        
        # Setup logging
        self.logger = logging.getLogger('MAVLink')
        
        # Try to fix serial port issues
        self._fix_serial_port_issues()
        self.connect()
    
    def _fix_serial_port_issues(self):
        """Fix common serial port issues on Raspberry Pi"""
        try:
            # Kill any processes using the serial port
            subprocess.run(['sudo', 'fuser', '-k', self.connection_string], 
                          capture_output=True, check=False)
            time.sleep(0.5)
            
            # Reset the serial port
            subprocess.run(['sudo', 'stty', '-F', self.connection_string, 'sane'], 
                          capture_output=True, check=False)
            
            # Set proper permissions
            subprocess.run(['sudo', 'chmod', '666', self.connection_string], 
                          capture_output=True, check=False)
            
            self.logger.info(f"Serial port {self.connection_string} reset")
            
        except Exception as e:
            self.logger.warning(f"Could not reset serial port: {e}")
    
    def connect(self):
        """Enhanced connection with better error handling"""
        try:
            self.logger.info(f"Connecting to CubeOrange at {self.connection_string}...")
            
            # Use a more permissive connection method
            self.mavlink_connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud,
                source_system=MAVLINK_SYSTEM_ID,
                source_component=MAVLINK_COMPONENT_ID,
                timeout=MAVLINK_TIMEOUT,
                robust_parsing=True
            )
            
            # Wait for heartbeat with shorter timeout
            self.logger.info("Waiting for heartbeat...")
            heartbeat = self.mavlink_connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                self.connected = True
                self.last_heartbeat = time.time()
                self.connection_attempts = 0
                
                self.logger.info(f"MAVLink connection established! System: {heartbeat.autopilot}")
                self.request_data_streams()
                
                # Start receiver thread
                self.mavlink_thread = threading.Thread(target=self._mavlink_receiver, daemon=True)
                self.mavlink_thread.start()
                
                return True
            else:
                raise Exception("No heartbeat received")
                
        except Exception as e:
            self.logger.error(f"MAVLink connection failed: {e}")
            self.connected = False
            self.connection_attempts += 1
            
            if self.connection_attempts < self.max_connection_attempts:
                self.logger.info(f"Retrying connection in {MAVLINK_RETRY_INTERVAL} seconds...")
                time.sleep(MAVLINK_RETRY_INTERVAL)
                return self.connect()
            
            return False
    
    def request_data_streams(self):
        """Request necessary data streams"""
        try:
            # Request GPS data at 5Hz (reduced for reliability)
            self.mavlink_connection.mav.request_data_stream_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                5, 1
            )
            
            # Request attitude data at 5Hz
            self.mavlink_connection.mav.request_data_stream_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
                5, 1
            )
            
            self.logger.info("Data streams requested")
            
        except Exception as e:
            self.logger.error(f"Error requesting data streams: {e}")
    
    def _mavlink_receiver(self):
        """Message receiver with better error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running and self.connected:
            try:
                msg = self.mavlink_connection.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                
                consecutive_errors = 0
                msg_type = msg.get_type()
                current_time = time.time()
                
                with self.data_lock:
                    if msg_type == 'HEARTBEAT':
                        self.last_heartbeat = current_time
                        
                    elif msg_type == 'GPS_RAW_INT':
                        self.current_position = GPSPosition(
                            latitude=msg.lat / 1e7,
                            longitude=msg.lon / 1e7,
                            altitude=msg.alt / 1000.0,
                            timestamp=current_time,
                            fix_type=msg.fix_type,
                            satellites=msg.satellites_visible,
                            hdop=msg.eph / 100.0 if hasattr(msg, 'eph') else 0.0,
                            vdop=msg.epv / 100.0 if hasattr(msg, 'epv') else 0.0
                        )
                        
                    elif msg_type == 'GLOBAL_POSITION_INT':
                        self.current_position.latitude = msg.lat / 1e7
                        self.current_position.longitude = msg.lon / 1e7
                        self.current_position.altitude = msg.alt / 1000.0
                        self.relative_altitude = msg.relative_alt / 1000.0
                        self.current_heading = msg.hdg / 100.0 if msg.hdg != 65535 else self.current_heading
                        
                    elif msg_type == 'ATTITUDE':
                        yaw_rad = msg.yaw
                        self.current_heading = math.degrees(yaw_rad) % 360
                        
                    elif msg_type == 'VFR_HUD':
                        self.barometric_altitude = msg.alt
                        
                    elif msg_type == 'HOME_POSITION':
                        self.home_position = GPSPosition(
                            latitude=msg.latitude / 1e7,
                            longitude=msg.longitude / 1e7,
                            altitude=msg.altitude / 1000.0,
                            timestamp=current_time,
                            fix_type=3,
                            satellites=0
                        )
                        
            except Exception as e:
                consecutive_errors += 1
                if self.running and self.connected:
                    self.logger.warning(f"MAVLink receive error ({consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors - disconnecting")
                        self.connected = False
                        break
                    
                    time.sleep(0.1)
    
    def get_best_altitude(self) -> float:
        """Get the most accurate altitude available"""
        with self.data_lock:
            if self.relative_altitude > 0 and self.home_position:
                return self.home_position.altitude + self.relative_altitude
            
            if self.barometric_altitude > 0:
                return self.barometric_altitude
            
            return self.current_position.altitude
    
    def calculate_enhanced_gps_position(self, x_meters: float, y_meters: float, z_meters: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate GPS position from relative coordinates"""
        with self.data_lock:
            if self.current_position.fix_type < 3:
                return None, None, None
            
            # Calculate horizontal position
            bearing_to_target = math.degrees(math.atan2(x_meters, y_meters))
            absolute_bearing = (self.current_heading + bearing_to_target) % 360
            horizontal_distance = math.sqrt(x_meters**2 + y_meters**2)
            
            lat_rad = math.radians(self.current_position.latitude)
            lon_rad = math.radians(self.current_position.longitude)
            bearing_rad = math.radians(absolute_bearing)
            
            new_lat_rad = math.asin(
                math.sin(lat_rad) * math.cos(horizontal_distance / self.EARTH_RADIUS) +
                math.cos(lat_rad) * math.sin(horizontal_distance / self.EARTH_RADIUS) * math.cos(bearing_rad)
            )
            
            new_lon_rad = lon_rad + math.atan2(
                math.sin(bearing_rad) * math.sin(horizontal_distance / self.EARTH_RADIUS) * math.cos(lat_rad),
                math.cos(horizontal_distance / self.EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad)
            )
            
            # Calculate target altitude
            current_altitude = self.get_best_altitude()
            target_altitude = current_altitude + z_meters + WAYPOINT_ALTITUDE_OFFSET
            
            return math.degrees(new_lat_rad), math.degrees(new_lon_rad), target_altitude
    
    def add_detection_point(self, detection_3d) -> Optional[Dict]:
        """Add a detection point and create waypoint"""
        current_time = time.time()
        
        if current_time - self.last_point_time < GPS_UPDATE_INTERVAL:
            return None
        
        if detection_3d.distance < MIN_DISTANCE_FOR_GPS:
            return None
        
        with self.data_lock:
            if (self.current_position.fix_type < 3 or 
                self.current_position.hdop > GPS_ACCURACY_THRESHOLD):
                return None
        
        lat, lon, alt = self.calculate_enhanced_gps_position(
            detection_3d.x, detection_3d.y, detection_3d.z
        )
        
        if lat is None or lon is None or alt is None:
            return None
        
        gps_point = {
            'timestamp': current_time,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'relative_x': detection_3d.x,
            'relative_y': detection_3d.y,
            'relative_z': detection_3d.z,
            'distance': detection_3d.distance,
            'confidence': detection_3d.confidence,
            'vehicle_lat': self.current_position.latitude,
            'vehicle_lon': self.current_position.longitude,
            'vehicle_altitude': self.get_best_altitude(),
            'vehicle_heading': self.current_heading,
            'gps_accuracy': self.current_position.hdop
        }
        
        self.gps_points.append(gps_point)
        self.last_point_time = current_time
        
        # Upload waypoint (simplified for reliability)
        success = self._upload_simple_waypoint(lat, lon, alt)
        
        if success:
            bearing = math.degrees(math.atan2(detection_3d.x, detection_3d.y))
            bearing = (self.current_heading + bearing) % 360
            self.logger.info(f"New waypoint: {detection_3d.distance:.1f}m @ {bearing:.0f}° alt {alt:.1f}m")
        
        return gps_point
    
    def _upload_simple_waypoint(self, lat: float, lon: float, alt: float) -> bool:
        """Simplified waypoint upload"""
        try:
            if not self.connected:
                return False
            
            # Get current mission count
            self.mavlink_connection.mav.mission_request_list_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component
            )
            
            # Wait for mission count
            msg = self.mavlink_connection.recv_match(type='MISSION_COUNT', blocking=True, timeout=2)
            wp_seq = msg.count if msg else 1
            
            # Send mission count
            self.mavlink_connection.mav.mission_count_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                wp_seq + 1, 0
            )
            
            # Wait for mission request
            msg = self.mavlink_connection.recv_match(type='MISSION_REQUEST', blocking=True, timeout=2)
            
            if msg and msg.seq == wp_seq:
                # Send the waypoint
                self.mavlink_connection.mav.mission_item_int_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component,
                    wp_seq,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 1, 0, 5, 0, float('nan'),
                    int(lat * 1e7),
                    int(lon * 1e7),
                    alt, 0
                )
                
                # Wait for ACK
                ack_msg = self.mavlink_connection.recv_match(type='MISSION_ACK', blocking=True, timeout=2)
                return ack_msg and ack_msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED
            
            return False
            
        except Exception as e:
            self.logger.error(f"Waypoint upload error: {e}")
            return False
    
    def get_enhanced_status(self) -> Dict:
        """Get system status"""
        with self.data_lock:
            return {
                'connected': self.connected,
                'gps_fix': self.current_position.fix_type,
                'satellites': self.current_position.satellites,
                'gps_accuracy': self.current_position.hdop,
                'latitude': self.current_position.latitude,
                'longitude': self.current_position.longitude,
                'gps_altitude': self.current_position.altitude,
                'barometric_altitude': self.barometric_altitude,
                'relative_altitude': self.relative_altitude,
                'best_altitude': self.get_best_altitude(),
                'heading': self.current_heading,
                'last_update': time.time() - self.current_position.timestamp,
                'points_logged': len(self.gps_points),
                'connection_attempts': self.connection_attempts
            }
    
    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("Shutting down MAVLink handler...")
        self.running = False
        
        if self.mavlink_thread and self.mavlink_thread.is_alive():
            self.mavlink_thread.join(timeout=2.0)
        
        if self.mavlink_connection:
            self.mavlink_connection.close()
        
        print("✅ MAVLink handler shutdown complete")

# ==============================================================================
# IMPROVED AUTO-CALIBRATION SYSTEM
# ==============================================================================

class ImprovedAutoCalibrationSystem:
    def __init__(self, logger=None):
        self.logger = logger
        self.calibration_data = CalibrationData(
            focal_length_pixels=800.0,
            samples_collected=0,
            confidence_sum=0.0,
            completed=False,
            start_time=time.time()
        )
        
        # Improved sample collection
        self.distance_samples = []
        self.height_samples = []
        self.width_samples = []
        self.confidence_samples = []
        self.bbox_area_samples = []
        
        # More robust distance set
        self.calibration_distances = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.current_distance_idx = 0
        self.samples_per_distance = 5
        self.current_distance_samples = 0
        
        # Quality control
        self.min_confidence = CALIBRATION_MIN_CONFIDENCE
        self.max_aspect_ratio_deviation = 0.3
        self.min_bbox_stability = 0.95
        
        self.phase = "initializing"
        self.recent_detections = deque(maxlen=10)
        
        print("🎯 Starting improved automatic calibration system...")
        print("📏 Higher quality thresholds for better accuracy")
    
    def should_accept_sample(self, bbox, confidence: float, frame_width: int, frame_height: int) -> bool:
        """Enhanced sample acceptance with stricter quality control"""
        if self.calibration_data.completed:
            return False
        
        # Higher confidence threshold
        if confidence < self.min_confidence:
            return False
        
        # Check bbox size constraints (refined)
        bbox_area = bbox.width() * bbox.height()
        if bbox_area < CALIBRATION_MIN_BBOX_SIZE or bbox_area > CALIBRATION_MAX_BBOX_SIZE:
            return False
        
        # Stricter aspect ratio (should look like a person)
        aspect_ratio = bbox.height() / bbox.width()
        if aspect_ratio < 1.8 or aspect_ratio > 3.5:
            return False
        
        # Must be well-centered
        center_x = bbox.xmin() + bbox.width() / 2
        center_y = bbox.ymin() + bbox.height() / 2
        
        if abs(center_x - 0.5) > 0.25 or abs(center_y - 0.5) > 0.25:
            return False
        
        # Check detection stability
        current_detection = {
            'bbox_area': bbox_area,
            'aspect_ratio': aspect_ratio,
            'center_x': center_x,
            'center_y': center_y,
            'confidence': confidence
        }
        
        self.recent_detections.append(current_detection)
        
        if len(self.recent_detections) >= 5:
            # Check if recent detections are stable
            areas = [d['bbox_area'] for d in self.recent_detections[-5:]]
            confidences = [d['confidence'] for d in self.recent_detections[-5:]]
            
            area_stability = 1.0 - (np.std(areas) / np.mean(areas))
            confidence_stability = 1.0 - (np.std(confidences) / np.mean(confidences))
            
            if area_stability < self.min_bbox_stability or confidence_stability < 0.9:
                return False
        
        return True
    
    def add_calibration_sample(self, bbox, confidence: float, frame_width: int, frame_height: int):
        """Enhanced calibration sample processing"""
        if not self.should_accept_sample(bbox, confidence, frame_width, frame_height):
            return
        
        if self.phase == "initializing":
            self.phase = "collecting"
            self._announce_current_distance()
        
        person_height_pixels = bbox.height() * frame_height
        person_width_pixels = bbox.width() * frame_width
        bbox_area = bbox.width() * bbox.height()
        
        # Store sample data
        self.height_samples.append(person_height_pixels)
        self.width_samples.append(person_width_pixels)
        self.confidence_samples.append(confidence)
        self.bbox_area_samples.append(bbox_area)
        self.distance_samples.append(self.calibration_distances[self.current_distance_idx])
        
        self.calibration_data.samples_collected += 1
        self.calibration_data.confidence_sum += confidence
        self.current_distance_samples += 1
        
        # Calculate quality metrics for this sample
        aspect_ratio = bbox.height() / bbox.width()
        quality_indicators = [
            confidence,
            min(1.0, bbox_area * 10),
            max(0.0, 1.0 - abs(aspect_ratio - 2.2) / 1.0),
            max(0.0, 1.0 - abs((bbox.xmin() + bbox.width()/2) - 0.5) * 2),
        ]
        sample_quality = np.mean(quality_indicators)
        
        print(f"📊 Sample {self.calibration_data.samples_collected}: "
              f"height={person_height_pixels:.1f}px, conf={confidence:.2f}, "
              f"quality={sample_quality:.2f}")
        
        # Check if we have enough good samples for this distance
        if self.current_distance_samples >= self.samples_per_distance:
            self.current_distance_idx += 1
            self.current_distance_samples = 0
            
            if self.current_distance_idx < len(self.calibration_distances):
                self._announce_current_distance()
            else:
                self._finalize_calibration()
        
        # Safety timeout
        if time.time() - self.calibration_data.start_time > CALIBRATION_TIMEOUT:
            print("⏰ Calibration timeout - analyzing collected data")
            self._finalize_calibration()
    
    def _announce_current_distance(self):
        """Announce current distance with progress"""
        distance = self.calibration_distances[self.current_distance_idx]
        print(f"\n📏 Please position a person at {distance:.1f} meters distance")
        print(f"   Need {self.samples_per_distance} high-quality samples")
        print(f"   Progress: {self.current_distance_idx + 1}/{len(self.calibration_distances)} distances")
        print(f"   Requirements: Confidence >{self.min_confidence:.1f}, stable detection, well-centered")
    
    def _finalize_calibration(self):
        """Enhanced calibration finalization with quality assessment"""
        if len(self.height_samples) < 10:
            print("❌ Insufficient calibration data - using default values")
            self.calibration_data.completed = True
            return
        
        self.phase = "analyzing"
        print("\n🔬 Analyzing calibration data with quality assessment...")
        
        # Calculate focal length with weighted approach
        focal_lengths = []
        weights = []
        
        for i, (height_px, distance, confidence, area) in enumerate(
            zip(self.height_samples, self.distance_samples, 
                self.confidence_samples, self.bbox_area_samples)):
            
            # Calculate focal length from height
            focal_length = (height_px * distance) / AVERAGE_PERSON_HEIGHT
            focal_lengths.append(focal_length)
            
            # Weight by confidence and bbox area
            weight = confidence * (area ** 0.5)
            weights.append(weight)
        
        # Convert to numpy arrays
        focal_array = np.array(focal_lengths)
        weight_array = np.array(weights)
        
        # Remove outliers using weighted approach
        weighted_mean = np.average(focal_array, weights=weight_array)
        deviations = np.abs(focal_array - weighted_mean)
        threshold = 2 * np.sqrt(np.average(deviations**2, weights=weight_array))
        
        # Keep samples within threshold
        valid_mask = deviations <= threshold
        if np.sum(valid_mask) > 5:
            final_focal_lengths = focal_array[valid_mask]
            final_weights = weight_array[valid_mask]
            
            # Weighted average of valid samples
            self.calibration_data.focal_length_pixels = np.average(
                final_focal_lengths, weights=final_weights
            )
            
            # Calculate quality metrics
            focal_std = np.std(final_focal_lengths)
            valid_count = np.sum(valid_mask)
            avg_confidence = np.mean(np.array(self.confidence_samples)[valid_mask])
            
        else:
            # Fallback to simple mean if too few valid samples
            self.calibration_data.focal_length_pixels = np.mean(focal_array)
            focal_std = np.std(focal_array)
            valid_count = len(focal_array)
            avg_confidence = np.mean(self.confidence_samples)
        
        # Calculate overall quality score
        quality_factors = [
            min(1.0, valid_count / 20),
            min(1.0, avg_confidence / 0.9),
            max(0.0, 1.0 - focal_std / 100),
            min(1.0, len(set(self.distance_samples)) / 6)
        ]
        
        self.calibration_data.quality_score = np.mean(quality_factors)
        
        # Determine quality rating
        if self.calibration_data.quality_score > 0.8:
            quality_rating = "Excellent"
        elif self.calibration_data.quality_score > 0.6:
            quality_rating = "Good"
        elif self.calibration_data.quality_score > 0.4:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        self.calibration_data.completed = True
        self.phase = "completed"
        
        print(f"\n✅ ENHANCED CALIBRATION COMPLETED!")
        print(f"   Total samples: {self.calibration_data.samples_collected}")
        print(f"   Valid samples used: {valid_count}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Calculated focal length: {self.calibration_data.focal_length_pixels:.1f} pixels")
        print(f"   Standard deviation: {focal_std:.1f} pixels")
        print(f"   Quality score: {self.calibration_data.quality_score:.2f}")
        print(f"   Calibration quality: {quality_rating}")
        
        if self.calibration_data.quality_score < 0.5:
            print("⚠️  Consider recalibrating with more stable detections")
        
        if self.logger:
            self.logger.log_event('calibration_complete', 
                f'Focal length: {self.calibration_data.focal_length_pixels:.1f}px, '
                f'quality: {quality_rating} ({self.calibration_data.quality_score:.2f})')
    
    def get_focal_length(self) -> float:
        """Get the calibrated focal length"""
        return self.calibration_data.focal_length_pixels
    
    def is_completed(self) -> bool:
        """Check if calibration is completed"""
        return self.calibration_data.completed
    
    def get_quality_score(self) -> float:
        """Get calibration quality score"""
        return self.calibration_data.quality_score
    
    def get_status(self) -> str:
        """Get current calibration status"""
        if self.phase == "initializing":
            return "Initializing enhanced calibration..."
        elif self.phase == "collecting":
            distance = self.calibration_distances[self.current_distance_idx]
            return f"Collecting samples at {distance:.1f}m ({self.current_distance_samples}/{self.samples_per_distance}) - Quality mode"
        elif self.phase == "analyzing":
            return "Analyzing calibration data with quality assessment..."
        else:
            quality_rating = "Excellent" if self.calibration_data.quality_score > 0.8 else \
                           "Good" if self.calibration_data.quality_score > 0.6 else \
                           "Fair" if self.calibration_data.quality_score > 0.4 else "Poor"
            return f"Enhanced calibration complete: {quality_rating} (focal: {self.calibration_data.focal_length_pixels:.1f}px)"

# ==============================================================================
# CPU-OPTIMIZED SERVO CONTROLLER
# ==============================================================================

class CPUOptimizedServoController:
    def __init__(self, logger=None):
        self.logger = logger
        
        # Hardware initialization
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = adafruit_pca9685.PCA9685(self.i2c)
            self.pca.frequency = 50
            
            self.pan_servo = servo.Servo(self.pca.channels[0], min_pulse=500, max_pulse=2500)
            self.tilt_servo = servo.Servo(self.pca.channels[1], min_pulse=500, max_pulse=2500)
            
        except Exception as e:
            print(f"❌ Servo initialization failed: {e}")
            raise
        
        # State tracking
        self.current_pan = 90.0
        self.current_tilt = 90.0 - CAMERA_TILT_OFFSET
        self.target_pan = self.current_pan
        self.target_tilt = self.current_tilt
        
        # Optimized movement parameters
        self.last_move_time = 0
        self.min_move_interval = 1.0 / SERVO_UPDATE_RATE
        self.movement_threshold = 0.5  # Minimum movement in degrees
        
        # Simplified smoothing
        self.pan_filter = self.current_pan
        self.tilt_filter = self.current_tilt
        self.filter_alpha = 0.7  # Smoothing factor
        
        # Performance monitoring
        self.move_count = 0
        self.last_stats_time = time.time()
        
        # Initialize to center
        self._move_servos_direct(self.current_pan, self.current_tilt)
        
        if self.logger:
            self.logger.log_event('servo_init', f'CPU-optimized servo controller initialized at {SERVO_UPDATE_RATE}Hz')
        
        print(f"⚡ CPU-optimized servo controller ready at {SERVO_UPDATE_RATE}Hz")
    
    def move_to_optimized(self, pan_angle: float, tilt_angle: float):
        """CPU-optimized movement with intelligent throttling"""
        current_time = time.time()
        
        # Throttle updates to reduce CPU load
        if current_time - self.last_move_time < self.min_move_interval:
            return
        
        # Clamp angles
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        
        # Apply smoothing filter
        self.pan_filter = self.pan_filter * (1 - self.filter_alpha) + pan_angle * self.filter_alpha
        self.tilt_filter = self.tilt_filter * (1 - self.filter_alpha) + tilt_angle * self.filter_alpha
        
        # Check if movement is significant enough
        pan_diff = abs(self.pan_filter - self.current_pan)
        tilt_diff = abs(self.tilt_filter - self.current_tilt)
        
        if pan_diff > self.movement_threshold or tilt_diff > self.movement_threshold:
            self._move_servos_direct(self.pan_filter, self.tilt_filter)
            self.current_pan = self.pan_filter
            self.current_tilt = self.tilt_filter
            self.last_move_time = current_time
            self.move_count += 1
        
        # Update targets for external monitoring
        self.target_pan = pan_angle
        self.target_tilt = tilt_angle
    
    def _move_servos_direct(self, pan_angle: float, tilt_angle: float):
        """Direct servo movement"""
        try:
            self.pan_servo.angle = pan_angle
            self.tilt_servo.angle = tilt_angle
        except Exception as e:
            if self.logger:
                self.logger.log_event('servo_error', f'Movement error: {e}')
    
    def get_enhanced_state(self):
        """Get servo state"""
        return (
            self.current_pan, self.current_tilt,
            0.0, 0.0,  # Simplified - no velocity tracking
            self.target_pan, self.target_tilt
        )
    
    def get_performance_stats(self):
        """Get performance statistics"""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        
        if elapsed > 0:
            moves_per_sec = self.move_count / elapsed
            return {
                'moves_per_second': moves_per_sec,
                'total_moves': self.move_count,
                'uptime': elapsed
            }
        return {}
    
    def shutdown(self):
        """Shutdown procedure"""
        if self.logger:
            self.logger.log_event('servo_shutdown', 'CPU-optimized servo controller shutting down')
        
        print("🛑 Shutting down CPU-optimized servo controller...")
        
        # Return to center
        try:
            self._move_servos_direct(90, 90)
            time.sleep(0.3)
        except:
            pass
        
        # Print performance stats
        stats = self.get_performance_stats()
        if stats:
            print(f"📊 Servo Performance: {stats['moves_per_second']:.1f} moves/sec, "
                  f"{stats['total_moves']} total moves")
        
        print("✅ Servo controller shutdown complete")

# ==============================================================================
# LIGHTWEIGHT TRACKING SYSTEM
# ==============================================================================

class LightweightTracker:
    def __init__(self, servo_controller, calibration_system, logger=None):
        self.servo = servo_controller
        self.calibration_system = calibration_system
        self.logger = logger
        
        # Frame properties
        self.frame_center_x = 640
        self.frame_center_y = 360
        self.frame_width = 1280
        self.frame_height = 720
        
        # Simplified distance calculation
        self.focal_length = 800.0  # Will be updated from calibration
        
        # Lightweight tracking state
        self.last_detection_time = time.time()
        self.lock_on_target = False
        self.tracking_confidence = 0.0
        
        # Reduced memory usage
        self.recent_distances = deque(maxlen=5)
        self.recent_positions = deque(maxlen=3)
        
        # Performance counters
        self.detections_processed = 0
        self.movements_sent = 0
        
        # Current detection
        self.current_detection_3d = None
        
        print("🎯 Lightweight tracker initialized")
    
    def update_frame_properties(self, width: int, height: int):
        """Update frame properties"""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.frame_center_x = width // 2
            self.frame_center_y = height // 2
            
            if self.logger:
                self.logger.log_event('resolution_change', f'Frame: {width}x{height}')
            
            print(f"📺 Frame: {width}x{height}")
    
    def track_person_lightweight(self, bbox, confidence: float, frame_count: int) -> bool:
        """Lightweight person tracking"""
        self.detections_processed += 1
        
        # Handle calibration
        if not self.calibration_system.is_completed():
            self.calibration_system.add_calibration_sample(
                bbox, confidence, self.frame_width, self.frame_height
            )
            # Update focal length if calibration completed
            if self.calibration_system.is_completed():
                self.focal_length = self.calibration_system.get_focal_length()
                # Scale for current resolution
                self.focal_length = self.focal_length * (self.frame_height / 480)
        
        # Skip processing if frame should be skipped
        if frame_count % (FRAME_SKIP_COUNT + 1) != 0:
            return False
        
        # Simple distance calculation
        person_height_pixels = bbox.height() * self.frame_height
        distance = (AVERAGE_PERSON_HEIGHT * self.focal_length) / person_height_pixels if person_height_pixels > 0 else 0
        distance = max(0.5, min(20.0, distance))  # Clamp distance
        
        # Smooth distance
        self.recent_distances.append(distance)
        smoothed_distance = np.median(self.recent_distances) if self.recent_distances else distance
        
        # Calculate 3D position (simplified)
        current_pan, current_tilt, _, _, _, _ = self.servo.get_enhanced_state()
        
        # Convert angles to radians
        pan_rad = math.radians(current_pan - 90)
        actual_tilt_angle = current_tilt + CAMERA_TILT_OFFSET
        tilt_rad = math.radians(90 - actual_tilt_angle)
        
        # Calculate 3D position
        horizontal_distance = smoothed_distance * math.cos(tilt_rad)
        x = horizontal_distance * math.sin(pan_rad)
        y = horizontal_distance * math.cos(pan_rad)
        z = smoothed_distance * math.sin(tilt_rad) + SERVO_MOUNT_HEIGHT
        
        # Create detection object
        self.current_detection_3d = Detection3D(
            x=x, y=y, z=z,
            distance=smoothed_distance,
            confidence=confidence,
            bbox_area=bbox.width() * bbox.height(),
            timestamp=time.time()
        )
        
        # Calculate bbox center
        center_x = (bbox.xmin() + bbox.width() * 0.5) * self.frame_width
        center_y = (bbox.ymin() + bbox.height() * 0.5) * self.frame_height
        
        # Store position for smoothing
        self.recent_positions.append((center_x, center_y))
        if len(self.recent_positions) > 1:
            # Simple position smoothing
            avg_x = np.mean([pos[0] for pos in self.recent_positions])
            avg_y = np.mean([pos[1] for pos in self.recent_positions])
            center_x, center_y = avg_x, avg_y
        
        # Calculate errors
        error_x = center_x - self.frame_center_x
        error_y = center_y - self.frame_center_y
        
        # Adaptive dead zone
        dead_zone = max(DEAD_ZONE, min(40, DEAD_ZONE * (smoothed_distance / 5.0)))
        
        # Check if movement needed
        if abs(error_x) > dead_zone or abs(error_y) > dead_zone:
            # Calculate movement with distance compensation
            distance_factor = min(1.5, max(0.5, 1.0 / smoothed_distance))
            confidence_factor = min(1.5, confidence + 0.3)
            
            # Simplified movement calculation
            pan_adjustment = -error_x * (PAN_SENSITIVITY / self.frame_width) * distance_factor * confidence_factor
            tilt_adjustment = error_y * (TILT_SENSITIVITY / self.frame_height) * distance_factor * confidence_factor
            
            # Calculate new targets with smoothing
            new_pan = current_pan + pan_adjustment * SMOOTHING_FACTOR
            new_tilt = current_tilt + tilt_adjustment * SMOOTHING_FACTOR
            
            # Send movement
            self.servo.move_to_optimized(new_pan, new_tilt)
            self.movements_sent += 1
            
            # Update tracking state
            if not self.lock_on_target:
                self.lock_on_target = True
                if self.logger:
                    self.logger.log_event('target_lock', f'Target locked: {smoothed_distance:.2f}m')
                print(f"🎯 Target locked: {smoothed_distance:.1f}m")
        
        # Update state
        self.tracking_confidence = min(1.0, confidence)
        self.last_detection_time = time.time()
        
        return True
    
    def handle_lost_target_lightweight(self, frame_count: int):
        """Lightweight lost target handling"""
        if self.lock_on_target and time.time() - self.last_detection_time > DETECTION_TIMEOUT:
            self.lock_on_target = False
            self.tracking_confidence = 0.0
            if self.logger:
                self.logger.log_event('target_lost', 'Target lost')
            print("🔍 Target lost")
    
    def is_tracking_active(self) -> bool:
        """Check if tracking is active"""
        return (time.time() - self.last_detection_time) < DETECTION_TIMEOUT
    
    def get_current_detection(self) -> Optional[Detection3D]:
        """Get current detection data"""
        return self.current_detection_3d
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'locked_on': self.lock_on_target,
            'confidence': self.tracking_confidence,
            'detections_processed': self.detections_processed,
            'movements_sent': self.movements_sent,
            'tracking_active': self.is_tracking_active()
        }

# ==============================================================================
# OPTIMIZED DATA LOGGER
# ==============================================================================

class OptimizedDataLogger:
    def __init__(self, log_dir="servo_logs", gps_handler=None):
        self.gps_handler = gps_handler
        script_dir = Path(__file__).resolve().parent
        self.log_dir = script_dir / log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.log_dir / f"optimized_servo_data_{timestamp}.csv"
        self.json_file = self.log_dir / f"optimized_session_{timestamp}.json"
        self.gps_csv_file = self.log_dir / f"optimized_gps_points_{timestamp}.csv"
        
        # Optimized CSV headers
        self.csv_headers = [
            'timestamp', 'frame_count', 'pan_angle', 'tilt_angle',
            'detection_confidence', 'tracking_confidence', 'person_detected',
            'tracking_active', 'lock_on_target', 'distance_meters',
            'x_position', 'y_position', 'z_position', 'calibration_status'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.csv_headers)
        
        # GPS headers
        self.gps_headers = [
            'timestamp', 'detection_lat', 'detection_lon', 'detection_alt',
            'vehicle_lat', 'vehicle_lon', 'vehicle_alt', 'relative_x', 'relative_y', 'relative_z',
            'confidence', 'distance', 'gps_accuracy'
        ]
        
        with open(self.gps_csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(self.gps_headers)
        
        # Session data
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
                'tracking_time': 0.0,
                'gps_points_created': 0,
                'calibration_samples': 0
            },
            'events': []
        }
        
        print(f"📊 Optimized data logging to: {self.log_dir}")
    
    def log_frame_data_optimized(self, frame_count, servo_state, detection_data, tracking_stats, calibration_status):
        """Optimized frame data logging"""
        try:
            current_time = time.time()
            
            # Unpack servo state
            pan_angle, tilt_angle, _, _, _, _ = servo_state
            
            # Initialize default values
            distance = x_pos = y_pos = z_pos = 0.0
            detection_confidence = tracking_confidence = 0.0
            person_detected = tracking_active = lock_on_target = False
            
            # Process detection data
            if detection_data:
                distance = detection_data.get('distance', 0.0)
                x_pos = detection_data.get('x_position', 0.0)
                y_pos = detection_data.get('y_position', 0.0)
                z_pos = detection_data.get('z_position', 0.0)
                detection_confidence = detection_data.get('confidence', 0.0)
                person_detected = True
                
                # Handle GPS waypoint creation
                if self.gps_handler and distance >= MIN_DISTANCE_FOR_GPS:
                    detection_3d = Detection3D(
                        x=x_pos, y=y_pos, z=z_pos,
                        distance=distance, confidence=detection_confidence,
                        bbox_area=detection_data.get('bbox_area', 0.0),
                        timestamp=current_time
                    )
                    
                    gps_point = self.gps_handler.add_detection_point(detection_3d)
                    
                    if gps_point:
                        self.session_data['statistics']['gps_points_created'] += 1
                        
                        # Log GPS point
                        with open(self.gps_csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                gps_point['timestamp'],
                                gps_point['latitude'],
                                gps_point['longitude'],
                                gps_point['altitude'],
                                gps_point['vehicle_lat'],
                                gps_point['vehicle_lon'],
                                gps_point['vehicle_altitude'],
                                gps_point['relative_x'],
                                gps_point['relative_y'],
                                gps_point['relative_z'],
                                gps_point['confidence'],
                                gps_point['distance'],
                                gps_point.get('gps_accuracy', 0.0)
                            ])
            
            # Process tracking stats
            if tracking_stats:
                tracking_confidence = tracking_stats.get('confidence', 0.0)
                tracking_active = tracking_stats.get('tracking_active', False)
                lock_on_target = tracking_stats.get('locked_on', False)
            
            # Write to CSV (only every 5th frame to reduce I/O)
            if frame_count % 5 == 0:
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        current_time, frame_count, pan_angle, tilt_angle,
                        detection_confidence, tracking_confidence, person_detected,
                        tracking_active, lock_on_target, distance,
                        x_pos, y_pos, z_pos, calibration_status
                    ])
            
            # Update session statistics
            self.session_data['statistics']['total_detections'] += person_detected
            if tracking_active:
                self.session_data['statistics']['tracking_time'] += 1/30.0
            
        except Exception as e:
            print(f"Optimized logging error: {e}")
    
    def log_event(self, event_type: str, description: str, metadata: Dict = None):
        """Log events"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'description': description,
            'metadata': metadata or {}
        }
        self.session_data['events'].append(event)
        print(f"📝 {event_type}: {description}")
    
    def finalize_session_optimized(self):
        """Finalize session with summary"""
        self.session_data['end_time'] = datetime.now().isoformat()
        
        if self.gps_handler:
            self.session_data['gps_status'] = self.gps_handler.get_enhanced_status()
        
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            # Print summary
            stats = self.session_data['statistics']
            print(f"\n📊 OPTIMIZED SESSION COMPLETE:")
            print(f"   Total detections: {stats['total_detections']}")
            print(f"   GPS waypoints: {stats['gps_points_created']}")
            print(f"   Log files: {self.log_dir}")
            
        except Exception as e:
            print(f"Session save error: {e}")

# ==============================================================================
# OPTIMIZED CALLBACK CLASS
# ==============================================================================

class OptimizedAppCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0
        self.processing_times = deque(maxlen=50)  # Reduced memory
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
    
    def new_function(self):
        return "Optimized High-Performance Tracking v3.0: "
    
    def update_performance_metrics(self, processing_time):
        """Update performance tracking metrics"""
        self.processing_times.append(processing_time)
        self.fps_counter += 1
        
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.processing_times:
            return {"fps": 0, "avg_processing_time": 0, "max_processing_time": 0}
        
        return {
            "fps": self.current_fps,
            "avg_processing_time": np.mean(self.processing_times),
            "max_processing_time": np.max(self.processing_times)
        }

# ==============================================================================
# MAIN OPTIMIZED CALLBACK FUNCTION
# ==============================================================================

def cpu_optimized_callback(pad, info, user_data):
    """CPU-optimized callback with reduced processing"""
    processing_start = time.time()
    
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    frame_count = user_data.get_count()
    
    # Less frequent frame property updates
    if frame_count % 60 == 0:  # Every 2 seconds at 30fps
        format, width, height = get_caps_from_pad(pad)
        if width and height:
            tracker.update_frame_properties(width, height)
    
    # Skip frame rendering for better performance
    frame = None
    if user_data.use_frame and frame_count % 3 == 0:  # Render every 3rd frame
        format, width, height = get_caps_from_pad(pad)
        if format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Extract detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Find best person detection (simplified)
    best_person = None
    best_score = 0
    
    for detection in detections:
        if detection.get_label() == "person":
            confidence = detection.get_confidence()
            if confidence >= MIN_CONFIDENCE:
                bbox = detection.get_bbox()
                area = bbox.width() * bbox.height()
                score = confidence * (1 + area)
                
                if score > best_score:
                    best_score = score
                    best_person = {'bbox': bbox, 'confidence': confidence}
    
    # Process tracking (lightweight)
    tracking_success = False
    if best_person:
        tracking_success = tracker.track_person_lightweight(
            best_person['bbox'], best_person['confidence'], frame_count
        )
        
        # Reduced status output frequency
        if frame_count % 90 == 0:  # Every 3 seconds
            detection_3d = tracker.get_current_detection()
            if detection_3d:
                print(f"🎯 Tracking: Conf {best_person['confidence']:.2f}, "
                      f"Distance: {detection_3d.distance:.2f}m, "
                      f"Position: ({detection_3d.x:.1f}, {detection_3d.y:.1f}, {detection_3d.z:.1f})")
    else:
        tracker.handle_lost_target_lightweight(frame_count)
    
    # Simplified frame drawing
    if user_data.use_frame and frame is not None:
        _draw_lightweight_overlay(frame, tracker, calibration_system, mavlink_handler, 
                                best_person, frame_count, user_data)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    # Reduced logging frequency
    if frame_count % 10 == 0:  # Log every 10th frame
        servo_state = servo_controller.get_enhanced_state()
        detection_data = None
        if best_person and tracking_success:
            detection_3d = tracker.get_current_detection()
            if detection_3d:
                detection_data = {
                    'distance': detection_3d.distance,
                    'x_position': detection_3d.x,
                    'y_position': detection_3d.y,
                    'z_position': detection_3d.z,
                    'confidence': best_person['confidence'],
                    'bbox_area': detection_3d.bbox_area
                }
        
        tracking_stats = tracker.get_tracking_stats()
        calibration_status = calibration_system.get_status()
        
        data_logger.log_frame_data_optimized(
            frame_count, servo_state, detection_data, tracking_stats, calibration_status
        )
    
    # Update performance metrics
    processing_time = time.time() - processing_start
    user_data.update_performance_metrics(processing_time)
    
    return Gst.PadProbeReturn.OK

def _draw_lightweight_overlay(frame, tracker, calibration_system, mavlink_handler, 
                            best_person, frame_count, user_data):
    """Lightweight overlay with minimal drawing"""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Simple crosshair
    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 255), 2)
    
    # Calibration status
    if not calibration_system.is_completed():
        status = calibration_system.get_status()
        cv2.putText(frame, "CALIBRATION MODE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, status[:50], (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Detection info
    if best_person:
        detection_3d = tracker.get_current_detection()
        if detection_3d:
            # Main tracking info
            cv2.putText(frame, f"TRACKING: {detection_3d.distance:.2f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Confidence
            cv2.putText(frame, f"Conf: {best_person['confidence']:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Simple bounding box
        bbox = best_person['bbox']
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int((bbox.xmin() + bbox.width()) * width)
        y2 = int((bbox.ymin() + bbox.height()) * height)
        
        # Dynamic color based on confidence
        color = (0, int(255 * best_person['confidence']), int(255 * (1 - best_person['confidence'])))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Distance label
        if detection_3d:
            cv2.putText(frame, f"{detection_3d.distance:.1f}m", 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Servo information (simplified)
    servo_state = servo_controller.get_enhanced_state()
    pan, tilt, _, _, _, _ = servo_state
    
    cv2.putText(frame, f"Pan: {pan:.1f}° Tilt: {tilt:.1f}°", 
               (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # GPS status (if available)
    if mavlink_handler and mavlink_handler.connected:
        status = mavlink_handler.get_enhanced_status()
        gps_color = (0, 255, 0) if status['gps_fix'] >= 3 else (0, 0, 255)
        gps_text = f"GPS: {status['satellites']} sats"
        cv2.putText(frame, gps_text, (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 2)
        
        # Waypoint counter
        cv2.putText(frame, f"Waypoints: {status['points_logged']}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Performance info (reduced frequency)
    if frame_count % 30 == 0:
        perf_stats = user_data.get_performance_stats()
        cv2.putText(frame, f"FPS: {perf_stats['fps']}", 
                   (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# ==============================================================================
# SIGNAL HANDLER FOR CLEAN SHUTDOWN
# ==============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n🛑 Received signal {signum}. Initiating clean shutdown...")
    global shutdown_requested
    shutdown_requested = True

# ==============================================================================
# MAIN INITIALIZATION AND EXECUTION
# ==============================================================================

# Global shutdown flag
shutdown_requested = False

def main():
    """Main execution function with optimized error handling"""
    global servo_controller, tracker, calibration_system, mavlink_handler, data_logger
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Initializing ENHANCED servo tracking system v3.0...")
    print("   ⚡ CPU-optimized processing")
    print("   🎯 Improved automatic calibration")
    print("   🛰️ Fixed GPS waypoint generation")
    print("   📊 Optimized data logging")
    
    # Initialize components in order
    try:
        # 1. MAVLink GPS Handler (with fixed connection management)
        print("\n🛰️ Initializing fixed MAVLink GPS handler...")
        mavlink_handler = None
        try:
            mavlink_handler = FixedMAVLinkGPSHandler()
            if mavlink_handler.connected:
                print("✅ MAVLink GPS handler connected successfully")
            else:
                print("⚠️  MAVLink connection failed - continuing without GPS")
        except Exception as e:
            print(f"⚠️  MAVLink initialization failed: {e}")
            print("   Continuing without GPS waypoint functionality")
            mavlink_handler = None
        
        # 2. Optimized Data Logger
        print("\n📊 Initializing optimized data logger...")
        data_logger = OptimizedDataLogger(gps_handler=mavlink_handler)
        print("✅ Optimized data logger ready")
        
        # 3. Improved Auto-Calibration System
        print("\n🎯 Initializing improved automatic calibration system...")
        calibration_system = ImprovedAutoCalibrationSystem(data_logger)
        print("✅ Improved auto-calibration system ready")
        
        # 4. CPU-Optimized Servo Controller
        print("\n⚡ Initializing CPU-optimized servo controller...")
        servo_controller = CPUOptimizedServoController(data_logger)
        print("✅ CPU-optimized servo controller ready")
        
        # 5. Lightweight Tracker
        print("\n🎯 Initializing lightweight tracker...")
        tracker = LightweightTracker(servo_controller, calibration_system, data_logger)
        print("✅ Lightweight tracker ready")
        
        # 6. GStreamer Application
        print("\n📹 Setting up optimized GStreamer detection pipeline...")
        project_root = Path(__file__).resolve().parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            os.environ["HAILO_ENV_FILE"] = str(env_file)
        
        user_data = OptimizedAppCallback()
        app = GStreamerDetectionApp(cpu_optimized_callback, user_data)
        
        print("\n🎬 OPTIMIZED SYSTEM READY!")
        
        # Display system status
        if mavlink_handler and mavlink_handler.connected:
            status = mavlink_handler.get_enhanced_status()
            print(f"\n🛰️ GPS Status:")
            print(f"   Connection: {'Connected' if status['connected'] else 'Disconnected'}")
            print(f"   Fix Type: {status['gps_fix']} ({'3D' if status['gps_fix'] >= 3 else '2D/No Fix'})")
            print(f"   Satellites: {status['satellites']}")
            print(f"   Accuracy: {status['gps_accuracy']:.1f}m HDOP")
            print(f"   Position: {status['latitude']:.6f}, {status['longitude']:.6f}")
            print(f"   Best Altitude: {status['best_altitude']:.1f}m")
        
        print(f"\n📁 Data logging to: {data_logger.log_dir}")
        print("\n🎯 Improved auto-calibration will start when person detected")
        print("📏 Follow prompts for high-quality calibration")
        print("\n⚡ System optimizations:")
        print(f"   Servo update rate: {SERVO_UPDATE_RATE}Hz (reduced from 100Hz)")
        print(f"   Frame processing: Every {FRAME_SKIP_COUNT + 1} frames")
        print(f"   GPS update interval: {GPS_UPDATE_INTERVAL}s")
        print(f"   Waypoint creation distance: >{MIN_DISTANCE_FOR_GPS}m")
        print(f"   Memory usage: Optimized with smaller buffers")
        
        print("\n🚀 STARTING OPTIMIZED TRACKING...")
        print("Press Ctrl+C to stop")
        
        # Start the main application
        try:
            app.run()
        except KeyboardInterrupt:
            print("\n🛑 Interrupt received - shutting down...")
        except Exception as e:
            print(f"\n❌ Application error: {e}")
            data_logger.log_event('error', f'Application error: {str(e)}')
            raise
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        return 1
    
    finally:
        # Optimized shutdown procedure
        print("\n🔄 Initiating optimized shutdown procedure...")
        
        try:
            # 1. Finalize data logging
            if 'data_logger' in locals():
                print("📊 Finalizing optimized data logs...")
                data_logger.finalize_session_optimized()
            
            # 2. Shutdown servo controller
            if 'servo_controller' in locals():
                print("⚡ Shutting down servo controller...")
                servo_controller.shutdown()
            
            # 3. Close MAVLink connection
            if 'mavlink_handler' in locals() and mavlink_handler:
                print("🛰️ Closing MAVLink connection...")
                mavlink_handler.shutdown()
            
            # 4. Print final statistics
            if 'tracker' in locals():
                stats = tracker.get_tracking_stats()
                print(f"\n📈 Final Statistics:")
                print(f"   Detections processed: {stats['detections_processed']}")
                print(f"   Servo movements sent: {stats['movements_sent']}")
                print(f"   Final tracking status: {'Active' if stats['tracking_active'] else 'Inactive'}")
            
            if 'servo_controller' in locals():
                servo_stats = servo_controller.get_performance_stats()
                if servo_stats:
                    print(f"   Servo performance: {servo_stats['moves_per_second']:.1f} moves/sec")
            
        except Exception as e:
            print(f"⚠️  Shutdown error: {e}")
        
        print("✅ Optimized shutdown complete")
        return 0

# ==============================================================================
# PERFORMANCE MONITORING (LIGHTWEIGHT)
# ==============================================================================

class LightweightPerformanceMonitor:
    """Lightweight performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)  # Reduced from 1000
        self.monitoring = True
        
        # Try to import psutil for CPU monitoring
        try:
            import psutil
            self.process = psutil.Process()
            self.cpu_monitoring = True
        except ImportError:
            self.cpu_monitoring = False
            print("⚠️  psutil not available - CPU monitoring disabled")
    
    def log_frame_time(self, frame_time):
        """Log frame processing time"""
        self.frame_times.append(frame_time)
    
    def get_stats(self):
        """Get current performance statistics"""
        stats = {
            'uptime': time.time() - self.start_time,
            'frames_processed': len(self.frame_times)
        }
        
        if self.frame_times:
            stats.update({
                'avg_frame_time': np.mean(self.frame_times),
                'max_frame_time': np.max(self.frame_times),
                'fps_estimate': 1.0 / np.mean(self.frame_times) if np.mean(self.frame_times) > 0 else 0
            })
        
        if self.cpu_monitoring:
            try:
                stats['cpu_percent'] = self.process.cpu_percent()
                stats['memory_mb'] = self.process.memory_info().rss / 1024 / 1024
            except:
                pass
        
        return stats

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def check_optimized_system_requirements():
    """Check system requirements for optimized version"""
    print("🔍 Checking optimized system requirements...")
    
    requirements_met = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8+ required, found {python_version}")
        requirements_met = False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}")
    
    # Check required packages
    required_packages = [
        'numpy', 'cv2', 'hailo', 'board', 'busio', 
        'adafruit_motor', 'adafruit_pca9685', 'pymavlink'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            requirements_met = False
    
    # Check I2C interface
    try:
        import board
        import busio
        i2c = busio.I2C(board.SCL, board.SDA)
        i2c.deinit()
        print("✅ I2C interface available")
    except Exception as e:
        print(f"❌ I2C interface issue: {e}")
        requirements_met = False
    
    # Check serial port
    if Path(MAVLINK_CONNECTION).exists():
        print(f"✅ Serial port {MAVLINK_CONNECTION} available")
    else:
        print(f"⚠️  Serial port {MAVLINK_CONNECTION} not found")
        print("   GPS waypoint functionality will be disabled")
    
    return requirements_met

def print_optimized_system_info():
    """Print optimized system information"""
    print("\n" + "="*60)
    print("🔧 ENHANCED SERVO TRACKING SYSTEM v3.0 - OPTIMIZED")
    print("="*60)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {sys.platform}")
    print(f"📁 Working directory: {Path.cwd()}")
    print("="*60)
    
    print("\n🎛️  OPTIMIZED CONFIGURATION:")
    print(f"   Servo update rate: {SERVO_UPDATE_RATE}Hz (reduced for stability)")
    print(f"   Detection confidence: {MIN_CONFIDENCE}")
    print(f"   Frame skip count: {FRAME_SKIP_COUNT} (process every {FRAME_SKIP_COUNT + 1} frames)")
    print(f"   GPS update interval: {GPS_UPDATE_INTERVAL}s")
    print(f"   Waypoint distance: >{MIN_DISTANCE_FOR_GPS}m")
    print(f"   Calibration samples: {AUTO_CALIBRATION_SAMPLES}")
    print(f"   Calibration confidence: >{CALIBRATION_MIN_CONFIDENCE}")
    
    print("\n🔧 HARDWARE SETUP:")
    print(f"   Servo mount height: {SERVO_MOUNT_HEIGHT}m")
    print(f"   Camera tilt offset: {CAMERA_TILT_OFFSET}°")
    print(f"   MAVLink connection: {MAVLINK_CONNECTION} @ {MAVLINK_BAUD} baud")
    
    print("\n⚡ PERFORMANCE OPTIMIZATIONS:")
    print(f"   Reduced memory usage: Smaller deques and buffers")
    print(f"   CPU optimization: Throttled servo updates")
    print(f"   I/O optimization: Reduced logging frequency")
    print(f"   Frame processing: Selective rendering")

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Print optimized system information
    print_optimized_system_info()
    
    # Check system requirements
    if not check_optimized_system_requirements():
        print("\n❌ System requirements not met. Please install missing packages.")
        sys.exit(1)
    
    # Start lightweight performance monitoring
    perf_monitor = LightweightPerformanceMonitor()
    
    try:
        # Run main application
        exit_code = main()
        
        # Print final performance stats
        final_stats = perf_monitor.get_stats()
        print(f"\n📊 OPTIMIZED PERFORMANCE SUMMARY:")
        print(f"   Uptime: {final_stats['uptime']:.1f}s")
        print(f"   Frames processed: {final_stats['frames_processed']}")
        if 'fps_estimate' in final_stats:
            print(f"   Estimated FPS: {final_stats['fps_estimate']:.1f}")
            print(f"   Avg frame time: {final_stats['avg_frame_time']*1000:.1f}ms")
        if 'cpu_percent' in final_stats:
            print(f"   CPU usage: {final_stats['cpu_percent']:.1f}%")
        if 'memory_mb' in final_stats:
            print(f"   Memory usage: {final_stats['memory_mb']:.1f}MB")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        print("\n👋 Optimized system shutdown complete!")

# ==============================================================================
# END OF ENHANCED SERVO TRACKING SYSTEM v3.0
# ==============================================================================