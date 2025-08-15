#!/usr/bin/env python3
"""
Camera Calibration System for Person Detection Distance Calculation

This module provides comprehensive camera calibration tools to determine:
- Focal length in pixels
- Horizontal Field of View (FOV)
- Vertical Field of View (FOV)
- Lens distortion parameters (optional)

Usage:
1. Run calibration with a person at known distances
2. Use checkerboard pattern for precise calibration
3. Save calibration parameters to config file
4. Load calibration in main detection system

Author: AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import json
import os
import time
import math
from datetime import datetime
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------------------------------------------------------------------------
# CALIBRATION CONFIGURATION
# -----------------------------------------------------------------------------------------------

# Default person measurements (adjust for your setup)
DEFAULT_PERSON_HEIGHT = 1.7  # meters (5'7")
DEFAULT_PERSON_WIDTH = 0.45  # meters (shoulder width)

# Calibration distances for multi-point calibration
CALIBRATION_DISTANCES = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]  # meters

# Video capture settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_RESOLUTION = (1920, 1080)  # Full HD
CAPTURE_FPS = 30

# Checkerboard calibration settings
CHECKERBOARD_SIZE = (9, 6)  # Internal corners (width, height)
CHECKERBOARD_SQUARE_SIZE = 0.025  # 2.5cm squares

# UI Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (0, 255, 0)
THICKNESS = 2

# Output configuration
CONFIG_FILE = "camera_calibration.json"
CALIBRATION_IMAGES_DIR = "calibration_images"

class CameraCalibrator:
    """Comprehensive camera calibration system"""
    
    def __init__(self, camera_index=DEFAULT_CAMERA_INDEX, resolution=DEFAULT_RESOLUTION):
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        
        # Calibration data storage
        self.calibration_data = {
            'focal_length_pixels': None,
            'horizontal_fov': None,
            'vertical_fov': None,
            'camera_matrix': None,
            'distortion_coefficients': None,
            'resolution': resolution,
            'calibration_date': None,
            'calibration_method': None,
            'measurements': []
        }
        
        # Measurement storage for analysis
        self.distance_measurements = []
        self.pixel_measurements = []
        self.fov_measurements = []
        
        # UI state
        self.current_instruction = ""
        self.measuring_active = False
        self.roi_start = None
        self.roi_end = None
        
        # Create output directory
        Path(CALIBRATION_IMAGES_DIR).mkdir(exist_ok=True)
    
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
            
            # Get actual resolution (may differ from requested)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (actual_width, actual_height)
            self.calibration_data['resolution'] = self.resolution
            
            print(f"✅ Camera initialized: {actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def run_person_calibration(self, person_height=DEFAULT_PERSON_HEIGHT):
        """Interactive person-based calibration"""
        print("\n🎯 PERSON-BASED CALIBRATION")
        print("=" * 50)
        print("Instructions:")
        print("1. Have a person stand at marked distances")
        print("2. Click and drag to measure person height in pixels")
        print("3. Press 's' to save measurement")
        print("4. Press 'n' for next distance")
        print("5. Press 'q' to quit")
        print("6. Press 'c' to calculate results")
        
        if not self.initialize_camera():
            return False
        
        cv2.namedWindow('Person Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Person Calibration', self._mouse_callback)
        
        distance_index = 0
        current_distance = CALIBRATION_DISTANCES[distance_index]
        measurements_at_distance = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Draw UI
            display_frame = frame.copy()
            self._draw_calibration_ui(display_frame, current_distance, len(measurements_at_distance))
            
            # Draw measurement ROI if active
            if self.roi_start and self.roi_end:
                cv2.rectangle(display_frame, self.roi_start, self.roi_end, (0, 255, 0), 2)
                height_pixels = abs(self.roi_end[1] - self.roi_start[1])
                cv2.putText(display_frame, f"Height: {height_pixels}px", 
                           (10, 120), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
            
            cv2.imshow('Person Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and self.roi_start and self.roi_end:
                # Save measurement
                height_pixels = abs(self.roi_end[1] - self.roi_start[1])
                measurement = {
                    'distance_meters': current_distance,
                    'height_pixels': height_pixels,
                    'person_height_meters': person_height,
                    'timestamp': time.time()
                }
                measurements_at_distance.append(measurement)
                self.calibration_data['measurements'].append(measurement)
                
                print(f"📏 Measurement saved: {current_distance}m = {height_pixels}px")
                
                # Reset ROI
                self.roi_start = None
                self.roi_end = None
                
            elif key == ord('n'):
                # Next distance
                if measurements_at_distance:
                    distance_index = (distance_index + 1) % len(CALIBRATION_DISTANCES)
                    current_distance = CALIBRATION_DISTANCES[distance_index]
                    measurements_at_distance = []
                    print(f"📐 Move person to {current_distance}m distance")
                else:
                    print("❌ Take at least one measurement before moving to next distance")
                    
            elif key == ord('c'):
                # Calculate calibration
                if len(self.calibration_data['measurements']) >= 3:
                    self._calculate_person_calibration(person_height)
                else:
                    print("❌ Need at least 3 measurements for calibration")
        
        cv2.destroyAllWindows()
        return True
    
    def run_checkerboard_calibration(self):
        """Professional checkerboard-based calibration"""
        print("\n🏁 CHECKERBOARD CALIBRATION")
        print("=" * 50)
        print("Instructions:")
        print("1. Print a checkerboard pattern (9x6 internal corners)")
        print("2. Hold it at various angles and distances")
        print("3. Press 's' to capture calibration image")
        print("4. Capture 15-20 good images")
        print("5. Press 'c' to calculate calibration")
        print("6. Press 'q' to quit")
        
        if not self.initialize_camera():
            return False
        
        # Prepare object points
        objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= CHECKERBOARD_SQUARE_SIZE
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        captured_images = 0
        
        cv2.namedWindow('Checkerboard Calibration', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = frame.copy()
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
            
            if ret_corners:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners
                cv2.drawChessboardCorners(display_frame, CHECKERBOARD_SIZE, corners2, ret_corners)
                
                # Show ready to capture
                cv2.putText(display_frame, "READY - Press 's' to capture", 
                           (10, 30), FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
            else:
                cv2.putText(display_frame, "Position checkerboard in view", 
                           (10, 30), FONT, FONT_SCALE, (0, 0, 255), THICKNESS)
            
            # Show status
            cv2.putText(display_frame, f"Captured: {captured_images}", 
                       (10, 60), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
            cv2.putText(display_frame, "Press 'c' to calibrate (need 15+ images)", 
                       (10, 90), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
            
            cv2.imshow('Checkerboard Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and ret_corners:
                # Capture calibration image
                objpoints.append(objp)
                imgpoints.append(corners2)
                captured_images += 1
                
                # Save calibration image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"{CALIBRATION_IMAGES_DIR}/checkerboard_{timestamp}.jpg"
                cv2.imwrite(img_path, frame)
                
                print(f"📸 Captured image {captured_images}: {img_path}")
                
                # Brief flash feedback
                cv2.rectangle(display_frame, (0, 0), self.resolution, (255, 255, 255), -1)
                cv2.imshow('Checkerboard Calibration', display_frame)
                cv2.waitKey(200)
                
            elif key == ord('c') and captured_images >= 10:
                # Calculate calibration
                print("🔄 Calculating camera calibration...")
                ret_cal, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                
                if ret_cal:
                    self._save_checkerboard_calibration(camera_matrix, dist_coeffs, ret_cal)
                    print("✅ Checkerboard calibration completed!")
                else:
                    print("❌ Calibration failed")
        
        cv2.destroyAllWindows()
        return True
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
            self.roi_end = None
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_start:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.roi_start:
                self.roi_end = (x, y)
    
    def _draw_calibration_ui(self, frame, current_distance, measurement_count):
        """Draw calibration UI overlay"""
        # Background for text
        cv2.rectangle(frame, (5, 5), (500, 150), (0, 0, 0), -1)
        
        # Instructions
        cv2.putText(frame, f"Distance: {current_distance}m", 
                   (10, 30), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
        cv2.putText(frame, f"Measurements: {measurement_count}", 
                   (10, 60), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
        cv2.putText(frame, "Click & drag to measure person height", 
                   (10, 90), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
        cv2.putText(frame, "s=save, n=next distance, c=calculate, q=quit", 
                   (10, 120), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
        
        # Center crosshair
        center_x, center_y = self.resolution[0] // 2, self.resolution[1] // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)
    
    def _calculate_person_calibration(self, person_height):
        """Calculate calibration parameters from person measurements"""
        if len(self.calibration_data['measurements']) < 3:
            print("❌ Need at least 3 measurements")
            return False
        
        print("\n🔄 Calculating calibration parameters...")
        
        # Extract data for analysis
        distances = []
        pixel_heights = []
        
        for measurement in self.calibration_data['measurements']:
            distances.append(measurement['distance_meters'])
            pixel_heights.append(measurement['height_pixels'])
        
        distances = np.array(distances)
        pixel_heights = np.array(pixel_heights)
        
        # Calculate focal length using: f = (pixel_size * distance) / real_size
        # Rearranged: f = pixel_height * distance / person_height
        focal_lengths = pixel_heights * distances / person_height
        
        # Use median focal length for robustness
        focal_length = np.median(focal_lengths)
        
        # Calculate FOV from focal length
        width, height = self.resolution
        horizontal_fov = 2 * math.degrees(math.atan(width / (2 * focal_length)))
        vertical_fov = 2 * math.degrees(math.atan(height / (2 * focal_length)))
        
        # Store results
        self.calibration_data.update({
            'focal_length_pixels': float(focal_length),
            'horizontal_fov': float(horizontal_fov),
            'vertical_fov': float(vertical_fov),
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'person_measurement',
            'person_height_used': person_height
        })
        
        # Print results
        print(f"📊 CALIBRATION RESULTS:")
        print(f"   Focal Length: {focal_length:.1f} pixels")
        print(f"   Horizontal FOV: {horizontal_fov:.1f}°")
        print(f"   Vertical FOV: {vertical_fov:.1f}°")
        print(f"   Resolution: {width}x{height}")
        
        # Calculate accuracy metrics
        predicted_pixels = (person_height * focal_length) / distances
        errors = np.abs(predicted_pixels - pixel_heights)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"   Mean Error: {mean_error:.1f} pixels")
        print(f"   Max Error: {max_error:.1f} pixels")
        
        # Save calibration
        self.save_calibration()
        
        # Show accuracy plot
        self._plot_calibration_accuracy(distances, pixel_heights, predicted_pixels)
        
        return True
    
    def _save_checkerboard_calibration(self, camera_matrix, dist_coeffs, reprojection_error):
        """Save checkerboard calibration results"""
        focal_length_x = camera_matrix[0, 0]
        focal_length_y = camera_matrix[1, 1]
        focal_length = (focal_length_x + focal_length_y) / 2  # Average
        
        width, height = self.resolution
        horizontal_fov = 2 * math.degrees(math.atan(width / (2 * focal_length_x)))
        vertical_fov = 2 * math.degrees(math.atan(height / (2 * focal_length_y)))
        
        self.calibration_data.update({
            'focal_length_pixels': float(focal_length),
            'horizontal_fov': float(horizontal_fov),
            'vertical_fov': float(vertical_fov),
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'reprojection_error': float(reprojection_error),
            'calibration_date': datetime.now().isoformat(),
            'calibration_method': 'checkerboard'
        })
        
        print(f"📊 CHECKERBOARD CALIBRATION RESULTS:")
        print(f"   Focal Length: {focal_length:.1f} pixels")
        print(f"   Horizontal FOV: {horizontal_fov:.1f}°")
        print(f"   Vertical FOV: {vertical_fov:.1f}°")
        print(f"   Reprojection Error: {reprojection_error:.3f}")
        
        self.save_calibration()
    
    def _plot_calibration_accuracy(self, distances, actual_pixels, predicted_pixels):
        """Plot calibration accuracy"""
        try:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(distances, actual_pixels, color='blue', label='Measured', s=50)
            plt.plot(distances, predicted_pixels, color='red', label='Predicted', linewidth=2)
            plt.xlabel('Distance (m)')
            plt.ylabel('Pixel Height')
            plt.title('Calibration Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            errors = np.abs(predicted_pixels - actual_pixels)
            plt.bar(range(len(errors)), errors)
            plt.xlabel('Measurement #')
            plt.ylabel('Error (pixels)')
            plt.title('Measurement Errors')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{CALIBRATION_IMAGES_DIR}/calibration_accuracy.png", dpi=150)
            plt.show()
            
        except Exception as e:
            print(f"Could not create accuracy plot: {e}")
    
    def save_calibration(self, filename=CONFIG_FILE):
        """Save calibration data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
            print(f"💾 Calibration saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filename=CONFIG_FILE):
        """Load calibration data from JSON file"""
        try:
            if not Path(filename).exists():
                print(f"❌ Calibration file {filename} not found")
                return False
            
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            
            print(f"📂 Calibration loaded from {filename}")
            self._print_calibration_summary()
            return True
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False
    
    def _print_calibration_summary(self):
        """Print calibration summary"""
        data = self.calibration_data
        print("\n📊 LOADED CALIBRATION:")
        print(f"   Method: {data.get('calibration_method', 'Unknown')}")
        print(f"   Date: {data.get('calibration_date', 'Unknown')}")
        print(f"   Focal Length: {data.get('focal_length_pixels', 'N/A')} pixels")
        print(f"   Horizontal FOV: {data.get('horizontal_fov', 'N/A')}°")
        print(f"   Vertical FOV: {data.get('vertical_fov', 'N/A')}°")
        print(f"   Resolution: {data.get('resolution', 'Unknown')}")
    
    def test_distance_calculation(self, person_height=DEFAULT_PERSON_HEIGHT):
        """Test distance calculation with current calibration"""
        if not self.calibration_data.get('focal_length_pixels'):
            print("❌ No calibration data loaded")
            return False
        
        print("\n🎯 DISTANCE CALCULATION TEST")
        print("Click and drag to measure person height")
        print("Press 'q' to quit")
        
        if not self.initialize_camera():
            return False
        
        cv2.namedWindow('Distance Test', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Distance Test', self._mouse_callback)
        
        focal_length = self.calibration_data['focal_length_pixels']
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Draw measurement ROI and calculate distance
            if self.roi_start and self.roi_end:
                cv2.rectangle(display_frame, self.roi_start, self.roi_end, (0, 255, 0), 2)
                height_pixels = abs(self.roi_end[1] - self.roi_start[1])
                
                if height_pixels > 0:
                    # Calculate distance: distance = (real_height * focal_length) / pixel_height
                    distance = (person_height * focal_length) / height_pixels
                    
                    cv2.putText(display_frame, f"Height: {height_pixels}px", 
                               (10, 30), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
                    cv2.putText(display_frame, f"Distance: {distance:.2f}m", 
                               (10, 60), FONT, FONT_SCALE, (0, 255, 255), THICKNESS)
            
            cv2.putText(display_frame, "Click & drag to measure distance", 
                       (10, self.resolution[1] - 20), FONT, FONT_SCALE, FONT_COLOR, THICKNESS)
            
            cv2.imshow('Distance Test', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main calibration interface"""
    parser = argparse.ArgumentParser(description='Camera Calibration System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--resolution', type=str, default='1920x1080', help='Camera resolution (WxH)')
    parser.add_argument('--height', type=float, default=DEFAULT_PERSON_HEIGHT, help='Person height in meters')
    parser.add_argument('--load', action='store_true', help='Load existing calibration')
    parser.add_argument('--test', action='store_true', help='Test distance calculation')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        resolution = DEFAULT_RESOLUTION
    
    # Create calibrator
    calibrator = CameraCalibrator(args.camera, resolution)
    
    try:
        if args.load:
            calibrator.load_calibration()
            
        if args.test:
            calibrator.test_distance_calculation(args.height)
        else:
            # Main calibration menu
            print("\n🎥 CAMERA CALIBRATION SYSTEM")
            print("=" * 50)
            print("Choose calibration method:")
            print("1. Person-based calibration (simple, good accuracy)")
            print("2. Checkerboard calibration (professional, highest accuracy)")
            print("3. Load existing calibration")
            print("4. Test distance calculation")
            print("5. Quit")
            
            while True:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    calibrator.run_person_calibration(args.height)
                elif choice == '2':
                    calibrator.run_checkerboard_calibration()
                elif choice == '3':
                    calibrator.load_calibration()
                elif choice == '4':
                    calibrator.test_distance_calculation(args.height)
                elif choice == '5':
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
    
    finally:
        calibrator.cleanup()
    
    print("\n✅ Calibration system closed")

if __name__ == "__main__":
    main()
