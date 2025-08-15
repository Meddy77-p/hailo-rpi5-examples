#!/usr/bin/env python3
"""
Camera Configuration Loader

Simple utility to load camera calibration parameters for the person detection system.
Integrates with the main detection code to provide accurate distance calculations.

Usage:
    from camera_config_loader import load_camera_config, CameraConfig
    
    config = load_camera_config()
    focal_length = config.focal_length_pixels
    h_fov = config.horizontal_fov

Author: AI Assistant
Date: 2025
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

# Default configuration file
DEFAULT_CONFIG_FILE = "camera_calibration.json"

# Fallback values if no calibration available (typical webcam values)
FALLBACK_CONFIG = {
    'focal_length_pixels': 452,
    'horizontal_fov': 70.6,
    'vertical_fov': 55.9,
    'resolution': [1920, 1080],
    'camera_matrix': None,
    'distortion_coefficients': None
}

@dataclass
class CameraConfig:
    """Camera configuration data structure"""
    focal_length_pixels: float
    horizontal_fov: float  # degrees
    vertical_fov: float   # degrees
    resolution: tuple     # (width, height)
    camera_matrix: Optional[np.ndarray] = None
    distortion_coefficients: Optional[np.ndarray] = None
    calibration_method: Optional[str] = None
    calibration_date: Optional[str] = None
    reprojection_error: Optional[float] = None
    
    def __post_init__(self):
        """Convert lists to numpy arrays if needed"""
        if self.camera_matrix is not None and not isinstance(self.camera_matrix, np.ndarray):
            self.camera_matrix = np.array(self.camera_matrix)
        
        if self.distortion_coefficients is not None and not isinstance(self.distortion_coefficients, np.ndarray):
            self.distortion_coefficients = np.array(self.distortion_coefficients)
    
    @property
    def width(self) -> int:
        """Image width in pixels"""
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        """Image height in pixels"""
        return self.resolution[1]
    
    @property
    def center_x(self) -> float:
        """Image center X coordinate"""
        return self.width / 2.0
    
    @property
    def center_y(self) -> float:
        """Image center Y coordinate"""
        return self.height / 2.0
    
    @property
    def has_distortion_correction(self) -> bool:
        """Check if distortion correction parameters are available"""
        return (self.camera_matrix is not None and 
                self.distortion_coefficients is not None)
    
    def calculate_distance(self, pixel_height: float, real_height: float) -> float:
        """
        Calculate distance to object based on pixel height
        
        Args:
            pixel_height: Height of object in pixels
            real_height: Real height of object in meters
            
        Returns:
            Distance to object in meters
        """
        if pixel_height <= 0:
            return float('inf')
        
        return (real_height * self.focal_length_pixels) / pixel_height
    
    def calculate_pixel_height(self, distance: float, real_height: float) -> float:
        """
        Calculate expected pixel height for object at given distance
        
        Args:
            distance: Distance to object in meters
            real_height: Real height of object in meters
            
        Returns:
            Expected pixel height
        """
        if distance <= 0:
            return 0
        
        return (real_height * self.focal_length_pixels) / distance
    
    def pixel_to_angle(self, pixel_offset: float, axis: str = 'horizontal') -> float:
        """
        Convert pixel offset from center to angle
        
        Args:
            pixel_offset: Offset from image center in pixels
            axis: 'horizontal' or 'vertical'
            
        Returns:
            Angle in degrees
        """
        if axis.lower() == 'horizontal':
            fov = self.horizontal_fov
            dimension = self.width
        else:
            fov = self.vertical_fov
            dimension = self.height
        
        # Convert pixel offset to angle
        angle_per_pixel = fov / dimension
        return pixel_offset * angle_per_pixel
    
    def angle_to_pixel(self, angle: float, axis: str = 'horizontal') -> float:
        """
        Convert angle to pixel offset from center
        
        Args:
            angle: Angle in degrees
            axis: 'horizontal' or 'vertical'
            
        Returns:
            Pixel offset from center
        """
        if axis.lower() == 'horizontal':
            fov = self.horizontal_fov
            dimension = self.width
        else:
            fov = self.vertical_fov
            dimension = self.height
        
        # Convert angle to pixel offset
        pixels_per_angle = dimension / fov
        return angle * pixels_per_angle
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort image points using camera calibration
        
        Args:
            points: Array of image points (N, 1, 2)
            
        Returns:
            Undistorted points
        """
        if not self.has_distortion_correction:
            return points
        
        import cv2
        return cv2.undistortPoints(points, self.camera_matrix, self.distortion_coefficients)
    
    def print_summary(self):
        """Print configuration summary"""
        print(f"📷 Camera Configuration:")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   Focal Length: {self.focal_length_pixels:.1f} pixels")
        print(f"   Horizontal FOV: {self.horizontal_fov:.1f}°")
        print(f"   Vertical FOV: {self.vertical_fov:.1f}°")
        if self.calibration_method:
            print(f"   Calibration Method: {self.calibration_method}")
        if self.calibration_date:
            print(f"   Calibration Date: {self.calibration_date}")
        if self.reprojection_error:
            print(f"   Reprojection Error: {self.reprojection_error:.3f}")
        print(f"   Distortion Correction: {'Yes' if self.has_distortion_correction else 'No'}")


def load_camera_config(config_file: str = DEFAULT_CONFIG_FILE) -> CameraConfig:
    """
    Load camera configuration from JSON file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        CameraConfig object with loaded or fallback parameters
    """
    config_path = Path(config_file)
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Loaded camera config from {config_file}")
            
            # Create config object from loaded data
            config = CameraConfig(
                focal_length_pixels=data.get('focal_length_pixels', FALLBACK_CONFIG['focal_length_pixels']),
                horizontal_fov=data.get('horizontal_fov', FALLBACK_CONFIG['horizontal_fov']),
                vertical_fov=data.get('vertical_fov', FALLBACK_CONFIG['vertical_fov']),
                resolution=tuple(data.get('resolution', FALLBACK_CONFIG['resolution'])),
                camera_matrix=data.get('camera_matrix'),
                distortion_coefficients=data.get('distortion_coefficients'),
                calibration_method=data.get('calibration_method'),
                calibration_date=data.get('calibration_date'),
                reprojection_error=data.get('reprojection_error')
            )
            
            return config
            
        except Exception as e:
            print(f"⚠️  Error loading config file {config_file}: {e}")
            print("Using fallback configuration")
    else:
        print(f"⚠️  Config file {config_file} not found. Using fallback configuration")
        print("Run camera calibration to create accurate configuration")
    
    # Return fallback configuration
    return CameraConfig(
        focal_length_pixels=FALLBACK_CONFIG['focal_length_pixels'],
        horizontal_fov=FALLBACK_CONFIG['horizontal_fov'],  
        vertical_fov=FALLBACK_CONFIG['vertical_fov'],
        resolution=tuple(FALLBACK_CONFIG['resolution']),
        calibration_method='fallback'
    )


def save_camera_config(config: CameraConfig, config_file: str = DEFAULT_CONFIG_FILE) -> bool:
    """
    Save camera configuration to JSON file
    
    Args:
        config: CameraConfig object to save
        config_file: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data = {
            'focal_length_pixels': config.focal_length_pixels,
            'horizontal_fov': config.horizontal_fov,
            'vertical_fov': config.vertical_fov,
            'resolution': list(config.resolution),
            'calibration_method': config.calibration_method,
            'calibration_date': config.calibration_date,
            'reprojection_error': config.reprojection_error
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if config.camera_matrix is not None:
            data['camera_matrix'] = config.camera_matrix.tolist()
        
        if config.distortion_coefficients is not None:
            data['distortion_coefficients'] = config.distortion_coefficients.tolist()
        
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"💾 Camera config saved to {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False


def update_main_detection_constants(config: CameraConfig, output_file: str = "updated_constants.py"):
    """
    Generate Python constants file for main detection system
    
    Args:
        config: CameraConfig object
        output_file: Output Python file with constants
    """
    constants_template = f'''# Auto-generated camera constants from calibration
# Generated on: {config.calibration_date or "Unknown"}
# Calibration method: {config.calibration_method or "Unknown"}

# Camera and distance calculation constants (CALIBRATED VALUES)
CAMERA_FOV_HORIZONTAL = {config.horizontal_fov:.1f}  # degrees - calibrated horizontal FOV
CAMERA_FOV_VERTICAL = {config.vertical_fov:.1f}     # degrees - calibrated vertical FOV
FOCAL_LENGTH_PIXELS = {config.focal_length_pixels:.1f}  # Calibrated focal length in pixels

# Image resolution
IMAGE_WIDTH = {config.width}
IMAGE_HEIGHT = {config.height}

# Original fallback values (for reference)
# CAMERA_FOV_HORIZONTAL = 70.6  # degrees - typical webcam horizontal FOV
# CAMERA_FOV_VERTICAL = 55.9   # degrees - typical webcam vertical FOV  
# FOCAL_LENGTH_PIXELS = 452    # Approximate focal length in pixels

# Person measurements (adjust if needed)
AVERAGE_PERSON_HEIGHT = 1.7   # meters (5'7")
AVERAGE_PERSON_WIDTH = 0.45   # meters (shoulder width)

# Distance calculation function
def calculate_distance_to_person(pixel_height, person_height=AVERAGE_PERSON_HEIGHT):
    """Calculate distance to person based on pixel height"""
    if pixel_height <= 0:
        return float('inf')
    return (person_height * FOCAL_LENGTH_PIXELS) / pixel_height

def calculate_expected_pixel_height(distance, person_height=AVERAGE_PERSON_HEIGHT):
    """Calculate expected pixel height for person at given distance"""
    if distance <= 0:
        return 0
    return (person_height * FOCAL_LENGTH_PIXELS) / distance
'''
    
    try:
        with open(output_file, 'w') as f:
            f.write(constants_template)
        print(f"📝 Updated constants saved to {output_file}")
        print("Copy the constants to your main detection script")
        return True
    except Exception as e:
        print(f"❌ Error saving constants: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    print("🎥 Camera Configuration Loader Test")
    print("=" * 40)
    
    # Load configuration
    config = load_camera_config()
    config.print_summary()
    
    print("\n🧪 Testing distance calculations:")
    
    # Test distance calculations
    test_distances = [1.0, 2.0, 5.0, 10.0]
    person_height = 1.7  # meters
    
    print(f"Person height: {person_height}m")
    print(f"{'Distance (m)':<12} {'Pixel Height':<12} {'Calculated Distance':<18}")
    print("-" * 45)
    
    for distance in test_distances:
        pixel_height = config.calculate_pixel_height(distance, person_height)
        calc_distance = config.calculate_distance(pixel_height, person_height)
        
        print(f"{distance:<12.1f} {pixel_height:<12.1f} {calc_distance:<18.2f}")
    
    print("\n🎯 Testing angle calculations:")
    
    # Test angle calculations  
    test_pixels = [-100, -50, 0, 50, 100]
    print(f"{'Pixel Offset':<12} {'Horizontal Angle':<16} {'Vertical Angle':<14}")
    print("-" * 45)
    
    for pixel_offset in test_pixels:
        h_angle = config.pixel_to_angle(pixel_offset, 'horizontal')
        v_angle = config.pixel_to_angle(pixel_offset, 'vertical')
        
        print(f"{pixel_offset:<12} {h_angle:<16.2f} {v_angle:<14.2f}")
    
    # Generate constants file
    print("\n📝 Generating constants file...")
    update_main_detection_constants(config)
    
    print("\n✅ Configuration loader test completed!")
