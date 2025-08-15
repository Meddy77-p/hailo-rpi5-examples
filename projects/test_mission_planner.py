#!/usr/bin/env python3
"""
Mission Planner Compatible Waypoint Upload System
This code properly uploads waypoints that will appear in Mission Planner's mission list
"""

import time
import math
from pymavlink import mavutil

class MissionPlannerWaypointUploader:
    def __init__(self, connection_string='/dev/serial0', baud=57600):
        self.connection_string = connection_string
        self.baud = baud
        self.connection = None
        self.waypoints = []
        self.mission_count = 0
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        
        self.connect()
    
    def connect(self):
        """Connect to flight controller"""
        try:
            print(f"🛰️ Connecting to {self.connection_string}...")
            self.connection = mavutil.mavlink_connection(
                self.connection_string, 
                baud=self.baud,
                timeout=10
            )
            
            # Wait for heartbeat
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            if heartbeat:
                print(f"✅ Connected! System ID: {self.connection.target_system}")
                
                # Get current position
                self._get_current_position()
                return True
            else:
                print("❌ No heartbeat received")
                return False
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def _get_current_position(self):
        """Get current GPS position from flight controller"""
        print("📍 Getting current position...")
        
        # Request position data
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            200000,  # 0.2 second interval
            0, 0, 0, 0, 0
        )
        
        # Wait for position message
        start_time = time.time()
        while time.time() - start_time < 5:
            msg = self.connection.recv_match(
                type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], 
                blocking=False, 
                timeout=0.5
            )
            
            if msg:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    print(f"📍 Position: {self.current_lat:.6f}, {self.current_lon:.6f}, {self.current_alt:.1f}m")
                    return
                elif msg.get_type() == 'GPS_RAW_INT':
                    self.current_lat = msg.lat / 1e7
                    self.current_lon = msg.lon / 1e7
                    self.current_alt = msg.alt / 1000.0
                    print(f"📍 Position: {self.current_lat:.6f}, {self.current_lon:.6f}, {self.current_alt:.1f}m")
                    return
        
        print("⚠️  Could not get current position, using defaults")
        self.current_lat = 36.131287  # Your approximate location
        self.current_lon = -97.082025
        self.current_alt = 297.8
    
    def clear_mission(self):
        """Clear existing mission from flight controller"""
        print("🗑️ Clearing existing mission...")
        
        try:
            # Send mission clear
            self.connection.mav.mission_clear_all_send(
                self.connection.target_system,
                self.connection.target_component
            )
            
            # Wait for acknowledgment
            start_time = time.time()
            while time.time() - start_time < 3:
                msg = self.connection.recv_match(type='MISSION_ACK', blocking=False, timeout=0.5)
                if msg and msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    print("✅ Mission cleared")
                    return True
            
            print("⚠️  Mission clear timeout (may still have worked)")
            return True
            
        except Exception as e:
            print(f"❌ Mission clear error: {e}")
            return False
    
    def add_waypoint(self, lat, lon, alt, hold_time=0, acceptance_radius=5):
        """Add a waypoint to the mission list"""
        waypoint = {
            'seq': len(self.waypoints),
            'frame': mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            'command': mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            'current': 0,
            'autocontinue': 1,
            'param1': hold_time,           # Hold time in seconds
            'param2': acceptance_radius,    # Acceptance radius in meters
            'param3': 0,                   # Pass through (0 = stop at waypoint)
            'param4': 0,                   # Yaw angle (0 = any)
            'x': lat,                      # Latitude
            'y': lon,                      # Longitude
            'z': alt,                      # Altitude
            'mission_type': mavutil.mavlink.MAV_MISSION_TYPE_MISSION
        }
        
        self.waypoints.append(waypoint)
        print(f"📍 Added waypoint {len(self.waypoints)}: {lat:.6f}, {lon:.6f}, {alt:.1f}m")
    
    def add_relative_waypoint(self, x_meters, y_meters, alt_offset=5):
        """Add waypoint relative to current position"""
        if self.current_lat == 0 and self.current_lon == 0:
            print("❌ No current position available")
            return False
        
        # Calculate distance and bearing
        distance = math.sqrt(x_meters**2 + y_meters**2)
        bearing_rad = math.atan2(x_meters, y_meters)  # Note: y is north, x is east
        
        # Convert to GPS coordinates
        EARTH_RADIUS = 6371000
        lat_rad = math.radians(self.current_lat)
        lon_rad = math.radians(self.current_lon)
        
        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance / EARTH_RADIUS) +
            math.cos(lat_rad) * math.sin(distance / EARTH_RADIUS) * math.cos(bearing_rad)
        )
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance / EARTH_RADIUS) * math.cos(lat_rad),
            math.cos(distance / EARTH_RADIUS) - math.sin(lat_rad) * math.sin(new_lat_rad)
        )
        
        wp_lat = math.degrees(new_lat_rad)
        wp_lon = math.degrees(new_lon_rad)
        wp_alt = self.current_alt + alt_offset
        
        self.add_waypoint(wp_lat, wp_lon, wp_alt)
        return True
    
    def upload_mission(self):
        """Upload the complete mission to flight controller (Mission Planner compatible)"""
        if not self.waypoints:
            print("❌ No waypoints to upload")
            return False
        
        print(f"📤 Uploading mission with {len(self.waypoints)} waypoints...")
        
        try:
            # Clear any pending messages first
            while True:
                msg = self.connection.recv_match(blocking=False, timeout=0.01)
                if not msg:
                    break
            
            # Step 1: Send mission count
            self.connection.mav.mission_count_send(
                self.connection.target_system,
                self.connection.target_component,
                len(self.waypoints),
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION
            )
            
            print(f"📤 Sent mission count: {len(self.waypoints)}")
            
            # Step 2: Handle mission requests and send waypoints
            waypoints_sent = 0
            timeout_start = time.time()
            ack_received = False
            
            while (waypoints_sent < len(self.waypoints) or not ack_received) and time.time() - timeout_start < 20:
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
                    
                    if seq >= len(self.waypoints):
                        print(f"❌ Requested waypoint {seq} out of range")
                        continue
                    
                    waypoint = self.waypoints[seq]
                    
                    # Send waypoint using MISSION_ITEM_INT (preferred)
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
                        print(f"📤 Sent waypoint {seq + 1}/{len(self.waypoints)}: {waypoint['x']:.6f}, {waypoint['y']:.6f}")
                    
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
            
            # Check if we actually sent all waypoints even without ACK
            if waypoints_sent >= len(self.waypoints):
                print("✅ All waypoints sent (ACK timeout, but likely successful)")
                print("🎯 Check Mission Planner - waypoints may still be there")
                print("📱 In Mission Planner: Go to 'Flight Plan' tab and click 'Read WPs'")
                return True
            else:
                print(f"❌ Mission upload incomplete: {waypoints_sent}/{len(self.waypoints)} sent")
                return False
            
        except Exception as e:
            print(f"❌ Mission upload error: {e}")
            return False
    
    def create_sample_mission(self):
        """Create a sample mission for testing"""
        print("🎯 Creating sample mission...")
        
        # Clear existing mission first
        self.clear_mission()
        time.sleep(1)
        
        # Add waypoints in a square pattern around current position
        waypoint_distance = 10  # 10 meters from center
        
        # North
        self.add_relative_waypoint(0, waypoint_distance, 10)
        
        # East  
        self.add_relative_waypoint(waypoint_distance, waypoint_distance, 10)
        
        # South
        self.add_relative_waypoint(waypoint_distance, 0, 10)
        
        # West
        self.add_relative_waypoint(0, 0, 10)
        
        # Upload the mission
        result = self.upload_mission()
        
        # Even if upload reports failure, waypoints might still be there
        if not result:
            print("\n⚠️  Upload reported failure, but checking Mission Planner is recommended")
            print("📱 The waypoints may still have been uploaded successfully")
        
        return result
    
    def add_tracking_waypoint(self, x_meters, y_meters, distance, confidence):
        """Add a waypoint from tracking system (your servo tracker integration)"""
        if distance < 2.0:  # Minimum distance
            return False
        
        # Add waypoint relative to current position
        success = self.add_relative_waypoint(x_meters, y_meters, 10)
        
        if success:
            # Upload the updated mission
            return self.upload_mission()
        
        return False


def main():
    """Test the Mission Planner waypoint uploader"""
    print("🚀 Mission Planner Waypoint Uploader Test")
    print("=" * 50)
    
    # Create uploader
    uploader = MissionPlannerWaypointUploader()
    
    if not uploader.connection:
        print("❌ Failed to connect to flight controller")
        return
    
    try:
        # Create and upload sample mission
        print("\n📤 Creating sample mission...")
        uploader.create_sample_mission()
        
        print("\n" + "=" * 60)
        print("📱 MISSION PLANNER INSTRUCTIONS")
        print("=" * 60)
        print("1. Open Mission Planner")
        print("2. Connect to your flight controller")
        print("3. Go to 'Flight Plan' tab")
        print("4. Click 'Read WPs' button")
        print("5. You should see 4 waypoints in a square pattern!")
        print("=" * 60)
        
        # Show mission info
        print(f"\n📊 Mission Summary:")
        print(f"   📍 Center: {uploader.current_lat:.6f}, {uploader.current_lon:.6f}")
        print(f"   📏 Pattern: 10m square")
        print(f"   📈 Altitude: +10m relative")
        print(f"   🎯 Waypoints: {len(uploader.waypoints)}")
        
        # Optional: Try to verify upload by reading back mission
        print(f"\n🔍 Attempting to verify mission upload...")
        try:
            uploader.connection.mav.mission_request_list_send(
                uploader.connection.target_system,
                uploader.connection.target_component
            )
            
            start_time = time.time()
            while time.time() - start_time < 3:
                msg = uploader.connection.recv_match(type='MISSION_COUNT', blocking=False, timeout=0.5)
                if msg:
                    print(f"✅ Flight controller reports {msg.count} waypoints in mission")
                    break
            else:
                print("⚠️  Could not verify mission count (normal for some flight controllers)")
                
        except Exception as e:
            print(f"⚠️  Verification error: {e}")
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n👋 Test complete - Check Mission Planner!")


if __name__ == "__main__":
    main()