#!/usr/bin/env python3
"""
Real-time Waypoint Visualizer
Shows waypoints being created from person detection in Mission Planner
and local matplotlib display
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow
import numpy as np
from collections import deque
import time
import threading
from pathlib import Path
import csv

class WaypointVisualizer:
    """Real-time visualization of GPS waypoints"""
    
    def __init__(self, log_dir="servo_logs", max_points=50):
        self.log_dir = Path(log_dir)
        self.max_points = max_points
        
        # Find the latest GPS CSV file
        self.gps_file = self.find_latest_gps_file()
        if not self.gps_file:
            print("❌ No GPS log file found!")
            return
            
        print(f"📊 Monitoring GPS file: {self.gps_file}")
        
        # Data storage
        self.vehicle_positions = deque(maxlen=max_points)
        self.waypoint_positions = deque(maxlen=max_points)
        self.current_vehicle_pos = None
        self.current_heading = 0
        
        # Setup plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
        self.setup_plots()
        
        # File monitoring
        self.last_position = 0
        self.running = True
        
    def find_latest_gps_file(self):
        """Find the most recent GPS CSV file"""
        gps_files = list(self.log_dir.glob("gps_points_*.csv"))
        if not gps_files:
            return None
        return max(gps_files, key=lambda p: p.stat().st_mtime)
    
    def setup_plots(self):
        """Setup the matplotlib plots"""
        # Top-down view (ax1)
        self.ax1.set_title("Top-Down View (North-Up)")
        self.ax1.set_xlabel("East/West (meters)")
        self.ax1.set_ylabel("North/South (meters)")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        
        # Mission Planner style view (ax2)
        self.ax2.set_title("Mission Planner View (GPS)")
        self.ax2.set_xlabel("Longitude")
        self.ax2.set_ylabel("Latitude")
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize empty plots
        self.vehicle_trail, = self.ax1.plot([], [], 'b-', alpha=0.5, label='Vehicle Path')
        self.vehicle_dot = Circle((0, 0), 2, color='blue', label='Vehicle')
        self.ax1.add_patch(self.vehicle_dot)
        
        self.waypoint_scatter = self.ax1.scatter([], [], c='red', s=100, 
                                                marker='*', label='Waypoints')
        
        # GPS view
        self.gps_vehicle_trail, = self.ax2.plot([], [], 'b-', alpha=0.5)
        self.gps_waypoint_scatter = self.ax2.scatter([], [], c='red', s=100, marker='*')
        
        # Legends
        self.ax1.legend(loc='upper right')
        
        # Vehicle heading arrow
        self.heading_arrow = None
        
    def read_new_data(self):
        """Read new GPS data from CSV file"""
        try:
            with open(self.gps_file, 'r') as f:
                # Skip to last position
                f.seek(self.last_position)
                
                reader = csv.DictReader(f)
                if self.last_position == 0:
                    # First read - get headers
                    header_line = f.readline()
                    self.last_position = f.tell()
                    return
                
                for row in reader:
                    if not row:
                        continue
                        
                    try:
                        # Parse waypoint data
                        wp_data = {
                            'timestamp': float(row['timestamp']),
                            'wp_lat': float(row['detection_lat']),
                            'wp_lon': float(row['detection_lon']),
                            'wp_alt': float(row['detection_alt']),
                            'vehicle_lat': float(row['vehicle_lat']),
                            'vehicle_lon': float(row['vehicle_lon']),
                            'heading': float(row['vehicle_heading']),
                            'rel_x': float(row['relative_x']),
                            'rel_y': float(row['relative_y']),
                            'confidence': float(row['confidence'])
                        }
                        
                        # Store vehicle position
                        self.current_vehicle_pos = (wp_data['vehicle_lat'], wp_data['vehicle_lon'])
                        self.current_heading = wp_data['heading']
                        self.vehicle_positions.append((wp_data['rel_x'], wp_data['rel_y']))
                        
                        # Store waypoint
                        self.waypoint_positions.append({
                            'local': (wp_data['rel_x'], wp_data['rel_y']),
                            'gps': (wp_data['wp_lat'], wp_data['wp_lon']),
                            'confidence': wp_data['confidence']
                        })
                        
                        print(f"🎯 New waypoint: {wp_data['rel_x']:.1f}m, {wp_data['rel_y']:.1f}m "
                              f"(Confidence: {wp_data['confidence']:.2f})")
                        
                    except (KeyError, ValueError) as e:
                        continue
                
                self.last_position = f.tell()
                
        except Exception as e:
            print(f"Error reading GPS data: {e}")
    
    def update_plot(self, frame):
        """Update the plots with new data"""
        # Read new data
        self.read_new_data()
        
        if not self.waypoint_positions:
            return
        
        # Update local view (top-down)
        if self.vehicle_positions:
            # Vehicle trail
            vehicle_x = [p[0] for p in self.vehicle_positions]
            vehicle_y = [p[1] for p in self.vehicle_positions]
            self.vehicle_trail.set_data(vehicle_y, vehicle_x)  # Swap for North-up
            
            # Current vehicle position
            self.vehicle_dot.center = (0, 0)  # Vehicle at origin
            
            # Update heading arrow
            if self.heading_arrow:
                self.heading_arrow.remove()
            
            # Arrow showing vehicle heading
            arrow_len = 5
            dx = arrow_len * np.sin(np.radians(self.current_heading))
            dy = arrow_len * np.cos(np.radians(self.current_heading))
            self.heading_arrow = Arrow(0, 0, dx, dy, width=2, color='blue', alpha=0.7)
            self.ax1.add_patch(self.heading_arrow)
        
        # Update waypoints
        if self.waypoint_positions:
            wp_x = [wp['local'][0] for wp in self.waypoint_positions]
            wp_y = [wp['local'][1] for wp in self.waypoint_positions]
            colors = [wp['confidence'] for wp in self.waypoint_positions]
            
            self.waypoint_scatter.set_offsets(np.c_[wp_y, wp_x])  # Swap for North-up
            self.waypoint_scatter.set_array(np.array(colors))
            
            # Update GPS view
            gps_wp_lat = [wp['gps'][0] for wp in self.waypoint_positions]
            gps_wp_lon = [wp['gps'][1] for wp in self.waypoint_positions]
            self.gps_waypoint_scatter.set_offsets(np.c_[gps_wp_lon, gps_wp_lat])
            self.gps_waypoint_scatter.set_array(np.array(colors))
        
        # Update GPS vehicle trail
        if self.current_vehicle_pos:
            # This would need historical GPS positions - simplified for now
            self.gps_vehicle_trail.set_data([self.current_vehicle_pos[1]], [self.current_vehicle_pos[0]])
        
        # Auto-scale axes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Add grid references
        self.ax1.set_xlim(-50, 50)
        self.ax1.set_ylim(-50, 50)
        
    def show_mission_planner_info(self):
        """Display information about Mission Planner integration"""
        info_text = """
🗺️  MISSION PLANNER INTEGRATION:

1. The waypoints are being sent to Mission Planner automatically
2. In Mission Planner:
   - Go to FLIGHT DATA tab
   - You'll see waypoints appear as the system detects people
   - Green lines show the path to waypoints
   - Numbers indicate waypoint sequence

3. Waypoint Colors in Mission Planner:
   - White: Pending waypoints
   - Green: Active waypoint
   - Orange: Reached waypoints

4. To follow waypoints:
   - Set mode to GUIDED for immediate response
   - Or AUTO mode to follow mission

5. Monitor in real-time:
   - Distance to waypoint
   - ETA to waypoint
   - Current target

Press Ctrl+C to stop visualization
        """
        print(info_text)
    
    def run(self):
        """Run the visualization"""
        self.show_mission_planner_info()
        
        # Setup animation
        ani = animation.FuncAnimation(self.fig, self.update_plot, 
                                    interval=1000, blit=False)
        
        plt.tight_layout()
        plt.show()

# Additional function to monitor Mission Planner waypoints
def create_mission_planner_monitor():
    """
    Create a simple status monitor for Mission Planner waypoints
    """
    monitor_text = """
#!/usr/bin/env python3
# Mission Planner Waypoint Monitor

from pymavlink import mavutil
import time

# Connect to vehicle (adjust connection string as needed)
connection = mavutil.mavlink_connection('/dev/serial0', baud=57600)

print("Waiting for heartbeat...")
connection.wait_heartbeat()
print("Connected!")

while True:
    # Monitor mission status
    msg = connection.recv_match(type=['MISSION_CURRENT', 'MISSION_ITEM_REACHED', 
                                     'NAV_CONTROLLER_OUTPUT'], blocking=True, timeout=1)
    
    if msg:
        msg_type = msg.get_type()
        
        if msg_type == 'MISSION_CURRENT':
            print(f"Current waypoint: {msg.seq}")
            
        elif msg_type == 'MISSION_ITEM_REACHED':
            print(f"✅ Reached waypoint {msg.seq}")
            
        elif msg_type == 'NAV_CONTROLLER_OUTPUT':
            print(f"Distance to WP: {msg.wp_dist}m, Bearing: {msg.target_bearing}°")
    
    time.sleep(0.1)
"""
    
    # Save monitor script
    with open('mission_planner_monitor.py', 'w') as f:
        f.write(monitor_text)
    
    print("📝 Created mission_planner_monitor.py - run this to monitor waypoint progress")

if __name__ == "__main__":
    print("🚀 Starting Waypoint Visualizer...")
    print("Make sure the main tracking script is running!")
    
    # Create Mission Planner monitor script
    create_mission_planner_monitor()
    
    # Run visualizer
    visualizer = WaypointVisualizer()
    
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\n👋 Visualization stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
