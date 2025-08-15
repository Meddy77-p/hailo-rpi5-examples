import time
import board
import busio
from adafruit_motor import servo
import adafruit_pca9685

# Set up I2C connection
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize PCA9685
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50

# Choose channels
channel_servo1 = 0
channel_servo2 = 2

# Create servo objects
servo1 = servo.Servo(pca.channels[channel_servo1])
servo2 = servo.Servo(pca.channels[channel_servo2])

# Optional calibration
servo1_offset = 0
servo2_offset = 0

# Function to move servos in mirrored motion
def move_mirrored(angle):
    angle1 = max(0, min(180, angle + servo1_offset))
    angle2 = max(0, min(180, (180 - angle) + servo2_offset))
    
    servo1.angle = angle1
    servo2.angle = angle2
    print(f"Servo1 → {angle1}°, Servo2 → {angle2}°")
    time.sleep(2)

try:
    while True:
        move_mirrored(0)
        move_mirrored(45)
        move_mirrored(90)
        move_mirrored(135)
        move_mirrored(180)
        move_mirrored(90)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Always reset to 90° (neutral position) on exit
    print("Resetting servos to 90°...")
    servo1.angle = 90 + servo1_offset
    servo2.angle = 90 + servo2_offset
    time.sleep(1)
    print("Servos returned to center. Exiting.")
