import time
import board
import busio
from adafruit_motor import servo
import adafruit_pca9685

# Set up the I2C connection and PCA9685 board
i2c = busio.I2C(board.SCL, board.SDA)
pca = adafruit_pca9685.PCA9685(i2c)
pca.frequency = 50  # Set the frequency for the servos

# Create a servo object for tilt 
tilt_servo_1 = servo.Servo(pca.channels[1])
tilt_servo_2 = servo.Servo(pca.channels[2])

# Sweep the servo from 0 to 180 degrees and back
while True:
    # Sweep from 0 to 180 degrees
    for angle in range(0, 181, 10):  
        tilt_servo_1.angle = 180 - angle
        
        tilt_servo_2.angle = angle
        
        #print(f"Moving both servos to tilt angle: {angle}")
        time.sleep(0.5)  # Delay to allow the servos to move simultaneously

    # Sweep back from 180 to 0 degrees
    for angle in range(180, -1, -10):  
        tilt_servo_1.angle = angle
        p
        tilt_servo_2.angle = 180 - angle
        
        #print(f"Moving both servos to tilt angle: {angle}")
        time.sleep(0.5)  # Delay to allow the servos to move simultaneously
