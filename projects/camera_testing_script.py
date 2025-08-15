import cv2

for i in range(0, 36):  # Check video devices from 0 to 35
    cap = cv2.VideoCapture(f'/dev/video{i}')
    if cap.isOpened():
        print(f"Device /dev/video{i} is available")
        cap.release()
    else:
        print(f"Device /dev/video{i} is not accessible")