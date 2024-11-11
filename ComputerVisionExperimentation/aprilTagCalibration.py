import cv2
import apriltag
import numpy as np

# Constants
TAG_SIZE = 0.065  # 65 mm in meters
num_tags = 3  # Number of tags for calibration
image_points = []  # 2D image points
object_points = []  # 3D object points
frame_counter = 0

# Define 3D object points for the AprilTags in world coordinates
# Assuming the AprilTags are in the same plane, define them relative to a point
object_point = np.array([
    [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],  # Bottom-left corner
    [TAG_SIZE / 2, -TAG_SIZE / 2, 0],   # Bottom-right corner
    [TAG_SIZE / 2, TAG_SIZE / 2, 0],    # Top-right corner
    [-TAG_SIZE / 2, TAG_SIZE / 2, 0]    # Top-left corner
], dtype=np.float32)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create an AprilTag detector
detector = apriltag.Detector()

print("Press 'c' to capture a frame, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_frame)

    # Draw detections on the frame
    for detection in detections:
        for i in range(4):
            pt1 = tuple(map(int, detection.corners[i]))
            pt2 = tuple(map(int, detection.corners[(i + 1) % 4]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow('Camera Feed - Press "c" to capture', frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        # Capture the valid detections for calibration
        valid_detections = [detection for detection in detections if len(detection.corners) == 4]
        
        for detection in valid_detections:
            image_points.append(np.array(detection.corners, dtype=np.float32))
            object_points.append(object_point)

        print(f"Captured a frame for calibration. Number of valid detections: {len(valid_detections)}")
        
    elif key == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera using the collected object and image points
if len(image_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, gray_frame.shape[::-1], None, None
    )

    if ret:
        print("Calibration successful!")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs)

        # Save the calibration results if needed
        np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    else:
        print("Calibration failed.")
else:
    print("No images were captured for calibration.")

