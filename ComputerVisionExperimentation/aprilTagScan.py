import cv2
import apriltag
import numpy as np

# Constants
TAG_SIZE = 0.065  # Size of the AprilTag in meters

# Camera calibration parameters
camera_matrix = np.array([[577.68123418, 0, 390.84042288],
                          [0, 596.81380492, 261.04228411],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.02041343, 0.31564429, -0.02455964, 0.02487307, -0.34102883]])

# Define 3D object points for the AprilTag corners in world coordinates
object_point = np.array([
    [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],  # Bottom-left corner
    [TAG_SIZE / 2, -TAG_SIZE / 2, 0],   # Bottom-right corner
    [TAG_SIZE / 2, TAG_SIZE / 2, 0],    # Top-right corner
    [-TAG_SIZE / 2, TAG_SIZE / 2, 0]     # Top-left corner
], dtype=np.float32)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create an AprilTag detector
detector = apriltag.Detector()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
import cv2
import apriltag
import numpy as np

# Constants
TAG_SIZE = 0.065  # Size of the AprilTag in meters

# Camera calibration parameters
camera_matrix = np.array([[577.68123418, 0, 390.84042288],
                          [0, 596.81380492, 261.04228411],
                          [0, 0, 1]])
dist_coeffs = np.array([[-0.02041343, 0.31564429, -0.02455964, 0.02487307, -0.34102883]])

# Define 3D object points for the AprilTag corners in world coordinates
object_point = np.array([
    [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],  # Bottom-left corner
    [TAG_SIZE / 2, -TAG_SIZE / 2, 0],   # Bottom-right corner
    [TAG_SIZE / 2, TAG_SIZE / 2, 0],    # Top-right corner
    [-TAG_SIZE / 2, TAG_SIZE / 2, 0]     # Top-left corner
], dtype=np.float32)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create an AprilTag detector
detector = apriltag.Detector()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Use the original frame without undistortion
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray_frame)

    # Draw detections on the frame
    for detection in detections:
        for i in range(4):
            pt1 = tuple(map(int, detection.corners[i]))
            pt2 = tuple(map(int, detection.corners[(i + 1) % 4]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Get the detected tag ID
        tag_id = detection.tag_id
        print(f"Detected Tag ID: {tag_id}")

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

