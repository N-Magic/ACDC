import cv2
import apriltag
import numpy as np

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create an AprilTag detector
detector = apriltag.Detector()

# Set frame processing interval
frame_counter = 0
frame_interval = 5  # Process every 5th frame

# Define the physical size of the AprilTag in meters (65 mm = 0.065 m)
TAG_SIZE = 0.065  # 65 mm in meters

# Estimate the camera focal length (in pixels)
# Adjust this value according to your camera settings
FOCAL_LENGTH = 600  # This may need calibration

# Start capturing video frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Only process every 5th frame
    if frame_counter % frame_interval == 0:
        # Convert frame to grayscale for the AprilTag detector
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = detector.detect(gray_frame)

        # Draw detections and display tag info
        for detection in detections:
            # Draw a bounding box around the detected AprilTag
            for i in range(4):
                pt1 = tuple(map(int, detection.corners[i]))
                pt2 = tuple(map(int, detection.corners[(i + 1) % 4]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Calculate the size of the tag in pixels
            width = np.linalg.norm(detection.corners[0] - detection.corners[1])
            # Calculate distance to the AprilTag
            distance = (TAG_SIZE * FOCAL_LENGTH) / width

            # Display tag ID and distance at the center of the AprilTag
            tag_center = tuple(map(int, detection.center))
            cv2.putText(frame, f'ID: {detection.tag_id}', tag_center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {distance:.2f} m', (tag_center[0], tag_center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Feed with AprilTag Detection', frame)
    
    # Increment the frame counter
    frame_counter += 1

    # Press 'q' to exit the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
