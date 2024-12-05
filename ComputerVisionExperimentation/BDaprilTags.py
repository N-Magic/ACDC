import apriltag
import cv2 as cv
import numpy as np

TAG_SIZE = 0.065  # AprilTag size in meters
overlay_image = cv.imread('tiger.png')  # Overlay image (make sure the path is correct)

cap = cv.VideoCapture("/dev/video2")
if not cap.isOpened():
    print("Capture could not open")
    quit()

# Initialize the AprilTag detector
detector = apriltag.Detector()

while True:
    ret, frame = cap.read()  # Capture each frame from the camera

    if not ret:  # If frame capture fails
        print("Frame failed to capture")
        break

    gray_scale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    result = detector.detect(gray_scale)  # Detect AprilTags in the grayscale image

    if result:
        tag = result[0]  # Get the first detected tag

        # Get the corners of the AprilTag
        corners = tag.corners

        # Calculate the width and height of the overlay based on the tag size
        tag_width = int(np.linalg.norm(corners[0] - corners[1]))  # Distance between the first two corners (horizontal)
        tag_height = int(np.linalg.norm(corners[1] - corners[2]))  # Distance between the second and third corners (vertical)

        # Resize the overlay image to match the size of the tag (no rotation)
        resized_overlay = cv.resize(overlay_image, (tag_width, tag_height))

        # Calculate the position to place the overlay image (centered on the tag)
        center_x = int(corners[0][0] + (corners[2][0] - corners[0][0]) / 2)
        center_y = int(corners[0][1] + (corners[2][1] - corners[0][1]) / 2)

        # Determine the region of interest (ROI) in the frame where the overlay will be placed
        top_left_x = int(center_x - tag_width / 2)
        top_left_y = int(center_y - tag_height / 2)

        # Ensure the ROI is within the bounds of the frame
        if top_left_x < 0 or top_left_y < 0 or top_left_x + tag_width > frame.shape[1] or top_left_y + tag_height > frame.shape[0]:
            continue  # Skip this frame if the ROI is out of bounds

        # Place the resized overlay onto the frame
        frame[top_left_y:top_left_y + tag_height, top_left_x:top_left_x + tag_width] = resized_overlay

    # Show the updated frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
