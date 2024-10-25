import cv2

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize a counter to track frames
frame_counter = 0
frame_interval = 5  # Process every 5th frame

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
        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)
    
    # Increment the frame counter
    frame_counter += 1

    # Press 'q' to exit the camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
