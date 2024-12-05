import cv2

# Try accessing different video capture devices
for index in range(30):  # Try up to 4 devices (video0, video1, video2, video3)
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Device found at index {index}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Webcam {index}", frame)
            cv2.waitKey(0)  # Display the frame until a key is pressed
        cap.release()
    else:
        print(f"No device found at index {index}")

cv2.destroyAllWindows()

