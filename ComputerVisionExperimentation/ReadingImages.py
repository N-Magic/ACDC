
import cv2

img = cv2.imread("styles1.png", cv2.IMREAD_COLOR)

# Second Parameter is image array
cv2.imshow("image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()

print(img.shape)