import cv2
import numpy as np

# Load the image
image = cv2.imread('Rects.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image with the same dimensions as the original image
blank_image = np.ones_like(image)*255

# Copy rectangles and their contents to the blank image
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # x -= 50
    # y -= 50
    # w += 100
    # h += 100
    noise = 80
    # Extract the region of interest (ROI) from the original image
    roi = image[y:y + h+noise, x:x + w+noise]

    # Find the minimum area bounding rectangle (rotated rectangle) of the contour
    rect = cv2.minAreaRect(contour)

    # Get the rotation angle from the rectangle parameters
    angle = rect[-1]

    # If angle is close to 90 degrees, rotate the rectangle by 90 degrees
    if abs(angle) > 45:
        angle += 90+180

    # Rotate the ROI
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_roi = cv2.warpAffine(roi, rotation_matrix, (w, h))


    # Create a binary mask of the rotated ROI
    mask = cv2.threshold(cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    blank_image[y:y + h, x:x + w] = cv2.bitwise_and(blank_image[y:y + h, x:x + w], blank_image[y:y + h, x:x + w], mask=cv2.bitwise_not(mask))
    blank_image[y:y + h, x:x + w] += rotated_roi

# Draw the rectangle on the blank image
# Display the blank image with rotated rectangles
cv2.imshow('Rotated Rectangles', blank_image)
cv2.imwrite("Aligned.png", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
