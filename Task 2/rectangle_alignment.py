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
blank_image = np.ones_like(image) * 255

# Copy rectangles and their contents to the blank image
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    noise = 10
    x -= noise
    y -= noise
    w += noise*2
    h += noise*2
    # Extract the region of interest (ROI) from the original image
    roi = image[y:y + h, x:x + w]

    # Find the minimum area bounding rectangle (rotated rectangle) of the contour
    rect = cv2.minAreaRect(contour)

    # Get the rotation angle from the rectangle parameters
    angle = rect[-1]

    # Adjust the angle
    if abs(angle) > 45:
        angle += 270

    # Rotate the ROI
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_roi = cv2.warpAffine(roi, rotation_matrix, (w, h))

    # Create a mask where the black regions will be set to white
    mask = (rotated_roi[:, :, 0] == 0) & (rotated_roi[:, :, 1] == 0) & (rotated_roi[:, :, 2] == 0)
    mask = mask.astype(np.uint8) * 255

    # Dilate the mask to cover more area
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Apply the dilated mask to set black regions to white in the rotated ROI
    rotated_roi[dilated_mask == 255] = [255, 255, 255]
    blank_image[y:y + h, x:x + w] += rotated_roi

# Display the blank image with copied rectangles and contents
cv2.imshow('Rotated Rectangles', blank_image)
cv2.imwrite("Aligned.png", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
