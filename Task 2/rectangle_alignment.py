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

# Initialize a blank white image with the same size as the original image
blank_image = 255 * np.ones_like(image)

# Draw rotated rectangles on the blank image
for contour in contours:
    # Find the minimum area bounding rectangle (rotated rectangle) of the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[3]

    # Calculate length and breadth
    breadth = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    length = int(np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2))

    # Draw rectangle based on length and breadth
    if breadth > length:
        cv2.rectangle(blank_image, (x1, y1), (x1 + breadth, y1 + length), (0, 0, 0), 2)
    else:
        cv2.rectangle(blank_image, (x1, y1), (x1 + length, y1 + breadth), (0, 0, 0), 2)


# Draw the rectangle on the blank image
# Display the blank image with rotated rectangles
cv2.imshow('Rotated Rectangles', blank_image)
cv2.imwrite("Aligned.png", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
