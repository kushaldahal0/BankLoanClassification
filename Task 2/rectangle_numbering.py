import cv2


# Load the image
image = cv2.imread('Rects.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds as needed

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store the inner lines' perimeters and their indices
inner_lines = []

# Process each contour
for i, contour in enumerate(contours):
    # Check if the contour has a parent (is an inner contour)
    parent_idx = hierarchy[0][i][3]
    if parent_idx != -1:  # Changed from `!= 1` to `!= -1`
        perimeter = cv2.arcLength(contour, True)
        inner_lines.append((perimeter, i))

# Sort the inner lines based on their perimeters
inner_lines.sort()

# Assign numbers to the contours based on their sorted order
line_numbers = {length_index[1]: i + 1 for i, length_index in enumerate(inner_lines)}

# Draw and label the first four sorted inner contours
for length, index in inner_lines[:4]:  # Only the first four contours
    contour = contours[index]
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Red color
    number = line_numbers[index]
    cv2.putText(image, str(number), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color

# Display the result
cv2.imshow('Image with Rectangles and Extra Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('numbered.png', image)

