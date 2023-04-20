import cv2

# Read the image
img = cv2.imread('brain_scan.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to create binary image
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

# Find contours in binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours on
img_contours = img.copy()

# Set minimum area threshold
min_area = 1000

# Initialize a count of tumors
tumor_count = 0

# Loop through all contours
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # If the area is greater than the minimum threshold, draw the contour on the image and increment the tumor count
    if area > min_area:
        cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 2)
        tumor_count += 1

# Display the result
cv2.imshow('Detected Tumors', img_contours)
print(f"Number of tumors: {tumor_count}")
cv2.waitKey(0)
cv2.destroyAllWindows()
