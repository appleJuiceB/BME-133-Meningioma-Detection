import cv2 as cv
import base64
import cv2
import numpy as np


## ImageProcessor: This class is responsible for reading, preprocessing, and thresholding the MRI image.
class ImageProcessor:
    def __init__(self):
        self.mri_img = None
        self.thresh_img = None

    def set_image_data(self, contents):
        # Decode the image data from base64
        img = base64.b64decode(contents.split(',')[1])
        # Load the image using cv2
        nparr = np.frombuffer(img, np.uint8)
        self.mri_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def threshold_image(self):
        if self.mri_img is not None:
            ret, thresh_img = cv.threshold(self.mri_img, 160, 255, cv.THRESH_BINARY)
            self.thresh_img = thresh_img
            return thresh_img
        else:
            raise ValueError("No image data has been set.")

    def flood_fill(self):
        if self.thresh_img is not None:
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
            closed = cv.morphologyEx(self.thresh_img, cv.MORPH_CLOSE, kernel)
            closed = cv.erode(closed, None, iterations=12)
            closed = cv.dilate(closed, None, iterations=10)
            self.closed = closed
        else:
            raise ValueError("Threshold image not available.")

    def auto_canny(self, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(self.closed)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv.Canny(self.closed, lower, upper)

        self.canny = edged

       def find_brain_area(self):
        # Convert the image to grayscale
        gray = cv2.cvtColor(self.mri_img, cv2.COLOR_RGB2GRAY)

        # Threshold the image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, which is assumed to be the brain area
        max_contour = max(contours, key=cv2.contourArea)
        brain_area_mask = np.zeros_like(thresh)
        cv2.drawContours(brain_area_mask, [max_contour], 0, 255, -1)

        brain_area = cv2.contourArea(max_contour)

        return brain_area


##ContourDetector: This class is responsible for detecting and analyzing the contours in the MRI image.

class ContourDetector:
    def __init__(self, canny):
        self.canny = canny

    def find_contours(self):
        cnts, _ = cv.findContours(self.canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.cnts = cnts
        self.num_tumors = len(cnts)

    def draw_contours(self, img):
        cv.drawContours(img, self.cnts, -1, (0, 0, 255), 2)

    def compute_cross_sectional_area(self):
        area = 0
        for cnt in self.cnts:
            area += cv.contourArea(cnt)
        self.area = area

    def compute_tumor_severity(self):
        severity = self.area * self.num_tumors
        if severity > 1000:
            self.severity = "The patient is likely to experience adverse events from the tumor growth(s) at this size."
        else:
            self.severity = "The patient is not likely to experience adverse events from the tumor growth(s) at this size."
