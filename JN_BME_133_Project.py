import os
import matplotlib as plt
import cv2 as cv
import numpy as np

mri_img = cv.imread("crooked.jpg")

#Thresholding the image:
ret,thresh1 = cv.threshold(mri_img,160,255,cv.THRESH_BINARY)

cv.imshow("Thresholded MRI Image",thresh1)

cv.waitKey(0)
cv.destroyAllWindows()

# Flood-filling the image: 
kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
closed = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
closed = cv.erode(closed, None, iterations = 12)
closed = cv.dilate(closed, None, iterations = 10)
cv.imshow('newimage',closed)
cv.waitKey(0)

cv.destroyAllWindows()

#Delineating edges only:
# edged = cv.Canny(closed, 100, 200)
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(closed)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)

	# return the edged image
	return edged
canny = auto_canny(closed)
cv.imshow('finalimage',canny)
cv.waitKey(0)
cv.destroyAllWindows()

#Finding and Detecting Contours (Presence, Cross-Sectional Area and Severity):
(cnts, _) = cv.findContours(canny.copy(), cv.RETR_EXTERNAL,
cv.CHAIN_APPROX_SIMPLE)
print("Number of tumors detected:",len(cnts))
cv.drawContours(mri_img, cnts, -1, (0, 0, 255), 2)
cnt = cnts[0]
   
area = cv.contourArea(cnt)
print("The cross-sectional area of the tumor(s) mass is approximately:", area)


print("Severity of Tumor Growth")
severity = area*(len(cnts))
if severity > 1000:
	print("The patient is likely to experience adverse events from the tumor growth(s) at this size.")
else: 
	print("The patient is not likely to experience adverse events from the tumor growth(s) at this size.")

cv.imshow("backonimg", mri_img)
   
cv.waitKey(0)

cv.destroyAllWindows()

