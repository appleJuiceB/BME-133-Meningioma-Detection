#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# Psuedo Code:
# 
# 
# 1. Read the input image.
# 2. Convert the image to grayscale.
# 3. Apply a Gaussian blur to the grayscale image to remove noise.
# 4. Apply thresholding to the blurred image to create a binary image.
# 5. Find the contours in the binary image.
# 6. Create a copy of the original image to draw contours on.
# 7. Set a minimum area threshold for the blobs.
# 8. Initialize a count of blobs.
# 9. Loop through all contours:
#     a. Calculate the area of the contour.
#     b. If the area is greater than the minimum threshold, draw the contour on the copy of the original image and increment the blob count.
# 10. Display the result image with detected blobs and the count of blobs.
# 
# 

# In[2]:


pip install easygui


# In[1]:


## METHODS ## 
#method to open img
#for mac, require second waitkey
def display():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


# In[45]:


## TEST IMAGE ## 
import cv2 as cv2

def display():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# Load image file
image = cv.imread("Te-me_0015.jpeg")

dim=(500,590)
image=cv2.resize(image, dim)
cv2.imshow("image",image)
display()

#Thresholding 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("grey",gray)
display()
(T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh)
display()
(T, threshInv) = cv2.threshold(gray, 155, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshInv", threshInv)
display()


# In[2]:


##USER IMPUT IMAGE## 
import easygui
import cv2

def display():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
# Get the image file from the user using the file dialog
file_path = easygui.fileopenbox(default="*.jpg;*.jpeg;*.png", filetypes=["*.jpg", "*.jpeg", "*.png"])

# Call the function to read and display the image
image = cv2.imread(file_path)
cv2.imshow("Image", image)
display()
    
#Thresholding 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("grey",gray)
display()
(T, thresh) = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh",thresh)
display()
(T, threshInv) = cv2.threshold(gray, 155, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshInv", threshInv)
display()


# In[ ]:


#Morphological transformation 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", closed)
display()

#erosion erode away boundaries of foreground object 

closed = cv2.erode(closed, None, iterations = 14)
closed = cv2.dilate(closed, None, iterations = 13)

#canny edge dector - outine of tumor
canny = auto_canny(closed)

_, cnts, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
cv2.imshow("Image", image)
display() 

