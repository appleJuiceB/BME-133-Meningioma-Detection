import cv2 as cv
import base64
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

## ImageProcessor: This class is responsible for reading, preprocessing, and thresholding the MRI image.
class ImageProcessor:
    def __init__(self):
        self.mri_img = None
        self.thresh_img = None

    def set_original_image(self, original_img):
        self.original_img = original_img

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
        cntr_areas_list = []
        all_contour_areas = 0
        for cnt in self.cnts:
            area = cv.contourArea(cnt)
            all_contour_areas += area
            cntr_areas_list.append(area)
        self.cntr_areas_list = cntr_areas_list
        print(self.cntr_areas_list)
        self.area = all_contour_areas
        print(self.area)
        return


    def compute_tumor_severity(self):
        severity = self.area * self.num_tumors
        if severity > 1000:
            self.severity = "The patient is likely to experience adverse events from the tumor growth(s) at this size."
        else:
            self.severity = "The patient is not likely to experience adverse events from the tumor growth(s) at this size."

    def extract_contour_tumors(self,img):
            tumr_list = []
            mask = np.zeros_like(self.canny, dtype=np.uint8)
            for cntr in self.cnts:
                x, y, w, h = cv.boundingRect(cntr)
                img_crop = img[y:y + h, x:x + w]
                mask_crop = mask[y:y + h, x:x + w]
                result = cv.cvtColor(img_crop, cv.COLOR_BGR2BGRA)
                result[:, :, 3] = mask_crop
                tumr_list.append(result)
            self.tumr_list = tumr_list

    def write_extracted_contours_file(self):
            file_path = './Tumor_Images_Dash'
            if os.path.exists(file_path) != True:
                os.mkdir(file_path)
            elif len(os.listdir(file_path)) != 0:
                for f in os.listdir(file_path):
                    os.remove(os.path.join(file_path, f))

            counter = 1
            img_file_name_list = []
            for i in self.tumr_list:
                img_file_name = 'contour_tumor_' + str(counter) + '.png'
                cv.imwrite(os.path.join(file_path, img_file_name), cv.cvtColor(i, cv.COLOR_BGRA2BGR))
                img_file_name_list.append(img_file_name)
                counter = counter + 1
            self.img_file_name_list = img_file_name_list
                
    def compute_tumor_brain_occupation(self):
        all_tum_occup_list = []
        for a in self.cntr_areas_list:
            tum_occup = round((a / brain_area) * 100, 2)
            all_tum_occup_list.append(tum_occup)
        self.all_tum_occup_list = all_tum_occup_list
        
    def create_scatter_plot(self):
        plt.bar(self.img_file_name_list, self.all_tum_occup_list, color = 'maroon', width = 0.5)
        plt.xlabel('Tumor')
        plt.ylabel('Percent of Tumor Occupation (%)')
        plt.title('Portion of Tumor to Brain Occupation')
        plt.show()
