import cv2 as cv
import numpy as np
import easygui
import pandas as pd
import os


## ImageProcessor: This class is responsible for reading, preprocessing, and thresholding the MRI image.
class ImageProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_image(self):
        self.mri_img = cv.imread(self.file_path)

    def threshold_image(self):
        self.ret, self.thresh1 = cv.threshold(self.mri_img, 160, 255, cv.THRESH_BINARY)

    def flood_fill(self):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
        closed = cv.morphologyEx(self.thresh1, cv.MORPH_CLOSE, kernel)
        closed = cv.erode(closed, None, iterations=12)
        closed = cv.dilate(closed, None, iterations=10)
        self.closed = closed

    def auto_canny(self, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(self.closed)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv.Canny(self.closed, lower, upper)

        self.canny = edged

    def find_brain_area(self):
        # read image
        img = cv.imread(self.file_path, cv.IMREAD_GRAYSCALE)

        # apply threshold to convert the image to binary
        _, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # check OpenCV version
        if cv.__version__.startswith('3'):
            _, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # find the contour with the largest area (which corresponds to the brain)
        brain_contour = max(contours, key=cv.contourArea)
        self.brain_contour = brain_contour
        
        # calculate the area of the brain
        brain_area = cv.contourArea(brain_contour)

        return brain_area


##ContourDetector: This class is responsible for detecting and analyzing the contours in the MRI image.

class ContourDetector:
    def __init__(self, canny):
        self.canny = canny

    def find_contours(self):
        cnts, _ = cv.findContours(self.canny.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.cnts = cnts
        self.num_tumors = len(cnts)

 #-----------------------------------------------------

    def extract_contour_tumors(self):
        tumr_list = []
        mask = np.zeros_like(self.canny, dtype=np.uint8)
        for cntr in self.cnts:
            x,y,w,h = cv.boundingRect(cntr)
            img_crop = img_proc.mri_img[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]
            result = cv.cvtColor(img_crop, cv.COLOR_BGR2BGRA)
            result[:,:,3] = mask_crop
            tumr_list.append(result)
        self.tumr_list = tumr_list
        
    def write_extracted_contours_file(self):
        file_path = './Tumor_Images'
        if os.path.exists(file_path) != True:
            os.mkdir(file_path)

        counter = 1
        for i in self.tumr_list:
            img_file_name = 'contour_tumor_' + str(counter) + '.png'
            cv.imwrite(os.path.join(file_path, img_file_name), cv.cvtColor(i, cv.COLOR_BGRA2BGR))
            #imshow will show all "Edged Images"
            #cv.imshow("Edged Image" + str(counter), i)
            counter = counter + 1

#-----------------------------------------------------

    def draw_contours(self, img):
        cv.drawContours(img, self.cnts, -1, (0, 0, 255), 2)

    def compute_cross_sectional_area(self):
        area = 0
        cntr_areas_list = []
        all_contour_areas = 0
        for cnt in self.cnts:
            area += cv.contourArea(cnt)
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


def display():
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)


if __name__ == '__main__':
    # Get the image path from the user
    file_path = easygui.fileopenbox(default="*.jpg;*.jpeg;*.png", filetypes=["*.jpg", "*.jpeg", "*.png"])

    img_proc = ImageProcessor(file_path)
    img_proc.read_image()
    img_proc.threshold_image()
    img_proc.flood_fill()
    img_proc.auto_canny()

    brain_area = img_proc.find_brain_area()


    contour_detector = ContourDetector(img_proc.canny)
    contour_detector.find_contours()
    contour_detector.compute_cross_sectional_area()
    contour_detector.compute_tumor_severity()
    contour_detector.draw_contours(img_proc.mri_img)

    print("Number of tumors detected:", contour_detector.num_tumors) 
    print("The cross-sectional area of the tumor(s) mass is approximately:", contour_detector.area)
    print("Tumor:Brain area ratio: ", round(contour_detector.area / brain_area, 2))
    print("The tumor occupies approximately ", round((contour_detector.area / brain_area) * 100, 2),"% of the total cross-sectional area of the brain")
    print("Severity of Tumor Growth:", contour_detector.severity)

    # Create contour areas data frame
    df = pd.DataFrame(contour_detector.cntr_areas_list, columns = ["Contour Areas"])
    print ("data frame", df)

    #-----------------------------------------------------

    contour_detector.extract_contour_tumors()
    contour_detector.write_extracted_contours_file()

    #-----------------------------------------------------
    
    cv.imshow("MRI Image with Contours", img_proc.mri_img)
    display()
