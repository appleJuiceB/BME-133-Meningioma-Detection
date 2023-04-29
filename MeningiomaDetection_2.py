import cv2 as cv
import numpy as np
import easygui
import pandas as pd


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

    '''def write_contour_files(self):
        tumr_list = []
        mask = np.zeros_like(img_proc.closed, dtype=np.uint8)
        for cntr in self.cnts:
            x,y,w,h = cv.boundingRect(cntr)
            img_crop = img_proc.mri_img[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]

            cntr = cv.cvtColor(img_crop, cv.COLOR_BGR2BGRA)
            cntr[:,:,3] = mask_crop

            tumr_list.append(cntr)

        counter = 1
        for i in tumr_list:
            cv.imread(i)
            cv.imshow("Edged Image" + counter)
            
            counter += counter'''
        
#-----------------------------------------------------

    def draw_contours(self, img):
        cv.drawContours(img, self.cnts, -1, (0, 0, 255), 2)

    def compute_cross_sectional_area(self):
        cnt = self.cnts[0]
        area = cv.contourArea(cnt)
        self.area = area

    def compute_contour_areas_list(self):
        cntr_areas_list = []
        for cntr in self.cnts:
            area = cv.contourArea(cntr)
            cntr_areas_list.append(area)
        self.cntr_areas_list = cntr_areas_list

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

    contour_detector.compute_contour_areas_list()
    df = pd.DataFrame(contour_detector.cntr_areas_list, columns = ["Contour Areas"])
    print ("data frame", df)

    #-----------------------------------------------------

   # contour_detector.write_contour_files()

    #-----------------------------------------------------
    
    cv.imshow("MRI Image with Contours", img_proc.mri_img)
    display()
