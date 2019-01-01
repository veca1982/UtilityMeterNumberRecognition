__author__ = 'Krtalici'

from processor.prepocessor import Preprocessor
import cv2
import mahotas
import  numpy as np
from image_preprocesing import image_enhancment as ie
from image_preprocesing import image_tresholding as tresh


class FirstStage(Preprocessor):
    def __init__(self, original_image):
        super(FirstStage, self).__init__(original_image)
        self.filtered_image = None

    def apply_filtering_and_tresholding(self):
        #equ = self.__smoothing_homomorphic_filtering_hist_eq()

        eclipseKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        np.array([(0, 0, 1, 0, 0),
		  (1, 1, 1, 1, 1),
		  (1, 1, 1, 1, 1),
		  (1, 1, 1, 1, 1),
		  (0, 0, 1, 0, 0)])

        #filter 'kernel' for morph filtering-Rect type (5,5)
        rectangleKernel = np.ones((5, 5), np.uint8)
        gradient = cv2.morphologyEx(self.resized_grayed_image, cv2.MORPH_GRADIENT, eclipseKernel)
        #gradient = cv2.morphologyEx(equ, cv2.MORPH_GRADIENT, eclipseKernel)
        #cv2.imshow("Morpho Gradient", gradient)

        #higly blurring image to to have big spots who we are going to rectangled
        conneted = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, rectangleKernel)
        #cv2.imshow("MORPH_CLOSE", conneted)

        #tresholding pixel int less than 155 white, more than 155 black
        T = mahotas.thresholding.otsu(conneted)
        thresh = conneted.copy()
        thresh[thresh > T] = 255
        thresh[thresh < 255] = 0
        threshInv = cv2.bitwise_not(thresh)
        cv2.imshow("Plinomjer", threshInv)
        cv2.waitKey(0)
        self.filtered_image = threshInv

    def apply_filtering_and_tresholding2(self):
        adjusted = ie.adjust_gamma(self.original_image, 1.2)
        imageGray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        sharpened = ie.getHighBoostSharpImage(adjusted)
        treshed = tresh.treshImageOtsuWithCorrection(sharpened, 13)
        cv2.imshow("Treshed", treshed)
        cv2.waitKey()
        self.filtered_image = treshed

    def getCountoursAndHierachy(self):
        (self.cnts, self.hierarchy) = cv2.findContours(self.filtered_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = sorted([(c, cv2.boundingRect(c)) for c in self.cnts], key = lambda x: x[1][0])

        return (self.cnts, self.hierarchy)

    def __smoothing_homomorphic_filtering_hist_eq(self):
        img_noiese_removed = cv2.bilateralFilter(self.resized_grayed_image, 9, 75, 85)
        homomorphed = ie.homomorphic_filter(img_noiese_removed, gamma1=0.6, gamma2=2.2)
        return cv2.equalizeHist(homomorphed)

