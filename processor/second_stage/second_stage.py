__author__ = 'Krtalici'
#for ROI given by filtering of the first stage preprocwssing we find new countours in every ROI and taka the one that surrounds
#maximum Area

from processor.prepocessor import Preprocessor
import mahotas
import cv2
import numpy as np
from image_preprocesing import image_tresholding as tresh





class SeconStage(Preprocessor):
    def __init__(self,cnts,color_image_resized,grayed_image_resized):
        self.cnts = cnts # konture dobivene morphom i tresholdom
        self.cntsInRoi = None
        self.color_image_resized = color_image_resized
        self.grayed_image_resized = grayed_image_resized
        self.list_of_max_area_in_roi = []
        self.list_of_max_countours = []

    def findCntsInRoi(self):
        for (c,_) in self.cnts:
            (x, y, w, h) = cv2.boundingRect(c)

            if h > 300 and w > 300 and h < 30 and w < 30:
                pass


            if w >= 20 and h >= 20 and h < 200 and w < 200:
                #print x,y,h,w
                roi = self.grayed_image_resized[y : y + h, x : x + w]
                roicolor = self.color_image_resized[y : y + h, x : x + w]

                thresh = tresh.treshImageOtsuWithCorrection( roi, -15 )


                roiGrayed = cv2.cvtColor(roicolor, cv2.COLOR_BGR2GRAY)

                #cv2.imshow("Roi edged", roiGrayed)
                #cv2.waitKey()

                (self.cntsInRoi, _) = cv2.findContours(roiGrayed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #self.cntsInRoi = sorted([(cRoi, cv2.boundingRect(cRoi)[0]) for cRoi in self.cntsInRoi], key = lambda x: x[1])

                maxAreaCountour = 0
                countredWithMaxArea = None
                #uzimamo konture s najvecom povrsinom, solidna pretpostavka
                for cRoi in self.cntsInRoi:
                    if cv2.contourArea(cRoi) > maxAreaCountour:
                        countredWithMaxArea = cRoi
                        maxAreaCountour = cv2.contourArea(cRoi)

                #ako se pokrade slika koja nema dimenzija
                if countredWithMaxArea is not None:
                    [xi,yi,wi,hi] = cv2.boundingRect(countredWithMaxArea)
                    cv2.rectangle(roicolor,(xi,yi), (xi+wi,yi+hi), (0,255,0), 1)
                    roiColorResized = roicolor[yi : yi + hi, xi : xi + wi]
                    if wi > 20 and hi > 20:
                        self.list_of_max_area_in_roi.append( roiColorResized )
                        self.list_of_max_countours.append( countredWithMaxArea )

        return (self.list_of_max_area_in_roi,self.list_of_max_countours)

    def findCntsInRoi2(self):
        for (c, _) in self.cnts:
            (x, y, w, h) = cv2.boundingRect(c)

            if h > 300 and w > 300 and h < 40 and w < 40:
                pass


            if w >= 20 and h >= 20 and h < 200 and w < 200:
                roi = self.grayed_image_resized[y : y + h, x : x + w]
                roicolor = self.color_image_resized[y : y + h, x : x + w]

                roiGrayed = cv2.cvtColor(roicolor, cv2.COLOR_BGR2GRAY)

                cv2.imshow("Roi edged", roiGrayed)
                cv2.waitKey()

                coubtourArea = cv2.contourArea(c)

                if coubtourArea is not None:
                    self.list_of_max_area_in_roi.append( roicolor )
                    self.list_of_max_countours.append( c )

        return (self.list_of_max_area_in_roi,self.list_of_max_countours)



