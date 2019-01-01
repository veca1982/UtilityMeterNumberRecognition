__author__ = 'Krtalici'
import cv2
import mahotas

def treshImageOtsu(roi):
    thresh = roi.copy()
    T = mahotas.thresholding.otsu(roi)
    print T
    thresh[thresh > T] = 255
    thresh[thresh < T] = 0
    thresh = cv2.bitwise_not(thresh)
    return thresh

def treshImageOtsuWithCorrection( roi, correction ):
    thresh = roi.copy()
    T = mahotas.thresholding.otsu(roi)
    print T
    thresh[thresh > T+correction] = 255
    thresh[thresh < T+correction] = 0
    thresh = cv2.bitwise_not(thresh)
    return thresh

def treshImageWithTreshold(roi, T):
    thresh = roi.copy()
    thresh[thresh > T] = 255
    thresh[thresh < T] = 0
    thresh = cv2.bitwise_not(thresh)
    return thresh
