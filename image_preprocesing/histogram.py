__author__ = 'Krtalici'
import cv2
import numpy as np
from image_preprocesing import image_tresholding as tresh
from image_preprocesing import image_enhancment as ie

def get_grey_hist_eq( image ):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(image)
    return eq

def sharp_img(path_to_image):

    #Load source / input image as grayscale, also works on color images...
    imgIn = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Original", imgIn)
    blur = cv2.bilateralFilter(imgIn, 9, 75, 75)
    #blur = cv2.medianBlur(imgIn, 7)

    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0   #Identity, times two!

    #Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0

    #Subtract the two:
    kernel = kernel - boxFilter


    #Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    custom = cv2.filter2D(blur, -1, kernel)
    cv2.imshow("Sharpen", custom)

    customSharp = cv2.filter2D(custom, -1, kernel)
    cv2.imshow("Custom Sharpen", customSharp)

    treshed = tresh.treshImageOtsuWithCorrection(customSharp, 0)
    cv2.imshow('Treshed', treshed)
    cv2.waitKey(0)

def sharp_that_image(grey_image):

    #Load source / input image as grayscale, also works on color images...
    cv2.imshow("Original", grey_image)
    blur = cv2.bilateralFilter(grey_image, 9, 75, 75)
    #blur = cv2.medianBlur(imgIn, 7)

    #Create the identity filter, but with the 1 shifted to the right!
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0   #Identity, times two!

    #Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0

    #Subtract the two:
    kernel = kernel - boxFilter


    #Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    custom = cv2.filter2D(blur, -1, kernel)
    cv2.imshow("Sharpen", custom)

    #customSharp = cv2.filter2D(custom, -1, kernel)
    #cv2.imshow("Custom Sharpen", customSharp)

    treshed = tresh.treshImageOtsuWithCorrection(custom, 5)
    cv2.imshow('Treshed', treshed)
    cv2.waitKey(0)

if __name__ == "__main__":
    image = cv2.imread('../watermeter21.png')
    eq = get_grey_hist_eq(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Histogram Equalization", np.hstack([image, eq]))
    cv2.waitKey(0)
    sharp_that_image(grey_image=eq)