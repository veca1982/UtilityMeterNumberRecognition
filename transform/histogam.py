__author__ = 'Krtalici'

import cv2
import mahotas

def histogram_eq_on_color_image_ret_col_img( image ):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    b,g,r = cv2.split(image)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    eq = cv2.merge((b,g,r))
    return eq


def histogram_eq_on_color_image_ret_gry_img( image ):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(image)
    return eq


def histogram_eq_on_gray_image( image ):
    eq = cv2.equalizeHist(image)
    return eq

def sharpImage( image ):
    image = cv2.GaussianBlur(image, (5,5), 0)
    cv2.addWeighted(image,2.0, image, -0.5, 0)
    return image

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


if __name__ == "__main__":
    image = cv2.imread('../watermeter2.png')
    cv2.imshow("Original", image)
    image = histogram_eq_on_color_image_ret_col_img( image )
    cv2.imshow("Hist EQ", image)
    image = sharpImage( image )
    cv2.imshow("Sharp", image)
    cv2.waitKey()

