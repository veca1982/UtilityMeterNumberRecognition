__author__ = 'Krtalici'

import cv2
import numpy as np
import imutils
from math import atan, pi, ceil

image  =  cv2.imread('../trainingData/train14.jpg', 0)
print image.shape
cv2.imshow('original', image)


# You can read about Image Moments
# Chapter 18 Shape analysis - Page 633
# Digital Image Processing 4th Edition By William K. Pratt
# OR http://en.wikipedia.org/wiki/Image_moment

# grab the width and height of the image and compute
# moments for the image
(h, w) = image.shape[:2]
moments = cv2.moments(image)

# deskew the image by applying an affine transformation
skew = moments["mu11"] / moments["mu02"]
M = np.float32([
    [1, skew, -0.5 * w * skew],
    [0, 1, 0]])
rotated = cv2.warpAffine(image, M, (w, h),
    flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


# return the deskewed image

cv2.imshow('final', rotated)
cv2.waitKey()