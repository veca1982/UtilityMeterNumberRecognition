__author__ = 'Krtalici'
import cv2

def display_images(images):
    for image in images:
        cv2.imshow("Image", image)
        cv2.waitKey(0)


