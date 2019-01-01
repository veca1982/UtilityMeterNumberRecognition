__author__ = 'Krtalici'
import imutils
import cv2
from image_preprocesing import image_enhancment as ie

class Preprocessor(object):
    def __init__(self, original_image):
        self.original_image = original_image
        self.cnts = []
        self.hierarchy = []
        self.resized_image = None
        self.resized_grayed_image = None

    def crop_image(self):
        self.resized_image = imutils.resize(self.original_image, width=640)
        self.resized_grayed_image = cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2GRAY)
        #self.resized_grayed_image = ie.sharp_img_with_filter(self.resized_image)
        #self.resized_grayed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

