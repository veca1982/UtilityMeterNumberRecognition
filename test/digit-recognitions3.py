# import the necessary packages
import numpy as np
import cv2
from image_preprocesing import image_enhancment as ie

if __name__ == "__main__":
    # load the image, convert it to grayscale, and blur it
    image = cv2.imread('../watermeter21.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    vis = image.copy()
    mser = cv2.MSER()
    regions = mser.detect(image)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.imshow('img', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
