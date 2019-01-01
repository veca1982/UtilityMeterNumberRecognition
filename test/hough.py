__author__ = 'Krtalici'

import cv2
import numpy as np
from processor.first_stage.first_stage import FirstStage
from image_preprocesing import image_enhancment as ie


img = cv2.imread('../watermeter14.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adjusted = ie.adjust_gamma(img, 1.5)
imageGray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
first_stage_processor = FirstStage(img)
first_stage_processor.crop_image()
first_stage_processor.apply_filtering_and_tresholding()
img_processed = first_stage_processor.filtered_image
edges = cv2.Canny(img_processed, 50, 150, apertureSize=3)
cv2.imshow("Edges", edges)
cv2.waitKey(0)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*a)
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*a)

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('houghlines3.jpg', img)