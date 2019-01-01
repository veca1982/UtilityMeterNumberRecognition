import numpy as np
import cv2
from skimage import data, img_as_float
from skimage import exposure
from image_preprocesing import image_enhancment as ie


if __name__ == "__main__":
    image = cv2.imread('../watermeter21.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('orig', image)
    p2, p98 = np.percentile(image, (2, 97))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    image_to_return = ie.sharp_img_with_filter(img_rescale)
    final = cv2.bilateralFilter(image_to_return, 9, 75, 75)
    p2, p98 = np.percentile(image, (2, 97))
    img_rescale = exposure.rescale_intensity(final, in_range=(p2, p98))
    cv2.imshow('Streched', img_rescale)
    cv2.imwrite('../watermeter23.png', img_rescale)
    cv2.waitKey(0)