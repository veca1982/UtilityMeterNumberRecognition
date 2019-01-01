__author__ = 'Krtalici'

import cv2
import image_tresholding as tresh
import numpy as np
import scipy.fftpack # For FFT2
from skimage import exposure


def sharp_img_with_filter(image):
    #grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Create the identity filter, but with the 1 shifted to the right!
    grey = image.copy()
    kernel = np.zeros((9, 9), np.float32)
    kernel[4, 4] = 2.0   #Identity, times two!

    #Create a box filter:
    boxFilter = np.ones((9, 9), np.float32) / 81.0

    #Subtract the two:
    kernel = kernel - boxFilter

    #Note that we are subject to overflow and underflow here...but I believe that
    # filter2D clips top and bottom ranges on the output, plus you'd need a
    # very bright or very dark pixel surrounded by the opposite type.

    custom = cv2.filter2D(grey, -1, kernel)
    customSharpned = cv2.filter2D(custom, -1, kernel)

    return customSharpned


def homomorphic_filter(img, gamma1, gamma2):
    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

    # Set scaling factors and add

    Iout = gamma1*Ioutlow[0:rows, 0:cols] + gamma2*Iouthigh[0:rows,0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")

    return Ihmf2


def sharpImageWithLaplacian(image):
    kernel_size = 3
    scale = 0
    delta = 0
    ddepth = cv2.CV_16S

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_lap = cv2.Laplacian(gray, ddepth, ksize=kernel_size, scale=scale, delta=delta)

    dst = cv2.convertScaleAbs(gray_lap)

    grayBlur = cv2.GaussianBlur(gray, (5, 5), 0)

    sharpend = grayBlur + dst

    return sharpend


def sharpImage(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.addWeighted(image, 2.0, image, -0.5, 0)

    return image


def getHighBoostSharpImage( image ):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(image, (3, 3), 0)
    grayBlur = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
    #grayBlur = histogam.histogram_eq_on_gray_image(imageGray)
    mask = imageGray - grayBlur
    sharpened = imageGray + mask

    return sharpened


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def factory_method_processing(image, method_name):
    if method_name == "homomorphic":
        return remove_noise_and_get_grey_img(image)
    else:
        return getHighBoostSharpImage(image)


def remove_noise_and_get_grey_img(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #img_noiese_removed = cv2.bilateralFilter(imageGray, 9, 75, 75)
    img_homomorphed = homomorphic_filter(imageGray, gamma1=0.6, gamma2=2.2)
    p2, p98 = np.percentile(img_homomorphed, (2, 95))
    print p2, p98
    img_rescale = exposure.rescale_intensity(img_homomorphed, in_range=(p2, p98))
    img_loged = adjust_gamma(img_rescale, gamma=2.0)
    cv2.imshow('Contrasted', img_loged)
    cv2.waitKey(0)
    img_noiese_removed = cv2.bilateralFilter(img_loged, 9, 75, 95)
    #img_noiese_removed = cv2.medianBlur(img_loged, 5)
    cv2.imshow('Not noisy', img_noiese_removed)
    cv2.waitKey(0)
    img_to_return = sharp_img_with_filter(img_noiese_removed)
    cv2.imshow('ImgToReturn', img_noiese_removed)
    cv2.waitKey(0)
    return img_to_return

if __name__ == "__main__":
    image = cv2.imread('../watermeter22.png', 0)
    img = homomorphic_filter(image, gamma1=1.0, gamma2=1.5)
    cv2.imshow('homo', img)
    cv2.waitKey(0)

if __name__ == "__main2__":
    list_of_max_countours = []

    image = cv2.imread('../watermeter2.png')
    adjusted = adjust_gamma( image, 1.5 )
    imageGray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    sharpened = getHighBoostSharpImage( adjusted )
    treshed = tresh.treshImageOtsuWithCorrection( sharpened, -15 )

    sharp_grey_laplacian = sharpImageWithLaplacian( image )
    treshedLaplacian = tresh.treshImageOtsuWithCorrection( sharp_grey_laplacian, -19)

    (cnts, hierarchy) = cv2.findContours(treshed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(adjusted, cnts,-1, (0, 255, 0), 1)
    sortedCntByLeftToRight = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])

    c = max( cnts, key=cv2.contourArea )
    cv2.drawContours(adjusted, [c], 0, (0, 255, 0), 2)


    for (cnt,_) in sortedCntByLeftToRight :
        (x, y, w, h) = cv2.boundingRect(cnt)
        if w > 20 and h > 20 :
            #cv2.imshow("Roi", adjusted[y : y + h, x : x + w] )
            #cv2.waitKey(0)
            # find the largest contour in bounding rect
            c = max(cnt, key=cv2.contourArea)
            if len(c) > 0 :
                list_of_max_countours.append( np.array([c]) )
                cv2.drawContours( adjusted, [cnt], 0, (0, 0, 255), 2 )

    #cv2.imshow("Trashed", np.hstack( [treshed, treshedLaplacian] ) )

    #cv2.imshow("Original sharpned", np.hstack( [imageGray, sharpened] ) )

    #cv2.drawContours(adjusted, list_of_max_countours, -1, (0, 255, 0), 3)

    cv2.imshow("Original adjusted", np.hstack( [image, adjusted] ) )


    cv2.waitKey(0)

    #print 'Broj contura '+ str( len(cnts) ) + ' ' + str( len(hierarchy) )

