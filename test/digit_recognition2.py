import cv2
import joblib
from skimage.feature import hog
import numpy as np
from image_preprocesing import image_tresholding as tresh
from image_preprocesing import image_enhancment as ie
from processor.first_stage.first_stage import FirstStage
from metrics import utilityNumbers as un
from metrics import uitility as util

__author__ = 'Krtalici'



if __name__ == "__main__":
    #classifier
    #clf = joblib.load("../svm/digits_cls_printed.pkl")
    clf = joblib.load("../svm/digits_all_cls-3-4.pkl")
    #clf = joblib.load("../svm/digits_all_cls.pkl")
    #1. nadji konture
    image = cv2.imread('../watermeter15.jpg')
    #image = cv2.imread('G:/trainingData/imgTakenFromWtermeter/slika665.jpg')
    adjusted = ie.adjust_gamma(image, 1.2)

    first_stage_processor = FirstStage(image)
    first_stage_processor.crop_image()
    first_stage_processor.apply_filtering_and_tresholding()

    #1. nadji konture na zabluranoj morphanoj slici
    #2. izostri osnovnu sliku i stavi bound rectangle od zablurane slike na izostrenu sliku
    #3. od bound rectanglova iz zablurane slike nadji 5 onih koji imaju slicnu povrsinu
    #4. ako smo isprano uzeli znamenke iz ROi-ja dobiveni 1 korakom, opet nadji konture u ROI-u i uzmi konturu s najvecom povrsinom

    (cnts, hierarchy) = first_stage_processor.getCountoursAndHierachy()

    #2. izostri osnovnu sliku i stavi bound rectangle od zablurane slike na izostrenu sliku
    #sharpened = ie.getHighBoostSharpImage(adjusted)
    sharpened = ie.factory_method_processing(adjusted, 'homomorphic')
    treshed = tresh.treshImageOtsuWithCorrection(sharpened, -16)
    cv2.imshow('Treshed', treshed)
    cv2.waitKey(0)
    cnts = list(filter(lambda x: (x[1][2] > 46 and x[1][3] > 46 and x[1][2] < 200 and x[1][3] < 200), cnts))
    #rois = util.getRoisFromCnts(cnts, treshed)
    roisAndCordinates = util.getRoiAndCourdinatesInWholeImageFromCnts(cnts, treshed)
    most_likely_roisAndCordinates = un.getMostLikelyRois(roisAndCordinates)
    #for roidAndCordinate in most_likely_roisAndCordinates:
        #cv2.imshow('ROI', roidAndCordinate.roi)
        #print roidAndCordinate.x, roidAndCordinate.y, roidAndCordinate.w, roidAndCordinate.h, roidAndCordinate.id
        #cv2.waitKey(0)

    #exit(0)
    #for roi in rois:
        #cv2.imshow('ROI', roi)
        #cv2.waitKey(0)

    #3. od bound rectanglova iz zablurane slike nadji 5 onih koji imaju slicnu povrsinu i varijancu povrsine
    #most_likely_digits = un.getMostLikelyUtilityDigits(rois)
    most_likely_digits = []
    for roidAndCordinate in most_likely_roisAndCordinates:
        most_likely_digits.append(roidAndCordinate.roi)


    #for digit in most_likely_digits:
        #cv2.imshow('Digit', digit)
        #cv2.waitKey(0)
    #samples = np.empty((0, 28 * 28))
    #responses = [0, 0, 0, 1, 1]
    #4. ako smo isprano uzeli znamenke iz ROi-ja dobiveni 1 korakom, opet nadji konture u ROI-u i uzmi konturu s najvecom povrsinom
    for most_likely_digit in most_likely_digits:
        cnts = cv2.findContours(most_likely_digit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        digit_roi = most_likely_digit[y: y + h, x: x + w]
        #cv2.imshow("Digit ROI", digit_roi)
        #cv2.waitKey(0)
        #digit_roi = transform.deskew2(digit_roi)
        #digit_roi = transform.center_extent(digit_roi,(28,28))
        digit_roi = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
        #digit_roi = cv2.resize(digit_roi, (21, 28), interpolation=cv2.INTER_AREA)
        cv2.imshow('Digit ', digit_roi)
        cv2.waitKey()
        #sample = digit_roi.reshape((1, 28 * 28))
        #samples = np.append(samples, sample, 0)

        roi_hog_fd = hog(digit_roi, orientations=18, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=False, normalise=True)
        roi_hog_fd = np.array([roi_hog_fd], 'float64')

        #print roi_hog_fd.shape
        nbr = clf.predict(roi_hog_fd)
        print 'Broj je %d'%nbr

    #responses = np.array(responses, np.int)
    #responses = responses.reshape((responses.size, 1))

    #samplesData = open('G:/trainingData/generalsamples-3-4.data', 'ab')
    #responseData = open('G:/trainingData/generalresponses-3-4.data', 'ab')

    #np.savetxt(samplesData, samples)
    #np.savetxt(responseData, responses)






