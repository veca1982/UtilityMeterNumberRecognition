__author__ = 'Krtalici'
import cv2
import mahotas
import joblib
from skimage.feature import hog
import numpy as np
from image_preprocesing import image_tresholding as tresh
from image_preprocesing import image_enhancment as ie
from image_preprocesing import filter_countours as fc
from processor.second_stage.second_stage import SeconStage
from processor.first_stage.first_stage import FirstStage

from display import display
from metrics import utilityNumbers as un
from metrics import uitility as util
from transform import transform
import sys
import imutils





def fit_bounding_box_on_countour( cnts ) :
    for c in cnts :
        (x, y, w, h) = cv2.boundingRect( c )
        print (x, y, w, h)




if __name__ == "__main__":
    #classifier
    clf = joblib.load("../svm/digits_cls_printed.pkl")

    #1. nadji konture
    image = cv2.imread('../watermeter2.png')
    adjusted = ie.adjust_gamma( image, 1.5 )
    imageGray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    first_stage_processor = FirstStage(image)
    first_stage_processor.crop_image()
    first_stage_processor.apply_filtering_and_tresholding()

    (cnts,hierarchy) = first_stage_processor.getCountoursAndHierachy()
    #1. nadji konture na zabluranoj morphanoj slici
    #2. izostri osnovnu sliku i stavi bound rectangle od zablurane slike na izostrenu sliku
    #3. od bound rectanglova iz zablurane slike nadji 5 onih koji imaju slicnu povrsinu
    #4. ako smo isprano uzeli znamenke iz ROi-ja dobiveni 1 korakom, opet nadji konture u ROI-u i uzmi konturu s najvecom povrsinom

    sharpened = ie.getHighBoostSharpImage( adjusted )
    treshed = tresh.treshImageOtsuWithCorrection( sharpened, -15 )

    for cnt in cnts:
        (x,y,w,h) = cv2.boundingRect(cnt[0])
        cv2.rectangle(sharpened,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Wer",sharpened)
        cv2.waitKey(0)


    (cnts, hierarchy) = cv2.findContours(treshed.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #print hierarchy[0][1]
    #cv2.boundingRect(c) vraca (x,y,w,h)
    cnts = fc.filterByHierarchy(cnts,hierarchy)
    #print "Duljina"+str(len(cnts))

    cnts = sorted([(c, cv2.boundingRect(c)) for c in cnts], key = lambda x: (x[1][0]) )
    cnts = list(filter(lambda x: (x[1][2] > 20 and x[1][3] > 20 and x[1][2] < 150 and x[1][3] < 150), cnts) )

    cnts = util.returnMostLikleyRects( cnts )
    for cnt in cnts:
        print cnt[1]
        (x,y,w,h) = cnt[1]
        #print "X je "+str(h)
        roi_tresh = treshed[y : y + h, x : x + w]
        cv2.imshow("First stage processing",treshed[y : y + h, x : x + w])
        cv2.waitKey(0)
    sys.exit(0)

    #display.display_images( list_of_minrected_segments )
    #sys.exit(0)


    display.display_images( most_likly_digits )

    #sys.exit(0)
    #2. identificiraj brojeve
i = 0
for minrected_segment,max_countour in zip( most_likly_digits, list_of_max_countours ):

    minrectic = cv2.minAreaRect(max_countour)
    boxic = cv2.cv.BoxPoints(minrectic)
    boxic = np.int0(boxic)


    threshedRoi = minrected_segment.copy()

    T = mahotas.thresholding.otsu(threshedRoi)
    threshedRoi[threshedRoi > T] = 255
    threshedRoi[threshedRoi < T] = 0
    threshedRoi = cv2.bitwise_not(threshedRoi)
    threshedRoi = cv2.cvtColor(threshedRoi, cv2.COLOR_BGR2GRAY)
	#mora biti veca slika od 28 X 28
    if threshedRoi.shape[:2][0] > 20 and threshedRoi.shape[:2][1] > 20:
        #threshedRoi = transform.four_point_transform(threshedRoi,boxic)
        #threshedRoi = transform.deskew2(threshedRoi)
        threshedRoi = transform.center_extent(threshedRoi,(28,28))
		#eventualno mozda napravit skeleton od slova
        threshedRoi = cv2.resize(threshedRoi, (28, 28), interpolation=cv2.INTER_AREA)
    	#cv2.imwrite( '../trainingData/train_new_'+str(i)+'.jpg', threshedRoi )
        cv2.imshow("Digit", threshedRoi)
        cv2.waitKey()

        roi_hog_fd = hog(threshedRoi , orientations=22, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=False)
        roi_hog_fd = np.array([roi_hog_fd], 'float64')

        #print roi_hog_fd.shape
        nbr = clf.predict(roi_hog_fd)
        print "Broj je %d"%nbr

	i=i+1


    #3.


