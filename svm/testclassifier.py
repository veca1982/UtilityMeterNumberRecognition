# Import the modules
import cv2
import joblib
from skimage.feature import hog
import numpy as np
from transform import transform
from os import walk

# Load the classifier
#clf = joblib.load("digits_cls_printed.pkl")
#clf = joblib.load("digits_cls.pkl")
#clf = joblib.load("digits_1_cls.pkl")
def test_image_by_image(image_url):
    clf = joblib.load("digits_all_cls-4-4.pkl")

    im = cv2.imread(image_url, 0)
    roi = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    #roi = cv2.dilate(roi, (0.1, 0.1))
    #roi = transform.deskew2(roi)
    #roi = transform.skeleton(roi)

    #cv2.imshow("Roi", roi)
    #cv2.waitKey()

    roi_hog_fd = hog(roi.reshape((28, 28)), orientations=18, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                     visualise=False, normalise=True)
    roi_hog_fd = np.array([roi_hog_fd], 'float64')

    #print roi_hog_fd.shape
    nbr = clf.predict(roi_hog_fd)

    #print nbr
    return int(nbr[0])

def test_all_images_in_folder(folder_url, test_images):
    clf = joblib.load("digits_all_cls-4-4.pkl")


def get_images_ion_folder(folder_url):
    f = []
    for (dirpath, dirnames, filenames) in walk(folder_url):
        f.extend(filenames)
        #samo prvu iteraiju trebamo
        #break
    return f


def get_labeled_value_for_digit_image(image_name):
    print image_name.split('.')[0][-1]
    return int(image_name.split('.')[0][-1])


def get_precission_from_test_images(folder_url):
    image_names = get_images_ion_folder(folder_url)
    #1 ako je labelirano = prediktirano, 0 inace
    good_predicted = []
    for image_name in image_names:
        true_value = get_labeled_value_for_digit_image(image_name)
        predicted_value = test_image_by_image(folder_url+image_name)
        if true_value == predicted_value:
            good_predicted.append(1)
            #print 'Success'
        else:
            good_predicted.append(0)
            print 'Unsuccesfull ', image_name, ', predicted value', predicted_value

    good_predicted = np.asarray(good_predicted, int)
    print 'Tocnost je', float(good_predicted.sum()) / float(good_predicted.shape[0])*100, '%'
    return float(good_predicted.sum()) / float(good_predicted.shape[0])




if __name__ == '__main__':
    #image_url = 'G:/trainingData/capture8/train105_6.jpg'
    #print test_image_by_image(image_url)
    folder_url = 'G:/trainingData/capture8/'
    get_precission_from_test_images(folder_url)