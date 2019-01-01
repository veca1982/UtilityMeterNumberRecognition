from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
from svm.model import LinearSVC_proba
__author__ = 'Krtalici'



# Load the dataset
#dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
#features = np.array(dataset.data, 'int16')
#labels = np.array(dataset.target, 'int')

digits = np.loadtxt('G:/trainingData/generalsamples-3-4.data', np.float32)
labels = np.loadtxt('G:/trainingData/generalresponses-3-4.data', np.float32)
#responses = labels.reshape((labels.size,1))

# Extract the hog features
list_hog_feature = []
for digit in digits:
    digit.reshape((28, 28))
    hog_feature = hog(digit.reshape((28, 28)), orientations=18, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                      visualise=False, normalise=True)
    list_hog_feature.append(hog_feature)
    #list_hog_fd.append(feature)
hog_features = np.array(list_hog_feature, 'float64')
#pix_features = np.array(list_hog_fd, 'float64')
print "Count of digits in dataset", Counter(labels)

# Create an linear SVM object
clf = LinearSVC(C=0.1, dual=False, random_state=42)
#clf = LinearSVC_proba(C=0.1, dual=False, random_state=42)

# Perform the training
clf.fit(hog_features, labels)
#clf.fit(pix_features, labels)

# Save the classifier
joblib.dump(clf, "digits_all_cls-5-4.pkl", compress=3)
