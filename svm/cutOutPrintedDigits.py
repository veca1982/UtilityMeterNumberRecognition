import sys
import numpy as np
import cv2
from transform import transform

__author__ = 'Krtalici'


1
im = cv2.imread('G:/Ones_Extra.png')
#im = cv2.imread('G:/trainingData/Distoted/With extra Lines/Train_distort_5.png')
im3 = im.copy()
1
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# moramo koopirati da ne bi countouruing sjebo vadjenje slike
threshCopy = thresh.copy()

cv2.imshow('Thresh', thresh)
cv2.waitKey()

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow('ThreshCopy', threshCopy)
# cv2.waitKey()
samples = np.empty((0, 28 * 28))
responses = []
csvArray = []
keys = [i for i in range(48, 58)]
j = 0
for cnt in contours:
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = threshCopy[y:y + h, x:x + w]
            # roi = thresh[y:y+h, x:x+w]
            roismall = cv2.resize(roi, (28, 28))
            #roismall = transform.center_extent(roismall, (28, 28))
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                # cv2.imwrite('G:/trainingData/capture8/train'+str(j)+'_'+chr(key)+'.jpg', roismall)
                sample = roismall.reshape((1, 28 * 28))
                samples = np.append(samples, sample, 0)
                j = j + 1

responses = np.array(responses, np.int)
responses = responses.reshape((responses.size, 1))
print "training complete"

samplesData = open('G:/trainingData/generalsamples-3-4.data', 'ab')
responseData = open('G:/trainingData/generalresponses-3-4.data', 'ab')



np.savetxt(samplesData, samples)
np.savetxt(responseData, responses)
