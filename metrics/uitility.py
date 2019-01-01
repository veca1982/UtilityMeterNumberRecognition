__author__ = 'Krtalici'
import cv2
import numpy as np
from processor.model import models

def getCntsWithSameRectXCordinateThatHaveMaxArea( cnts ):
    tempXCordinate = 0
    maxCountourArea = 0
    indexWithMaxCountourArea = 0
    cntsWithMaxCountourArea = []
    i = 0
    for c in cnts :
        if i == 0 :
           tempXCordinate = c[1]
        if tempXCordinate != c[1]:
            cntsWithMaxCountourArea.append( cnts[indexWithMaxCountourArea] )
            #x cordinate
            tempXCordinate = c[1]
            maxCountourArea = cv2.contourArea( c[0] )
            indexWithMaxCountourArea = i
        else:
            if maxCountourArea < cv2.contourArea( c[0] ):
                maxCountourArea = cv2.contourArea( c[0] )
                indexWithMaxCountourArea = i
        i = i + 1

    return cntsWithMaxCountourArea


# definirat cemo pojas u pojas po y u kojem cemo trazit 5 uzastopnih boundinRectova oko kontura
def calculateAngleBetweenTwoVectors( v1, v2 ):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    if x_modulus != 0 and y_modulus != 0:
        cos_angle = dot / x_modulus / y_modulus # cosine of angle between x and y
    else:
        cos_angle = 0.99
    angle = np.arccos(cos_angle)
    return (angle * 360 / 2 / np.pi) # angle in degrees

def calculateCosAngleBetweenTwoVectors( v1, v2 ):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    if x_modulus != 0 and y_modulus != 0:
        cos_angle = dot / x_modulus / y_modulus # cosine of angle between x and y
    else:
        cos_angle = 0.99
    return cos_angle

def returnMostLikleyRects(cnts):
    most_likly_cnts = []
    breakOuterLoop = False

    print cnts[0][1][3] #dolazak do h kordinate cv2.roundingBoxa
    for i in range(0, len(cnts)):
        most_likly_cnts.append(cnts[i])
        for j in range(i+1, len(cnts)) :
            if len(cnts) > i + 5 :
                x_cordinate = cnts[j][1][0] - cnts[i][1][0]
                y_cordinate = cnts[j][1][1] - cnts[i][1][1]
                #if x_cordinate < 0 :
                   #x_cordinate = -1 * x_cordinate
                #if y_cordinate < 0 :
                   #y_cordinate = -1 * y_cordinate
                v1 = np.array([x_cordinate,y_cordinate])
                v2 = np.array([0,1])
                if ( calculateCosAngleBetweenTwoVectors(v1,v2) < 0.05 ):
                    most_likly_cnts.append(cnts[j])
                    if len(most_likly_cnts) == 5:
                        breakOuterLoop = True
                        break

        if breakOuterLoop:
            break
        else:
            most_likly_cnts = []

    return most_likly_cnts

def getRoisFromCnts(cnts, trashed):
    rois = []
    for cnt in cnts:
        (x, y, w, h) = cnt[1]
        roi = trashed[y:y + h, x:x + w]
        rois.append(roi)
    return rois

def getRoiAndCourdinatesInWholeImageFromCnts(cnts, trashed):
    roisAndCordinates = []
    i = 0
    for cnt in cnts:
        (x, y, w, h) = cnt[1]
        roi = trashed[y:y + h, x:x + w]
        roisAndCordinates.append(models.RoiAndCourdinatesInWholeImage(i, roi, x, y, w, h))
        i += 1
    return roisAndCordinates


if __name__ == "__main__":
	print calculateAngleBetweenTwoVectors(np.array([1,0]),np.array([52,24]))




