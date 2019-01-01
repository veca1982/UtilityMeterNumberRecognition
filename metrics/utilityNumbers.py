__author__ = 'Krtalici'
import numpy as np
import cv2
import uitility as ut
from processor.model import models


def getMostLikelyUtilityDigits(segment_list):
    index_with_min_difrence = 0
    min_variance = 0.0
    temp_min_variance = 0.0
    most_likly_digits = []

    #sortiraj po povrsini
    segment_list_sorted = sorted([segment for segment in segment_list], key=lambda tup: (tup.shape[0] * tup.shape[1]))
    i = 0
    for segment in segment_list_sorted:
        #print segment.shape
        if len(segment_list_sorted) > i+5:
            if i == 0 :
                min_variance = variance(segment_list_sorted, segment, i)
                temp_min_variance = min_variance

            else:
                temp_min_variance = variance(segment_list_sorted, segment, i)

            #print temp_min_variance

            if min_variance > temp_min_variance:
                min_variance = temp_min_variance
                index_with_min_difrence = i


        i = i + 1
    #print index_with_min_difrence
    segment_list_sorted = segment_list_sorted[index_with_min_difrence:index_with_min_difrence+5]

    for segment in segment_list:
        #print segment.shape
        for segment_sorted in segment_list_sorted:
            if segment.shape == segment_sorted.shape:
                if (segment - segment_sorted).any() == 0:
                    most_likly_digits.append(segment)

    return most_likly_digits


def getMostLikelyRois(roisAndCordinatesSortedByX):
    average_h = __get_avarage_h(roisAndCordinatesSortedByX)
    roisAndCordinatesSortedByY = sorted([roiAndCordinate for roiAndCordinate in roisAndCordinatesSortedByX], key=lambda x: x.y)
    groupOfRoisAndCordinatesSimilarByY = []
    groupsOfRoisAndCordinatesSimilarByY = models.RoiAndCordinatesGroupsOffFiveByY()

    for roiAndCordinate in roisAndCordinatesSortedByX:
        roi_cordinate_index = roisAndCordinatesSortedByY.index(roiAndCordinate)
        groupOfRoisAndCordinatesSimilarByY.append(roiAndCordinate)
        if roi_cordinate_index > 0:
            for i in range(roi_cordinate_index-1, -1, -1):
                if roiAndCordinate.y - roisAndCordinatesSortedByY[i].y < average_h/2:
                    #print roiAndCordinate.id, roisAndCordinatesSortedByY[i].id
                    groupOfRoisAndCordinatesSimilarByY.append(roisAndCordinatesSortedByY[i])
                else:
                    break
        if roi_cordinate_index < len(roisAndCordinatesSortedByY)-1:
            for i in range(roi_cordinate_index+1, len(roisAndCordinatesSortedByY)-1, 1):
                if roisAndCordinatesSortedByY[i].y - roiAndCordinate.y < average_h/2:
                    #print roiAndCordinate.id, roisAndCordinatesSortedByY[i].id
                    groupOfRoisAndCordinatesSimilarByY.append(roisAndCordinatesSortedByY[i])
                else:
                    break

        if len(groupOfRoisAndCordinatesSimilarByY) >= 5:
            groupsOfRoisAndCordinatesSimilarByY.groups.append(groupOfRoisAndCordinatesSimilarByY)
            #for roisAndCordinates in groupOfRoisAndCordinatesSimilarByY:
                #cv2.imshow('ROI', roisAndCordinates.roi)
                #print roi_cordinate_index, roisAndCordinates.x, roisAndCordinates.y, roisAndCordinates.w, roisAndCordinates.h, roisAndCordinates.id
                #cv2.waitKey(0)
        groupOfRoisAndCordinatesSimilarByY = []

    #sad filtriranje po povrsini
    #most_likely_rois = groupsOfRoisAndCordinatesSimilarByY.get_most_occuring_rois()
    most_likely_rois = groupsOfRoisAndCordinatesSimilarByY.get_most_likely_rois_by_area()
    return most_likely_rois




def getMostLikelyUtilityDigits4( segment_list , list_boxics ):
    most_likly_digits = []
    breakOuterLoop = False

    list_tuples = [ ( segment, box ) for segment, box in zip( segment_list, list_boxics ) ]
    print list_tuples[0][1][2][0][1] #dolazak do kordinate
    for i in range(0,len(list_tuples)):
        most_likly_digits.append(list_tuples[i][0])
        for j in range(i+1, len(list_tuples)) :
            if len(list_tuples) > i + 5 :
                x_cordinate = list_tuples[i][1][2][0][0] - list_tuples[j][1][2][0][0]
                y_cordinate = list_tuples[i][1][2][0][1] - list_tuples[j][1][2][0][1]
                if x_cordinate < 0 :
                   x_cordinate = -1 * x_cordinate
                if y_cordinate < 0 :
                   y_cordinate = -1 * y_cordinate
                v1 = np.array([x_cordinate,y_cordinate])
                v2 = np.array([0,1])
                if ( ut.calculateAngleBetweenTwoVectors(v1,v2) < 10 ):
                    most_likly_digits.append(list_tuples[j][0])
                    if len(most_likly_digits) == 5:
                        breakOuterLoop = True
                        break

        if breakOuterLoop :
            break
        else:
            most_likly_digits = []

    return most_likly_digits

#nadji 5 rectova za redom koji imaju izmedju x ose i vektora koji spaja pocetnu i odredisnu tocku
def getMostLikelyRects( segment_list , list_boxics ):
    most_likly_digits = []
    min_diffrence = 0
    index_with_min_difrence = 0

    list_tuples = [ ( segment, box ) for segment, box in zip( segment_list, list_boxics ) ]
    print list_tuples[0][1][2][0][0] #dolazak do kordinate
    i = 0
    for tuple in list_tuples:
        if len(list_tuples) > i + 5 :
            #print tuple[1]
            if i == 0 :
                min_diffrence = (5*tuple[1][2][0][0] - list_tuples[i+1][1][2][0][0] - list_tuples[i+2][1][2][0][0] - list_tuples[i+3][1][2][0][0] - list_tuples[i+4][1][2][0][0] - list_tuples[i+5][1][2][0][0])**2
                temp_min_diffrence = min_diffrence
            else:
                temp_min_diffrence = (5*tuple[1][2][0][0] - list_tuples[i+1][1][2][0][0] - list_tuples[i+2][1][2][0][0] - list_tuples[i+3][1][2][0][0] - list_tuples[i+4][1][2][0][0] - list_tuples[i+5][1][2][0][0])**2
            if min_diffrence > temp_min_diffrence:
                min_diffrence = temp_min_diffrence
                index_with_min_difrence = i
        i = i + 1

    most_likly_digits = segment_list[index_with_min_difrence:index_with_min_difrence+5]
    return most_likly_digits



def variance(segment_list_sorted,segment,i) :
    segment_list_sorted_shape = np.asarray([segment.shape for segment in segment_list_sorted])
    #print segment_list_sorted_shape[i:i+5]
    average = sum(segment_list_sorted_shape[i:i+5]) / len(segment_list_sorted_shape[i:i+5])
    varience = sum((average - segment_shape) ** 2 for segment_shape in segment_list_sorted_shape[i:i+5]) / len(segment_list_sorted_shape[i:i+5])
    return float (varience[0]*varience[1])/(average[0]*average[1])

def __get_avarage_h(roisAndCordinatesSortedByX):
    return sum(roiAndCordinate.h for roiAndCordinate in roisAndCordinatesSortedByX) / len(roisAndCordinatesSortedByX)



if __name__ == "__main__":
    print "Ovo je test"
