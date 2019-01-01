__author__ = 'Krtalici'
import cv2

def filterByHierarchy(cnts,hierarchy):
    cntsWithMAxHierarchy = []
    for i,cnt in enumerate(cnts):
        if hierarchy[0,i,3] == -1  and cv2.contourArea(cnt[0]) > 80:
            cntsWithMAxHierarchy.append(cnt)

    return cntsWithMAxHierarchy




