from __builtin__ import len
import numpy as np


class RoiAndCourdinatesInWholeImage(object):
    def __init__(self, id, roi, x, y, w, h):
        self.id = id
        self.roi = roi
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class RoiAndCordinatesGroupsOffFiveByY(object):
    def __init__(self):
        self.groups = []

    def __filter_grops(self):
        groups = []
        for group in self.groups:
            group_sorted_by_area = sorted([roiAndCordInWholeImage for roiAndCordInWholeImage in group], key=lambda x: x.w * x.h)
            group_sorted_by_area = self.__kick_out_roi_that_is_inside_another_roi(group_sorted_by_area)
            group_sorted_by_area = self.__kick_out_roi_that_is_most_dissimilar(group_sorted_by_area)
            group = sorted([roiAndCordInWholeImage for roiAndCordInWholeImage in group_sorted_by_area], key=lambda x: x.x)
            if len(group) >= 5:
                groups.append(group)
        return groups

    def get_most_occuring_rois(self):
        groups = self.__filter_grops()
        del self.groups
        if len(groups) > 1:
            for i in range(0, len(groups)-1):
                votes = []
                counter_of_full_intersections = 0
                for j in range(i+1, len(groups)-1):
                    intersection_set = set(groups[i]).intersection(groups[j])
                    if len(intersection_set) == 5:
                        counter_of_full_intersections += 1
                votes.append(counter_of_full_intersections)
            if sum(votes) > 0:
                index_with_most_votes = votes.index(max(votes))
                return groups[index_with_most_votes]
            else:
                #sreca prati hrabre :)
                return groups[0]

        elif len(groups) == 1:
            return groups[0]

        else:
            return None


    def __kick_out_roi_that_is_most_dissimilar(self, group_sorted_by_area):
        #average_area_roi_group = sum([group.w * group.h for group in group_sorted_by_area]) / len(group_sorted_by_area)
        while len(group_sorted_by_area) > 5:
            average_area_roi_group = sum([group.w * group.h for group in group_sorted_by_area]) / len(group_sorted_by_area)
            last_index = len(group_sorted_by_area)-1
            if abs(group_sorted_by_area[0].w*group_sorted_by_area[0].h - average_area_roi_group) > abs(group_sorted_by_area[last_index].w*group_sorted_by_area[last_index].h - average_area_roi_group):
                del group_sorted_by_area[0]
            else:
                del group_sorted_by_area[last_index]

        return group_sorted_by_area

    def __kick_out_roi_that_is_inside_another_roi(self, group_sorted_by_area):
        group_sorted_by_area_to_return = list(group_sorted_by_area)
        for roiAndCordinatesinWholeImg in group_sorted_by_area:
            if roiAndCordinatesinWholeImg in group_sorted_by_area_to_return:
                for roiAndCordinatesinWholeImgToReturn in group_sorted_by_area_to_return:
                    if roiAndCordinatesinWholeImg.x < roiAndCordinatesinWholeImgToReturn.x < \
                                    roiAndCordinatesinWholeImg.x + roiAndCordinatesinWholeImg.w:
                        index_to_delete = group_sorted_by_area_to_return.index(roiAndCordinatesinWholeImgToReturn)
                        del group_sorted_by_area_to_return[index_to_delete]

        return group_sorted_by_area_to_return

    def get_most_likely_rois_by_area(self):
        min_variance = 0
        index_to_return = 0
        groups = self.__filter_grops()
        del self.groups
        if len(groups) > 0:
            min_variance = self.__variance(groups[0])
            index_to_return = 0
        else:
            return None
        if len(groups) > 1:
            for i in range(1, len(groups)-1):
                temp_variance = self.__variance(groups[i])
                if temp_variance < min_variance:
                    min_variance = temp_variance
                    index_to_return = i

        return groups[index_to_return]

    def __variance(self, group):
        np_rois_shapes = np.asarray([rois_and_cordin_in_whole_img.roi.shape for rois_and_cordin_in_whole_img in group])
        #print rois_and_cordin_in_whole_img
        average = sum(np_rois_shapes) / len(np_rois_shapes)
        varience = sum((average - roi_shape) ** 2 for roi_shape in np_rois_shapes) / len(np_rois_shapes)
        return float(varience[0]*varience[1])/(average[0]*average[1])




