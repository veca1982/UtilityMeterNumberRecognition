__author__ = 'Krtalici'

# import the necessary packages
import numpy as np
import cv2
import math
import imutils
import mahotas

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def deskew(img):

	(h,w) = img.shape[:2]
	m = cv2.moments(img)

	# You can read about Image Moments
	# Chapter 18 Shape analysis - Page 633
	# Digital Image Processing 4th Edition By William K. Pratt
	# OR http://en.wikipedia.org/wiki/Image_moment

	x = m['m10']/m['m00']
	y = m['m01']/m['m00']
	mu02 = m['mu02']
	mu20 = m['mu20']
	mu11 = m['mu11']

	lambda1 = 0.5*( mu20 + mu02 ) + 0.5*( mu20**2 + mu02**2 - 2*mu20*mu02 + 4*mu11**2 )**0.5
	lambda2 = 0.5*( mu20 + mu02 ) - 0.5*( mu20**2 + mu02**2 - 2*mu20*mu02 + 4*mu11**2 )**0.5
	lambda_m = max(lambda1, lambda2)

	# Convert from radians to degrees
	angle =  math.ceil(math.atan((lambda_m - mu20)/mu11)*18000/math.pi)/100

	# Create a rotation matrix and use it to de-skew the image.
	center = tuple(map(int, (x, y)))
	rotmat = cv2.getRotationMatrix2D(center, angle , 1)
	rotatedImg = cv2.warpAffine(img, rotmat, (w, h), flags = cv2.INTER_CUBIC)
	return rotatedImg

def deskew2(image):
	(h, w) = image.shape[:2]
	# print image.shape
	moments = cv2.moments(image)

	# deskew the image by applying an affine transformation
	if moments["mu02"] != 0:
		skew = moments["mu11"] / moments["mu02"]
		M = np.float32([
    	[1, skew, -0.5 * w * skew],
    	[0, 1, 0]])
		image = cv2.warpAffine(image, M, (w, h),flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

	return image

def center_extent(image, size):
	# grab the extent width and height
	(eW, eH) = size

	# handle when the width is greater than the height
	if image.shape[1] > image.shape[0]:
		image = imutils.resize(image, width = eW)

	# otherwise, the height is greater than the width
	else:
		image = imutils.resize(image, height = eH)

	# allocate memory for the extent of the image and
	# grab it
	extent = np.zeros((eH, eW), dtype = "uint8")
	offsetX = (eW - image.shape[1]) / 2
	offsetY = (eH - image.shape[0]) / 2
	extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

	# compute the center of mass of the image and then
	# move the center of mass to the center of the image
	(cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
	(dX, dY) = ((size[0] / 2) - cX, (size[1] / 2) - cY)
	M = np.float32([[1, 0, dX], [0, 1, dY]])
	extent = cv2.warpAffine(extent, M, size)

	# return the extent of the image
	return extent

def skeleton(img):


	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)

	ret,img = cv2.threshold(img,127,255,0)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False

	while( not done):
		eroded = cv2.erode(img,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img,temp)
		skel = cv2.bitwise_or(skel,temp)
		img = eroded.copy()

		zeros = size - cv2.countNonZero(img)
		if zeros==size:
			done = True

	return skel


# image = cv2.imread('../trainingData/train6.jpg')
# cv2.imshow("Six orig", image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = deskew(image)
# cv2.imshow("deskew 6",image)
# cv2.waitKey()
if __name__ == "__main__":
	print angle([0,1],[1,0])