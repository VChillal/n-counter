import cv2 as cv
import numpy as np

image = cv.imread("C:/Users/magne/Downloads/coins.jpg")
grayImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
ret,thresholdImage = cv.threshold(grayImage,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresholdImage,cv.MORPH_OPEN,kernel,iterations=2)

sure_bg = cv.dilate(opening,kernel,iterations=3)
distanceTransform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(distanceTransform,distanceTransform.max()*0.7,255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret,markers = cv.connectedComponents(sure_fg)
print(ret)

