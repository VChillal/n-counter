import cv2 as cv
import numpy as np

filePath='C:/Users/magne/Downloads/tardigradeDNA.jpg'
image = cv.imread(filePath)

medianFilteredImage = cv.medianBlur(image,7)

subtractedImage = cv.subtract(image,medianFilteredImage)
grayImage = cv.cvtColor(subtractedImage,cv.COLOR_BGR2GRAY)

ret,thresholdImage = cv.threshold(grayImage,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
kernel = np.ones((1,1),np.uint8)

# openedImage = cv.morphologyEx(thresholdImage,cv.MORPH_OPEN,kernel)
distanceTransformedImage = cv.distanceTransform(thresholdImage,cv.DIST_L2,3)
distanceTransformedImage = np.uint8(distanceTransformedImage)
#
# #####
#
# ret,sure_fg = cv.threshold(distanceTransformedImage,0.05*distanceTransformedImage.max(),255,0)
# sure_bg = cv.dilate(thresholdImage,kernel,iterations=3)
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)
#
# ret, markers = cv.connectedComponents(thresholdImage)
# print(ret)
# markers = markers + 1
# markers[unknown==255] = 0
# markers = cv.watershed(image,markers)
# markedImage = image
# markedImage[markers == -1] = [0,255,0]
#
# cv.imshow("MA",markedImage)
# cv.imshow("SURE_FG",sure_fg)
# cv.imshow("SURE_BG",sure_bg)
cv.imshow("I",image)
cv.imshow("M",medianFilteredImage)
cv.imshow("S",subtractedImage)
cv.imshow("T",thresholdImage)

cv.waitKey(100000)

numNeurons,markers = cv.connectedComponents(thresholdImage)

print(numNeurons)

