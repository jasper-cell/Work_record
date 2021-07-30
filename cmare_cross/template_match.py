# import the necessary packages
# import argparse
# import cv2
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input image where we'll apply template matching")
# ap.add_argument("-t", "--template", type=str, required=True,
# 	help="path to template image")
# args = vars(ap.parse_args())
#
# # load the input image and template image from disk, then display
# # them on our screen
# print("[INFO] loading images...")
# image = cv2.imread(args["image"])
# template = cv2.imread(args["template"])
# cv2.imshow("Image", image)
# cv2.imshow("Template", template)
# # convert both the image and template to grayscale
# imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
#
# # perform template matching
# print("[INFO] performing template matching...")
# result = cv2.matchTemplate(imageGray, templateGray,
# 	cv2.TM_CCOEFF_NORMED)
# (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
#
#
# # determine the starting and ending (x, y)-coordinates of the
# # bounding box
# (startX, startY) = maxLoc
# endX = startX + template.shape[1]
# endY = startY + template.shape[0]
#
# # draw the bounding box on the image
# cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)


import cv2
import numpy as np
import imutils

# img = cv2.imread("C:/AE_PUFF/python_vision/2018_04_27/kk-3.jpg")
img = cv2.imread("test_images/B202104171_001.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(blur, 10, 100)

cnts_zc = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_zc = cnts_zc[0] if imutils.is_cv2() else cnts_zc[1]
for cnt in cnts_zc:
    if cv2.arcLength(cnt, closed=False) > 100:
        cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)

cv2.imshow("edged", edged)
cv2.imshow("res", img)
cv2.waitKey(0)