import glob
import cv2

# imagePathList = glob.glob("./images/*")
# for imagename in imagePathList:
#     name = imagename.split("\\")[-1].split(".")[0]
#     img = cv2.imread(imagename)
#     cv2.imwrite("./image_samples/{}.png".format(name), img)
import numpy as np
maskPathList = glob.glob("./masks_zc/*")
for maskname in maskPathList:
    name = maskname.split("\\")[-1].split(".")[0]
    print(name)
    mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    exit()
    res, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./Graymasks_zc/{}.jpg".format(name), mask)