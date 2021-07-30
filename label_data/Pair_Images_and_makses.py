import cv2
import numpy as np
import glob

imagepath_list = glob.glob("./DaTi_mask_0714_03/*/img.png", recursive=True)
print(imagepath_list)
maskpath_list = glob.glob("./DaTi_mask_0714_03/*/label.png", recursive=True)
print(maskpath_list)

for (imagePath, maskPath) in zip(imagepath_list, maskpath_list):
    image_name = imagePath.split("\\")[-2]
    mask_name = maskPath.split("\\")[-2]
    # print("image name: ", image_name)
    # print("mask name: ", mask_name)
    # exit()
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (1024, 768))
    mask = cv2.imread(maskPath,  cv2.IMREAD_GRAYSCALE)
    res, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (1024, 768))
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
    cv2.imwrite("./DaTi/append/images/{}.png".format(image_name), image)
    cv2.imwrite("./DaTi/append/masks/{}.png".format(mask_name), mask)
