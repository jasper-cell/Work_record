import cv2
import glob
import numpy as np
import time
import imutils

print(cv2.__version__)

imagePathList = glob.glob("./DaTi/append/images/*", recursive=True)
maskPathList = glob.glob("./DaTi/append/masks/*", recursive=True)
# h,v,hv
print(len(imagePathList))
# exit()
for (imageName, maskName) in zip(imagePathList, maskPathList):
    print(imageName)
    name_of_image = imageName.split("\\")[-1].split(".")[0]
    name_of_mask = maskName.split("\\")[-1].split(".")[0]

    # 进行原图的保存
    img = cv2.imread(imageName)
    mask = cv2.imread(maskName, cv2.IMREAD_GRAYSCALE)

    # 对图像和mask进行水平翻转
    flip_image_h = cv2.flip(img, 1)
    flip_mask_h = cv2.flip(mask, 1)

    # 对图像和mask进行垂直翻转
    flip_image_v = cv2.flip(img, 0)
    flip_mask_v = cv2.flip(mask, 0)

    # 对图像和mask进行水平垂直翻转
    flip_image_hv = cv2.flip(img, -1)
    flip_mask_hv = cv2.flip(mask, -1)

    rotate_image_45 = imutils.rotate(img, 45)
    rotate_mask_45 = imutils.rotate(mask, 45)

    rotate_image_90 = imutils.rotate(img, 90)
    rotate_mask_90 = imutils.rotate(mask, 90)

    rotate_image_135 = imutils.rotate(img, 135)
    rotate_mask_135 = imutils.rotate(mask, 135)

    rotate_image_225 = imutils.rotate(img, 225)
    rotate_mask_225 = imutils.rotate(mask, 225)

    rotate_image_270 = imutils.rotate(img, 270)
    rotate_mask_270 = imutils.rotate(mask, 270)

    rotate_image_315 = imutils.rotate(img, 315)
    rotate_mask_315 = imutils.rotate(mask, 315)

    cv2.imwrite("./DaTi/append/agumentations/images/{}.jpg".format(name_of_image), img)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_h.jpg".format(name_of_image), flip_image_h)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_v.jpg".format(name_of_image), flip_image_v)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_hv.jpg".format(name_of_image), flip_image_hv)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_r45.jpg".format(name_of_image), rotate_image_45)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_r90.jpg".format(name_of_image), rotate_image_90)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_r135.jpg".format(name_of_image), rotate_image_135)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_r225.jpg".format(name_of_image), rotate_image_225)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_r270.jpg".format(name_of_image), rotate_image_270)
    cv2.imwrite("./DaTi/append/agumentations/images/{}_r315.jpg".format(name_of_image), rotate_image_315)

    cv2.imwrite("./DaTi/append/agumentations/masks/{}.jpg".format(name_of_mask), mask)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_h.jpg".format(name_of_mask), flip_mask_h)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_v.jpg".format(name_of_mask), flip_mask_v)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_hv.jpg".format(name_of_mask), flip_mask_hv)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_r45.jpg".format(name_of_mask), rotate_mask_45)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_r90.jpg".format(name_of_mask), rotate_mask_90)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_r135.jpg".format(name_of_mask), rotate_mask_135)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_r225.jpg".format(name_of_mask), rotate_mask_225)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_r270.jpg".format(name_of_mask), rotate_mask_270)
    cv2.imwrite("./DaTi/append/agumentations/masks/{}_r315.jpg".format(name_of_mask), rotate_mask_315)
