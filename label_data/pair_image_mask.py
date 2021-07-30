import os
import time
import cv2
import glob

maskPathList = glob.glob("./masks_zc/*", recursive=True)
imagePathList = glob.glob("./filter_3/*", recursive=True)

mask_name_list = []
image_name_list = []

for maskname in maskPathList:
    mask = maskname.split('\\')[-1].split(".")[0]
    # print(mask)
    mask_name_list.append(mask)

print(len(mask_name_list))
# for imagename in imagePathList:
#     im = cv2.imread(imagename)
#     if(im.shape[2] == 1):
#         print(imagename)
#     image = imagename.split("\\")[-1].split(".")[0]
#     image_name_list.append(image)

count = 1
for imagepath in imagePathList:
    imagename = imagepath.split("\\")[-1].split(".")[0]
    if(imagename in mask_name_list):
        im = cv2.imread(imagepath)
        cv2.imwrite("./images_zc/{}.jpg".format(imagename), im)
        print("完成了{}".format(count))
        count += 1
