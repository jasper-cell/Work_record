# 用于展示真正用于训练的样本和对应的mask
import cv2
import glob
import numpy as np

imagePathList = glob.glob("./enhance_image/*", recursive=True)
maskPathList = glob.glob("./enhance_mask/*", recursive=True)

print(len(imagePathList))
print(len(maskPathList))

for (imagePath, maskPath) in zip(imagePathList, maskPathList):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (512,512))
    mask = cv2.imread(maskPath)
    mask = cv2.resize(mask, (512, 512))

    merge = np.hstack([image, mask])
    cv2.imshow("merge", merge)
    cv2.waitKey(0)