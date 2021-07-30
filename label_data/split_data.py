import cv2
import glob
import os

imagePathList = glob.glob('./Picture/*', recursive=True)

count = 1
file = 1
for path in imagePathList:
    if(count % 1000 == 0):
        file += 1
        print("round success")

    image = cv2.imread(path)
    print(image.shape)
    if(image.shape[0] == 768 and image.shape[1] == 1024):
        print(path)
        imagename = path.split("\\")[-1]
        save_path = "./filter_{}".format(file)

        if(not os.path.exists(save_path)):
            os.mkdir(save_path)

        cv2.imwrite(save_path+"/{}".format(imagename), image)
        count += 1

