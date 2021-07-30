import os
import glob
import time
import cv2

# 根据图像选择 save 或者 close，然后自动切换下一张
def getMaskJson(cmd, imagePathList):
    count = 1
    for path in imagePathList:
        print(path)
        print("处理了: {}张".format(count))
        name = path.split("\\")[-1].split(".")[0]
        save_path = "./jsons_zc"
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)
        res = os.popen(cmd + path + " -O " + "./jsons_zc/" + name +".json")
        res.close()
        count += 1


# 将json转为mask
def json_to_mask(cmd, imagePathList):
    for path in imagePathList:
        name = path.split("\\")[-1].split(".")[0]
        print("name: ", name)
        save_path = "./name_json_zc"
        if (not os.path.exists(save_path)):
            os.mkdir(save_path)
        res = os.popen(cmd + path + " -o " + "./name_json_zc/{}".format(name))


def standard_mask(mask_path):
    for mask in mask_path:
        image = cv2.imread(mask)
        maskname = mask.split('\\')[1]
        print(maskname)
        save_path = "./masks_zc"
        if (not os.path.exists(save_path)):
            os.mkdir(save_path)
        cv2.imwrite('./masks_zc/{}.png'.format(maskname), image)


if __name__ == '__main__':
    cmd1 = "labelme "
    # 需要标注图像的路径
    imagePathList = glob.glob("./filter_3/*", recursive=True)

    # 标注原始图像产生对应的json文件
    getMaskJson(cmd=cmd1, imagePathList=imagePathList)

    # 标注完成后给出一些存储json文件的缓冲操作
    time.sleep(100)

    cmd2 = "labelme_json_to_dataset "
    # 需要转换为mask的json文件路径
    jsonPathList = glob.glob("./jsons_zc/*", recursive=True)

    # 将json文件自动转换为mask图像
    json_to_mask(cmd=cmd2, imagePathList=jsonPathList)

    # 标准化mask图像
    mask_path = glob.glob("./name_json_zc/*/label.png", recursive=True)
    standard_mask(mask_path=mask_path)




