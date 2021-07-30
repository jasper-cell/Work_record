import os
import glob
import cv2


# 将json转为mask
def json_to_mask(cmd, imagePathList):
    count = 0
    for path in imagePathList:
        count += 1
        name = path.split("\\")[-1].split(".")[0]
        print("name: ", name)
        # save_path = "./DaTi_mask"
        # if (not os.path.exists(save_path)):
        #     os.mkdir(save_path)
        res = os.popen(cmd + path + " -o " + "./DaTi_mask_0714_03/{}".format(name))
    print("total process: ", count)


def standard_mask(mask_path):
    for mask in mask_path:
        image = cv2.imread(mask)
        maskname = mask.split('\\')[1]
        print(maskname)
        # save_path = "./masks_zc"
        # if (not os.path.exists(save_path)):
        #     os.mkdir(save_path)
        cv2.imwrite('./DaTi_fin_img/{}.png'.format(maskname), image)


if __name__ == '__main__':
    cmd2 = "labelme_json_to_dataset "
    # 需要转换为mask的json文件路径
    jsonPathList = glob.glob("./2021.7.5-7.9/*.json", recursive=True)

    # 将json文件自动转换为mask图像
    json_to_mask(cmd=cmd2, imagePathList=jsonPathList)

    # # 标准化mask图像
    # mask_path = glob.glob("./DaTi_mask/*/label.png", recursive=True)
    # standard_mask(mask_path=mask_path)
