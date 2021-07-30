from scipy.spatial import distance as dist
from imutils import perspective
import imutils
import cv2
from predict import *
import math
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import json


# 找到list中各个点之间的距离
def find_clearest(points):
    distance_list = []
    for (count, point_outer) in enumerate(points):
        for point_inner in points[count + 1:]:
            distance = (point_inner[0] - point_outer[0]) ** 2 + (point_inner[1] - point_outer[1]) ** 2
            distance_list.append(distance)
    return distance_list


# 对相应的颜色进行捕捉
def color_capture(image):
    # 定义对应的标准长度
    standard_width = 2.0

    # 定义颜色的上下界限
    greenLower = (78, 43, 46)
    greenUpper = (99, 255, 255)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换为hsv空间

    mask = cv2.inRange(hsv, greenLower, greenUpper)  # 根据颜色上下界获取对应的mask

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    print("cnts")
    if not cnts[1] or len(cnts) < 2:
        return None

    cnts = imutils.grab_contours(cnts)
    effective_points = []

    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 7 and radius < 10:
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(image, center, 5, (0, 0, 255), -1)
            effective_points.append(center)
    try:
        effective_points = np.array(effective_points)

        res_distance = find_clearest(effective_points)  # 计算各个点之间的距离

        min_distance = np.min(res_distance)  # 找出距离最短的两个点

        pixelMetric = standard_width / math.sqrt(min_distance)  # 计算对应的比例尺
        return pixelMetric
    except:
        return None


# 在图像中显示中文
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ***** 点到直线的距离:P到AB的距离*****
# P为线外一点，AB为线段两个端点
def getDist_P2L(PointP, Pointa, Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    # 求直线方程
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0] * Pointb[1] - Pointa[1] * Pointb[0]
    # 代入点到直线距离公式
    distance = (A * PointP[0] + B * PointP[1] + C) / math.sqrt(A * A + B * B)

    return distance


'''
对直尺的轮廓的良好直线段进行相应的处理操作
'''


def process_zc_contour(lines, param="lies"):
    """ 计算尺子宽度的对边
        :param lines: 直尺的轮廓线
        :param param: 直尺的摆放形式，比如： 垂直还是水平
        :return: 返回对应的边， 以及对边之间的垂直距离
    """

    # 直尺是水平摆放的时候
    if param == "lies":
        lines = np.squeeze(lines)  # 挤压轴
        print("lines original:", lines)
        try:
            arg_max_x = np.where(lines == np.max(lines[:, 0]))  # 找到等于最大值所在的点位
            print("最大X的值: ", arg_max_x)

            lines = np.delete(lines, np.unique(arg_max_x[0]).tolist(), axis=0)  # 删除最大x值所在的那一列, 使用unique的方式防止重复删除多余的行
            print("lines delete x max: ", lines)

            arg_min_x = np.where(lines == np.min(lines[:, 0]))
            print("最小X的值: ", arg_min_x)
            lines = np.delete(lines, np.unique(arg_min_x[0]).tolist(), axis=0)
            print("lines_delete_min_x: ", lines)

            arg_min_y = np.where(lines == np.min(lines[:, 1]))
            print("最小y值: ", arg_min_y)
            item_1 = lines[arg_min_y[0][0], :]
            print("item_1: ", item_1)

            arg_max_y = np.where(lines == np.max(lines[:, 1]))
            print("最大y值: ", arg_max_y)
            item_2 = lines[arg_max_y[0][0], :]
            print("item_2: ", item_2)

            distance = getDist_P2L((item_1[0], item_1[1]), (item_2[0], item_2[1]), (item_2[2], item_2[3]))
            print("distance: ", abs(distance))
            res = np.vstack((item_1, item_2))

            return res, abs(distance)
        except:
            return None, None

    # 尺子是垂直摆放那个的时候
    elif param == "stand":
        lines = np.squeeze(lines)  # 挤压轴
        print("lines original:", lines)

        arg_max_y = np.where(lines == np.max(lines[:, 1]))  # 找到等于最大值所在的点位
        print("最大y的值: ", arg_max_y)
        lines = np.delete(lines, np.unique(arg_max_y[0]).tolist(), axis=0)  # 删除最大x值所在的那一列
        print("lines delete y max: ", lines)

        arg_min_y = np.where(lines == np.min(lines[:, 1]))
        print("最小y的值: ", arg_min_y)
        lines = np.delete(lines, np.unique(arg_min_y[0]).tolist(), axis=0)
        print("lines delete y min: ", lines)

        arg_min_x = np.where(lines == np.min(lines[:, 0]))
        print("最小x值: ", arg_min_x)
        item_1 = lines[arg_min_x[0][0], :]
        print("item_1: ", item_1)

        arg_max_x = np.where(lines == np.max(lines[:, 0]))
        print("最大x值: ", arg_max_x)
        item_2 = lines[arg_max_x[0][0], :]
        print("item_2: ", item_2)

        distance = getDist_P2L((item_1[0], item_1[1]), (item_2[0], item_2[1]), (item_2[2], item_2[3]))
        print("distance: ", abs(distance))

        res = np.vstack((item_1, item_2))
        print("res: ", res)
        return res, abs(distance)


'''
计算两点之间的距离
'''


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


'''
计算直尺的宽度比例,主要用于比例尺的计算工作
'''


def process_contours_zc(edges, cnts_zc, pixelsPerMetric, image):
    if cnts_zc:  # 有轮廓才进行处理
        for c in cnts_zc:
            # 出于鲁棒性的考量， 直尺的周长应该更长
            if abs(cv2.arcLength(c, closed=False)) < 50:
                continue
                print("arcLength: ", cv2.arcLength(c, closed=True))

            # 计算与轮廓相切的最小的外切矩形
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # 对轮廓中的点进行左上，右上， 右下， 左下的排序操作
            box = perspective.order_points(box)

            # 计算顶边和底边的两个中心点
            (tl, tr, br, bl) = box

            top_dist = dist.euclidean(tl, tr)  # 顶边的长度
            left_dist = dist.euclidean(tl, bl)  # 侧边的长度
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)  # 对直尺轮廓中的直线进行提取

            if lines is None:
                return color_capture(image)

            elif lines.shape[0] < 2:
                return color_capture(image)  # 若无法找到足够的直线段则直接使用颜色捕捉进行处理

            elif top_dist >= left_dist:  # 直尺横着的
                res, distance = process_zc_contour(lines, param="lies")
                if res == None:
                    return None

            else:  # 直尺是竖着的
                res, distance = process_zc_contour(lines, param="stand")
                if res == None:
                    return None

            # 获取相应参考物的比例尺
            if pixelsPerMetric is None:
                pixelsPerMetricF = distance / 2.7  # 计算对应的比例尺
                print("根据直尺首次计算出的比例尺: ", pixelsPerMetricF)
                return pixelsPerMetricF

    else:  # 否者直接返回None，pixelsPerMetricF为None
        return None


'''
参考计算出的比例尺对样本的轮廓进行处理
'''


def process_contours(cnts, image, pixelsPerMetric, mask):
    # 计算与轮廓相切的最小的外切矩形
    orig = image.copy()

    '''
    设置sample_number字典用于记录图像中对应的各个样本的基本信息
    '''
    sample_number = {}
    sample_count = 1
    # 对每一个轮廓进行单独的处理
    if cnts:
        for c in cnts:
            '''在对应的样本中每一个样本作为key使用，其基本信息写入一个list中作为value使用'''
            sample_number[sample_count] = []
            # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
            if abs(cv2.arcLength(c, closed=True)) < 100:
                continue

            box = cv2.minAreaRect(c)  # 计算最小外接矩形
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # 对捕捉到的外接矩形的点进行排序处理，并绘制其对应的轮廓
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            # 绘制最小外接矩形对应的边界框的点位
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # 计算顶边和底边的两个中心点
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # 计算左边和右边的两个中心点
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # 绘制每一条边的中心点
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # 绘制中点之间的线段
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)

            # 对应中点的连线来作为对应的长和宽的值来使用
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # 获取相应参考物的比例尺
            if pixelsPerMetric is None:
                print("pixelsPerMetric is None: ", pixelsPerMetric)
                pixelsPerMetric = dB / 2.3  # 计算对应的比例尺
                print(pixelsPerMetric)

            # 计算物体的实际尺寸
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            res = cv2.arcLength(c, True)  # 计算对应的轮廓的弧长
            res_trans = res / pixelsPerMetric  # 转换为实际的尺寸

            ret = cv2.drawContours(orig, [c], -1, (50, 0, 212), 5)
            area = cv2.contourArea(c)  # 计算对应的轮廓的面积
            area_trans = area / (pixelsPerMetric * pixelsPerMetric)

            cv2.putText(orig, "number:{}".format(sample_count),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 0, 0), 2)
            # 绘制对应外接矩形的长和宽在相应的图像上
            sub_dict = {}
            sub_dict['height'] = dimA
            sub_dict['width'] = dimB

            if mask is None:
                # 计算外切矩形正中心位置处颜色的值
                (middleX, middleY) = midpoint([tlblX, tlblY], [trbrX, trbrY])
                color = orig[int(middleY), int(middleX)]
                color = color.tolist()  # 将numpy类型的数组转换为对应的python原生的list形式
            else:
                # 提取对应轮廓的颜色均值
                mean_color = cv2.mean(orig, mask)

            sub_dict['arclength'] = res_trans
            sub_dict['area'] = area_trans
            sub_dict['color'] = [int(mean_color[0]), int(mean_color[1]), int(mean_color[2])]

            sample_number[sample_count] = sub_dict
            sample_count += 1
        return orig, sample_number
    # 作为后备策略使用
    else:
        backup_img = np.zeros(orig.shape, dtype=np.uint8)
        sub_dict = {}
        sub_dict['height'] = 0
        sub_dict['width'] = 0
        sub_dict['arclength'] = 0
        sub_dict['area'] = 0
        sub_dict['color'] = [0, 0, 0]
        sample_number[sample_count] = sub_dict
        return backup_img, sample_number


def process_contours_labelme(cnts, image):
    img_path = "./tcp_files/test.png"
    json_path = "./tcp_files/test.json"

    cv2.imwrite(img_path, image)

    labelme_json = {"version": "4.5.7", "flags": {}, "shapes": []}
    print("cnts type: ", type(cnts))
    print("cnts", cnts[0].shape)
    new_cnts = []

    # 对每一个轮廓进行单独的处理
    if cnts:
        for c in cnts:
            '''在对应的样本中每一个样本作为key使用，其基本信息写入一个list中作为value使用'''
            # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
            if abs(cv2.arcLength(c, closed=True)) < 100:
                continue

            number_dict = {"label": "pos"}
            res = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.0008 * res, True)
            c_t = approx.reshape(approx.shape[0], approx.shape[2])
            number_dict["points"] = c_t.tolist()
            number_dict["group_id"] = None
            number_dict["shape_type"] = "polygon"
            number_dict["flags"] = {}

            labelme_json["shapes"].append(number_dict)

        labelme_json["imagePath"] = img_path
        with open(img_path, 'rb') as png_file:
            byte_content = png_file.read()

        labelme_json["imageData"] = base64.b64encode(byte_content).decode("utf-8")

        labelme_json["imageHeight"] = image.shape[1]
        labelme_json["imageWidth"] = image.shape[0]

        json_str = json.dumps(labelme_json, indent=2)

        with open(json_path, 'w') as json_file:
            json_file.write(json_str)

        res = os.popen("labelme " + json_path)
        res.close()

        with open(json_path, 'r') as load_f:
            load_dict = json.load(load_f)

        for i in range(len(load_dict["shapes"])):
            item = load_dict["shapes"][i]["points"]
            item = np.array(item, dtype=int)
            item = item.reshape(item.shape[0], 1, item.shape[1])
            new_cnts.append(item)
        return new_cnts
    else:
        return cnts


def integerate(img):
    args = get_args()  # 对参数进行解析操作
    width = args.width

    img = cv2.resize(img, (1024, 768))
    image = img.copy()
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 使用predict进行mask的提取

    net = UNet(n_channels=3, n_classes=1)  # 加载样本提取模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置处理设备
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))  # 加载模型

    net_zc = UNet(n_channels=3, n_classes=1)  # 加载直尺提取模型
    net_zc.to(device=device)
    net_zc.load_state_dict(torch.load(args.model_zc, map_location=device))  # 加载模型

    # 构建两个网络，同时获得对应的样本轮廓的mask和直尺的mask, 来一副新的图才进行计算
    mask, mask_zc = predict_img(net=net, net_zc=net_zc,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device)

    # 转换为对应的PIL.Image格式
    result = mask_to_image(mask)

    # 转为numpy.array的形式，使得之后的opencv能够进行调用
    mask = np.asarray(result)

    # 使用Canny进行相应的边缘检测
    edged = cv2.Canny(mask, 0, 100)  # 样本图像的边缘

    # 在经过边缘检测的图像中进行轮廓特征的提取
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)  # 找到对应的样本的轮廓
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 不同版本的opencv对应的contours的位置是不一致的

    # 用于存储每个像素点对应真实值的比例
    pixelsPerMetric = None

    # if mask_zc is not None:
    result_zc = mask_to_image(mask_zc)
    mask_zc = np.asarray(result_zc)  # 最原始的直尺mask
    backup_zc = np.zeros((mask_zc.shape[0] + 10, mask_zc.shape[1] + 10), dtype=np.uint8)  # backup_zc用于填充边缘的部分
    backup_zc[5:mask_zc.shape[0] + 5, 5: mask_zc.shape[1] + 5] = mask_zc[:, :]  # 用填充过的图等于对应的mask_zc中的信息

    edged_zc = cv2.Canny(backup_zc, 50, 150, apertureSize=3)  # 直尺边缘

    # 提取直尺对应的轮廓
    cnts_zc = cv2.findContours(edged_zc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_zc = cnts_zc[0] if imutils.is_cv2() else cnts_zc[1]

    pixelsPerMetric = process_contours_zc(edged_zc, cnts_zc, pixelsPerMetric, image)  # 计算比例尺
    new_cnt = process_contours_labelme(cnts, image)  # 对样本的轮廓进行相应的处理操作
    res = process_contours(new_cnt, image, pixelsPerMetric, mask=mask)  # 对样本的轮廓进行相应的处理操作
    return res, mask
