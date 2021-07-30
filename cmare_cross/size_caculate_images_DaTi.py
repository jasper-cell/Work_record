# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
import imutils
import cv2
from predict import *
import math
from PIL import Image, ImageDraw, ImageFont
import time
import glob
from multiprocessing import Pool


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

    effective_points = np.array(effective_points)
    print("effective_points: ", effective_points.shape)

    res_distance = find_clearest(effective_points)  # 计算各个点之间的距离

    min_distance = np.min(res_distance)  # 找出距离最短的两个点

    pixelMetric = standard_width / math.sqrt(min_distance)  # 计算对应的比例尺
    return pixelMetric


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


# 对直尺的轮廓的良好直线段进行相应的处理操作
def process_zc_contour(lines, param="lies"):
    """ 计算尺子宽度的对边
        :param lines: 直尺的轮廓线
        :param param: 直尺的摆放形式，比如： 垂直还是水平
        :return: 返回对应的边， 以及对边之间的垂直距离
    """

    # 直尺是水平摆放的时候
    if param == "lies":
        lines = np.squeeze(lines)  # 挤压轴
        print("lines squeeze:\n ", lines)
        arg_max_x = np.where(lines == np.max(lines[:, 0]))  # 找到等于最大值所在的点位
        print("arg max x:\n", arg_max_x)
        lines = np.delete(lines, np.unique(arg_max_x[0]).tolist(), axis=0)  # 删除最大x值所在的那一列, 使用unique的方式防止重复删除多余的行
        print("delete max x lines:\n", lines)
        arg_min_x = np.where(lines == np.min(lines[:, 0]))
        lines = np.delete(lines, np.unique(arg_min_x[0]).tolist(), axis=0)
        print("delete min x lines:\n", lines)
        arg_min_y = np.where(lines == np.min(lines[:, 1]))
        item_1 = lines[arg_min_y[0][0], :]

        arg_max_y = np.where(lines == np.max(lines[:, 1]))
        item_2 = lines[arg_max_y[0][0], :]

        distance = getDist_P2L((item_1[0], item_1[1]), (item_2[0], item_2[1]), (item_2[2], item_2[3]))
        res = np.vstack((item_1, item_2))

        return res, abs(distance)

    # 尺子是垂直摆放的时候
    elif param == "stand":
        lines = np.squeeze(lines)  # 挤压轴

        arg_max_y = np.where(lines == np.max(lines[:, 1]))  # 找到等于最大值所在的点位
        lines = np.delete(lines, np.unique(arg_max_y[0]).tolist(), axis=0)  # 删除最大x值所在的那一列

        arg_min_y = np.where(lines == np.min(lines[:, 1]))
        lines = np.delete(lines, np.unique(arg_min_y[0]).tolist(), axis=0)

        arg_min_x = np.where(lines == np.min(lines[:, 0]))
        item_1 = lines[arg_min_x[0][0], :]

        arg_max_x = np.where(lines == np.max(lines[:, 0]))
        item_2 = lines[arg_max_x[0][0], :]

        print("item1 shape:", item_1.shape)

        distance = getDist_P2L((item_1[0], item_1[1]), (item_2[0], item_2[1]), (item_2[2], item_2[3]))

        res = np.vstack((item_1, item_2))
        return res, abs(distance)


# 计算两点之间的距离
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# 计算直尺的宽度比例,主要用于比例尺的计算工作
def process_contours_zc(edges, cnts_zc, pixelsPerMetric, image):
    print("cnts_zc: ", cnts_zc)
    if cnts_zc:
        for c in cnts_zc:
            # 出于鲁棒性的考量， 直尺的周长应该更长
            if abs(cv2.arcLength(c, closed=True)) < 500:
                print("arcLength: ", cv2.arcLength(c, closed=True))
                continue

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
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=20)  # 对直尺轮廓中的直线进行提取
            print("lines: ", lines)
            if lines is None:
                return color_capture(image)

            elif lines.shape[0] < 2:
                return color_capture(image)  # 若无法找到足够的直线段则直接使用颜色捕捉进行处理

            elif top_dist >= left_dist:  # 直尺横着的
                res, distance = process_zc_contour(lines, param="lies")
                for line in res:
                    x1, y1, x2, y2 = line
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imshow("line_detect_possible_demo", image)
                    cv2.waitKey(0)

            elif top_dist < left_dist:  # 直尺是竖着的
                res, distance = process_zc_contour(lines, param="stand")
                for line in res:
                    x1, y1, x2, y2 = line
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.imshow("line_detect_possible_demo", image)
                    cv2.waitKey(0)

            # 获取相应参考物的比例尺
            if pixelsPerMetric is None:
                pixelsPerMetricF = distance / 2.7  # 计算对应的比例尺
                print("根据直尺首次计算出的比例尺: ", pixelsPerMetricF)
                return pixelsPerMetricF
    else:
        return None


# 计算样本的轮廓
def process_contours(cnts, image, pixelsPerMetric, mask):
    # 对每一个轮廓进行单独的处理
    # print("cnts: ", cnts)
    for c in cnts:
        # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
        if abs(cv2.arcLength(c, closed=True)) < 100:
            continue

        # 计算与轮廓相切的最小的外切矩形
        orig = image.copy()
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
            sample_dist = dist.euclidean(sample_A, sample_B)
            print("dA: {:.2f}, dB: {:.2f}".format(dA, dB))
            pixelsPerMetric = sample_dist / width
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

        # 绘制对应外接矩形的长和宽在相应的图像上

        cv2.putText(orig, "height: {:.2f}cm".format(dimA),
                    (int(30), int(70)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 0, 0), 2)

        # chImage = cv2ImgAddText(orig, "高: {:.2f}cm".format(dimA), 30, 70, (255, 0, 0), 30)

        cv2.putText(orig, "width: {:.2f}cm".format(dimB),
                    (int(30), int(100)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 0, 0), 2)

        # chImage = cv2ImgAddText(chImage, "宽: {:.2f}cm".format(dimB), 30, 100, (255, 0, 0), 30)

        if mask is None:
            # 计算外切矩形正中心位置处颜色的值
            (middleX, middleY) = midpoint([tlblX, tlblY], [trbrX, trbrY])
            color = orig[int(middleY), int(middleX)]
            color = color.tolist()  # 将numpy类型的数组转换为对应的python原生的list形式
        else:
            # 提取对应轮廓的颜色均值
            mean_color = cv2.mean(orig, mask)

        # 在图像上绘制对应的关键文本信息

        cv2.putText(orig, "arcLength:{:.1f}cm".format(res_trans), (int(30), int(130)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (122, 255, 255), 2)
        # chImage = cv2ImgAddText(chImage, "弧长: {:.1f}cm".format(res_trans), 30, 130, (122, 255, 255), 30)

        cv2.putText(orig, "area:{:.2f}cm2".format(area_trans), (int(30), int(160)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (122, 255, 255), 2)
        # chImage = cv2ImgAddText(chImage, "面积: {:.1f}cm2".format(area_trans), 30, 160, (122, 255, 255), 30)

        cv2.rectangle(orig, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)

        # cv2.rectangle(chImage, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)

        cv2.putText(orig, "color: B{}, G{}, R{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
                    (int(30), int(190)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (122, 255, 255), 2)
        # chImage = cv2ImgAddText(chImage, "颜色: 蓝{}, 绿{}, 红{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])), 30, 190, (122, 255, 255), 30)

        # 展示对应的图像
        cv2.imshow("Image", orig)
        cv2.waitKey(0)


if __name__ == '__main__':
    start_time = time.time()
    # pool = Pool(processes=4)
    # 定义对应的命令行参数
    args = get_args()
    width = args.width

    # 作为备用点计算，防止没有放置直尺。或者直尺没有捕捉到
    sample_A = (138, 681)
    sample_B = (179, 682)

    # 使用predict进行mask的提取
    net = UNet(n_channels=3, n_classes=1)  # 加载样本提取模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置处理设备
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))  # 加载模型

    net_zc = UNet(n_channels=3, n_classes=1)  # 加载直尺提取模型
    net_zc.to(device=device)
    net_zc.load_state_dict(torch.load(args.model_zc, map_location=device))  # 加载模型

    input_files = glob.glob("./DaTi_images/*.jpg", recursive=True)

    for i, fn in enumerate(input_files):
        start = time.time()  # 开始的时间
        img = cv2.imread(fn)  # 读取文件夹中的图像
        img = cv2.resize(img, (1024, 768))
        image = img.copy()
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

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

        result_zc = mask_to_image(mask_zc)
        mask_zc = np.asarray(result_zc)

        backup_zc = np.zeros((mask_zc.shape[0] + 10, mask_zc.shape[1] + 10), dtype=np.uint8)  # backup_zc用于填充边缘的部分
        backup_zc[5:mask_zc.shape[0] + 5, 5: mask_zc.shape[1] + 5] = mask_zc[:, :]  # 用填充过的图等于对应的mask_zc中的信息

        edged_zc = cv2.Canny(backup_zc, 50, 150, apertureSize=3)  # 直尺边缘

        # 提取直尺对应的轮廓
        cnts_zc = cv2.findContours(edged_zc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_zc = cnts_zc[0] if imutils.is_cv2() else cnts_zc[1]

        pixelsPerMetric = process_contours_zc(edged_zc, cnts_zc, pixelsPerMetric, image)
        process_contours(cnts, image, pixelsPerMetric, mask=mask)  # 对样本的轮廓进行相应的处理操作
        end = time.time()  # 开始的时间
        print("time consume: ", end - start)

    cv2.destroyAllWindows()
    # pool.close()
    print("multi time: {}".format(time.time() - start_time))
