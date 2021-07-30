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


def find_largest_rectangle(points, arg_max_x, arg_min_x, arg_max_y, arg_min_y):
    contourArea_max = 0
    count = 0
    results = None
    for item_max_x in arg_max_x[0]:
        for item_min_x in arg_min_x[0]:
            for item_max_y in arg_max_y[0]:
                for item_min_y in arg_min_y[0]:
                    right = points[item_max_x]  # 右边的点位
                    left = points[item_min_x]  # 左边的点位
                    bottom = points[item_max_y]  # 底部的点位
                    top = points[item_min_y]  # 顶部的点位
                    contour = np.array([[top, right, bottom, left]])
                    area = cv2.contourArea(contour)
                    # print("============^_^==============")
                    if area > contourArea_max:
                        print("function area: ", area)
                        # print("result: ", results)
                        contourArea_max = area
                        count += 1
                        results = contour

    return results


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

    print("color cnts: ", cnts)
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
        # print("lines original:", lines)

        arg_max_x = np.where(lines == np.max(lines[:, 0]))  # 找到等于最大值所在的点位
        # print("最大X的值: ", arg_max_x)

        lines = np.delete(lines, np.unique(arg_max_x[0]).tolist(), axis=0)  # 删除最大x值所在的那一列, 使用unique的方式防止重复删除多余的行
        # print("lines delete x max: ", lines)

        arg_min_x = np.where(lines == np.min(lines[:, 0]))
        # print("最小X的值: ", arg_min_x)
        lines = np.delete(lines, np.unique(arg_min_x[0]).tolist(), axis=0)
        # print("lines_delete_min_x: ", lines)

        arg_min_y = np.where(lines == np.min(lines[:, 1]))
        # print("最小y值: ", arg_min_y)
        item_1 = lines[arg_min_y[0][0], :]
        # print("item_1: ", item_1)

        arg_max_y = np.where(lines == np.max(lines[:, 1]))
        # print("最大y值: ", arg_max_y)
        item_2 = lines[arg_max_y[0][0], :]
        # print("item_2: ", item_2)

        distance = getDist_P2L((item_1[0], item_1[1]), (item_2[0], item_2[1]), (item_2[2], item_2[3]))
        print("distance: ", abs(distance))
        res = np.vstack((item_1, item_2))

        return res, abs(distance)

    # 尺子是垂直摆放那个的时候
    elif param == "stand":
        lines = np.squeeze(lines)  # 挤压轴
        # print("lines original:", lines)

        arg_max_y = np.where(lines == np.max(lines[:, 1]))  # 找到等于最大值所在的点位
        # print("最大y的值: ", arg_max_y)
        lines = np.delete(lines, np.unique(arg_max_y[0]).tolist(), axis=0)  # 删除最大x值所在的那一列
        # print("lines delete y max: ", lines)

        arg_min_y = np.where(lines == np.min(lines[:, 1]))
        # print("最小y的值: ", arg_min_y)
        lines = np.delete(lines, np.unique(arg_min_y[0]).tolist(), axis=0)
        # print("lines delete y min: ", lines)

        arg_min_x = np.where(lines == np.min(lines[:, 0]))
        # print("最小x值: ", arg_min_x)
        item_1 = lines[arg_min_x[0][0], :]
        # print("item_1: ", item_1)

        arg_max_x = np.where(lines == np.max(lines[:, 0]))
        # print("最大x值: ", arg_max_x)
        item_2 = lines[arg_max_x[0][0], :]
        # print("item_2: ", item_2)

        distance = getDist_P2L((item_1[0], item_1[1]), (item_2[0], item_2[1]), (item_2[2], item_2[3]))
        # print("distance: ", abs(distance))

        res = np.vstack((item_1, item_2))
        # print("res: ", res)
        return res, abs(distance)


# 计算两点之间的距离
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def get_point_to_draw(orig, c, point_top, point_right, point_bottom, point_left, length_pixels, width_pixels, whole_mean_color, whole_cnt, model='close'):
    '''
    设置对应的候选框
    '''
    if model == 'close':
        point_rt = (int(point_right[0]), int(point_top[1]))
        point_rb = (int(point_right[0]), int(point_bottom[1]))
        point_lb = (int(point_left[0]), int(point_bottom[1]))
        point_lt = (int(point_left[0]), int(point_top[1]))
        candidate_points = [point_rt, point_rb, point_lb, point_lt]
    elif model == 'far':
        point_t = (int(point_top[0]), int(point_top[1] - width_pixels))
        point_l = (int(point_left[0] - length_pixels), int(point_left[1]))
        point_b = (int(point_bottom[0]), int(point_bottom[1] + width_pixels))
        point_r = (int(point_right[0] + length_pixels), int(point_right[1]))
        # point_rt = (int(point_right[0] + length_pixels), int(point_top[1] - width_pixels))
        # point_rb = (int(point_right[0] + length_pixels), int(point_bottom[1] + width_pixels))
        # point_lb = (int(point_left[0] - length_pixels), int(point_bottom[1] + width_pixels))
        # point_lt = (int(point_left[0] - length_pixels), int(point_top[1] - width_pixels))
        # candidate_points = [point_t, point_l, point_b, point_r, point_rt, point_rb, point_lb, point_lt]
        candidate_points = [point_t, point_l, point_b, point_r]


    # 找到现有候选点中的在轮廓中的点
    for sub_point in candidate_points:
        # print(sub_point)
        sub_dis = cv2.pointPolygonTest(c, sub_point, True)
        flag = False
        for sub_cnt in whole_cnt:
            # 判断对应的点位是否存在于任何一个大体的轮廓之中
            if cv2.pointPolygonTest(sub_cnt, sub_point, False) == 1:  # 计算点与轮廓之间的相对位置
                flag = True
        if not flag:
            candidate_points.remove(sub_point)
            continue
        real_dis = abs(sub_dis) / pixelsPerMetric  # 真实的距离值到轮廓中
        if model == 'close':
            if real_dis > 1.0:
                candidate_points.remove(sub_point)
        elif model == 'far':
            if real_dis < 1.0:
                candidate_points.remove(sub_point)
        # print("dis: ", real_dis)
    # print("filtered_points: ", candidate_points)
    color_res_list = []

    for sub_point in candidate_points:
        sub_point_x = sub_point[0]
        sub_point_y = sub_point[1]
        sub_point_x1 = sub_point_x - 0.5 * length_pixels
        sub_point_x2 = sub_point_x + 0.5 * length_pixels
        sub_point_y1 = sub_point_y - 0.5 * width_pixels
        sub_point_y2 = sub_point_y + 0.5 * width_pixels
        candidate_area = orig[int(sub_point_y1):int(sub_point_y2), int(sub_point_x1):int(sub_point_x2)]
        sub_color_mean = cv2.mean(candidate_area)
        # print("sub color mean: ", sub_color_mean)
        # 计算与整体轮廓颜色均值的距离
        color_res = dist.euclidean(sub_color_mean, whole_mean_color)
        # print("color res: ", color_res)
        color_res_list.append(color_res)
    color_res_list = np.array(color_res_list)
    min_index = np.argmin(color_res_list)
    point_to_draw = candidate_points[min_index]
    return point_to_draw


# 计算直尺的宽度比例,主要用于比例尺的计算工作
def process_contours_zc(edges, cnts_zc, pixelsPerMetric, image):
    # print("cnts_zc: ", cnts_zc)
    if cnts_zc:
        for c in cnts_zc:
            # 出于鲁棒性的考量， 直尺的周长应该更长
            if abs(cv2.arcLength(c, closed=True)) < 700:
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
                # return color_capture(image)
                return None

            elif lines.shape[0] < 2:
                return None  # 若无法找到足够的直线段则直接使用颜色捕捉进行处理

            elif top_dist >= left_dist:  # 直尺横着的
                # dA = left_dist
                res, distance = process_zc_contour(lines, param="lies")
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                for line in res:
                    print(line)
                    x1, y1, x2, y2 = line
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.imshow("line_detect_possible_demo", image)
                    # cv2.waitKey(0)

            elif top_dist < left_dist:  # 直尺是竖着的
                # dA = top_dist
                res, distance = process_zc_contour(lines, param="stand")
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                for line in res:
                    print(line)
                    x1, y1, x2, y2 = line
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.imshow("line_detect_possible_demo", image)
                    # cv2.waitKey(0)

            # 获取相应参考物的比例尺
            if pixelsPerMetric is None:
                pixelsPerMetricF = distance / 2.7  # 计算对应的比例尺
                print("根据直尺首次计算出的比例尺: ", pixelsPerMetricF)
                return pixelsPerMetricF
    else:
        return None


# 计算样本的轮廓
def process_contours(cnts, image, pixelsPerMetric, mask, whole_mean_color, whole_cnt):
    orig = image.copy()
    # 对每一个轮廓进行单独的处理
    for i, c in enumerate(cnts):
        # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
        if abs(cv2.arcLength(c, closed=True)) < 70:
            continue

        # 计算与轮廓相切的最小的外切矩形
        box = cv2.minAreaRect(c)  # 计算最小外接矩形

        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # 对捕捉到的外接矩形的点进行排序处理，并绘制其对应的轮廓
        box = perspective.order_points(box)
        # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        # 绘制最小外接矩形对应的边界框的点位
        # for (x, y) in box:
        #     cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 计算顶边和底边的两个中心点
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # 计算左边和右边的两个中心点
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # # 绘制每一条边的中心点
        # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        #
        # # 绘制中点之间的线段
        # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        #          (255, 0, 255), 2)
        # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        #          (255, 0, 255), 2)

        poalr_cnts = np.squeeze(c)

        arg_max_x = np.where(poalr_cnts == np.max(poalr_cnts[:, 0]))
        arg_min_x = np.where(poalr_cnts == np.min(poalr_cnts[:, 0]))
        arg_max_y = np.where(poalr_cnts == np.max(poalr_cnts[:, 1]))
        arg_min_y = np.where(poalr_cnts == np.min(poalr_cnts[:, 1]))

        sample_points = find_largest_rectangle(poalr_cnts, arg_max_x, arg_min_x, arg_max_y, arg_min_y)

        sample_points = sample_points.squeeze(0)
        point_center = np.array([int((tlblX + trbrX) / 2), int((tltrY + blbrY) / 2)])
        sample_points = np.append(sample_points, point_center).reshape(5, 2)
        # print(sample_points)

        point_top = sample_points[0]
        point_right = sample_points[1]
        point_bottom = sample_points[2]
        point_left = sample_points[3]
        point_center = sample_points[4]  # x, y

        # print("contains points: \n", )
        # exit()

        # print("arg_max_x:  ", arg_max_x)
        # print("arg_min_x:  ", arg_min_x)
        # print("arg_max_y:  ", arg_max_y)
        # print("arg_min_y:  ", arg_min_y)

        # 找出中心点的坐标与最外面轮廓各个顶点的远近

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

        # 将1cm换算为像素值
        width_pixels = 0.4 * pixelsPerMetric
        # 将2cm换算为像素值
        length_pixels = 0.8 * pixelsPerMetric

        # print("width pixels: ", width_pixels)
        # print("length oixels: ", length_pixels)
        # print(arg_min_x)

        '''
        根据癌灶区的轮廓的直径来对轮廓的标注方法进行选择
        '''
        # 如果轮廓的直径含有小于3的情况则进行四等分的划分
        contour_width = dist.euclidean(point_right, point_left) / pixelsPerMetric
        contour_height = dist.euclidean(point_top, point_bottom) / pixelsPerMetric

        if contour_height < 3 and contour_width < 3:
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
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
        else:
            cv2.rectangle(orig, (point_top[0] - int(length_pixels / 2), point_top[1] - int(width_pixels / 2)),
                          (point_top[0] + int(length_pixels / 2), point_top[1] + int(width_pixels / 2)), (255, 0, 0), 2)

            cv2.rectangle(orig, (point_bottom[0] - int(length_pixels / 2), point_bottom[1] - int(width_pixels / 2)),
                          (point_bottom[0] + int(length_pixels / 2), point_bottom[1] + int(width_pixels / 2)),
                          (255, 0, 0), 2)

            cv2.rectangle(orig, (point_center[0] - int(length_pixels / 2), point_center[1] - int(width_pixels / 2)),
                          (point_center[0] + int(length_pixels / 2), point_center[1] + int(width_pixels / 2)),
                          (255, 0, 0), 2)

            cv2.rectangle(orig, (point_left[0] - int(width_pixels / 2), point_left[1] - int(length_pixels / 2)),
                          (point_left[0] + int(width_pixels / 2), point_left[1] + int(length_pixels / 2)), (255, 0, 0),
                          2)

            cv2.rectangle(orig, (point_right[0] - int(width_pixels / 2), point_right[1] - int(length_pixels / 2)),
                          (point_right[0] + int(width_pixels / 2), point_right[1] + int(length_pixels / 2)),
                          (255, 0, 0),
                          2)
            point_to_draw = get_point_to_draw(orig, c, point_top, point_right, point_bottom, point_left, length_pixels, width_pixels, whole_mean_color, whole_cnt, model='close')
            point_to_draw_far = get_point_to_draw(orig, c, point_top, point_right, point_bottom, point_left, length_pixels, width_pixels, whole_mean_color, whole_cnt, model='far')
            # '''
            # 设置对应的候选框
            # '''
            # # 小于1cm的点位
            # point_rt = (int(point_right[0]), int(point_top[1]))
            # point_rb = (int(point_right[0]), int(point_bottom[1]))
            # point_lb = (int(point_left[0]), int(point_bottom[1]))
            # point_lt = (int(point_left[0]), int(point_top[1]))
            #
            # candidate_points = [point_rt, point_rb, point_lb, point_lt]
            #
            # # 找到现有候选点中的
            # for sub_point in candidate_points:
            #     print(sub_point)
            #     sub_dis = cv2.pointPolygonTest(c, sub_point, True)
            #     real_dis = abs(sub_dis) / pixelsPerMetric
            #     if real_dis > 1.0:
            #         candidate_points.remove(sub_point)
            #     print("dis: ", real_dis)
            # print("filtered_points: ", candidate_points)
            # color_res_list = []
            # for sub_point in candidate_points:
            #     sub_point_x = sub_point[0]
            #     sub_point_y = sub_point[1]
            #     sub_point_x1 = sub_point_x - 0.5 * length_pixels
            #     sub_point_x2 = sub_point_x + 0.5 * length_pixels
            #     sub_point_y1 = sub_point_y - 0.5 * width_pixels
            #     sub_point_y2 = sub_point_y + 0.5 * width_pixels
            #     candidate_area = orig[int(sub_point_y1):int(sub_point_y2), int(sub_point_x1):int(sub_point_x2)]
            #     sub_color_mean = cv2.mean(candidate_area)
            #     print("sub color mean: ", sub_color_mean)
            #     # 计算与整体轮廓颜色均值的距离
            #     color_res = dist.euclidean(sub_color_mean, whole_mean_color)
            #     print("color res: ", color_res)
            #     color_res_list.append(color_res)
            # color_res_list = np.array(color_res_list)
            # min_index = np.argmin(color_res_list)
            # point_to_draw = candidate_points[min_index]

            cv2.rectangle(orig, (point_to_draw[0] - int(length_pixels / 2), point_to_draw[1] - int(width_pixels / 2)),
                          (point_to_draw[0] + int(length_pixels / 2), point_to_draw[1] + int(width_pixels / 2)),
                          (0, 255, 0), 2)

            cv2.rectangle(orig, (point_to_draw_far[0] - int(length_pixels / 2), point_to_draw_far[1] - int(width_pixels / 2)),
                          (point_to_draw_far[0] + int(length_pixels / 2), point_to_draw_far[1] + int(width_pixels / 2)),
                          (0, 0, 255), 2)
            # cv2.rectangle(orig, (
            #     point_top[0] + int(5 + length_pixels) - int(length_pixels / 2),
            #     point_top[1] - 5 - int(width_pixels / 2)),
            #               (point_top[0] + int(5 + length_pixels) + int(length_pixels / 2),
            #                point_top[1] - 5 + int(width_pixels / 2)), (0, 0, 255), 2)

        res = cv2.arcLength(c, True)  # 计算对应的轮廓的弧长
        res_trans = res / pixelsPerMetric  # 转换为实际的尺寸

        ret = cv2.drawContours(orig, [c], -1, (50, 0, 212), 5)
        area = cv2.contourArea(c)  # 计算对应的轮廓的面积
        area_trans = area / (pixelsPerMetric * pixelsPerMetric)

        # 绘制对应外接矩形的长和宽在相应的图像上

        # cv2.putText(orig, "height: {:.2f}cm".format(dimA),
        #             (int(30), int(70)), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.65, (255, 0, 0), 2)

        chImage = cv2ImgAddText(orig, "长: {:.2f}cm".format(dimA), 30, 70, (255, 0, 0), 30)

        # cv2.putText(orig, "width: {:.2f}cm".format(dimB),
        #             (int(30), int(100)), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.65, (255, 0, 0), 2)

        chImage = cv2ImgAddText(chImage, "宽: {:.2f}cm".format(dimB), 30, 100, (255, 0, 0), 30)

        if mask is None:
            # 计算外切矩形正中心位置处颜色的值
            (middleX, middleY) = midpoint([tlblX, tlblY], [trbrX, trbrY])
            color = orig[int(middleY), int(middleX)]
            color = color.tolist()  # 将numpy类型的数组转换为对应的python原生的list形式
        else:
            # 提取对应轮廓的颜色均值
            mean_color = cv2.mean(orig, mask)

        # 在图像上绘制对应的关键文本信息

        # cv2.putText(orig, "arcLength:{:.1f}cm".format(res_trans), (int(30), int(130)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.65, (122, 255, 255), 2)
        chImage = cv2ImgAddText(chImage, "弧长: {:.1f}cm".format(res_trans), 30, 130, (0, 0, 255), 30)

        # cv2.putText(orig, "area:{:.2f}cm2".format(area_trans), (int(30), int(160)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.65, (122, 255, 255), 2)
        chImage = cv2ImgAddText(chImage, "面积: {:.1f}cm2".format(area_trans), 30, 160, (0, 0, 255), 30)

        # print("^^^^^contour area^^^^^^ {}: ".format(i), area_trans)
        print("dist horiontal {}: ".format(i), dist.euclidean((point_right), (point_left)) / pixelsPerMetric)
        print("dist horiontal {}: ".format(i), dist.euclidean((point_top), (point_bottom)) / pixelsPerMetric)

        cv2.rectangle(orig, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)

        cv2.rectangle(chImage, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)

        # cv2.putText(orig, "color: B{}, G{}, R{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
        #             (int(30), int(190)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (122, 255, 255), 2)
        chImage = cv2ImgAddText(chImage,
                                "颜色: 蓝{}, 绿{}, 红{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
                                30, 190, (0, 0, 255), 30)

        # 展示对应的图像
        # cv2.imshow("Image", orig)
        cv2.imshow("ChImage", chImage)
        # cv2.waitKey(0)


def process_contours_DaTi(cnts, image, pixelsPerMetric, mask):
    orig = image.copy()
    # 对每一个轮廓进行单独的处理
    for i, c in enumerate(cnts):
        # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
        if abs(cv2.arcLength(c, closed=True)) < 70:
            continue

        # 计算与轮廓相切的最小的外切矩形
        box = cv2.minAreaRect(c)  # 计算最小外接矩形

        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # 对捕捉到的外接矩形的点进行排序处理，并绘制其对应的轮廓
        box = perspective.order_points(box)

        # 计算顶边和底边的两个中心点
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # 计算左边和右边的两个中心点
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        poalr_cnts = np.squeeze(c)

        arg_max_x = np.where(poalr_cnts == np.max(poalr_cnts[:, 0]))
        arg_min_x = np.where(poalr_cnts == np.min(poalr_cnts[:, 0]))
        arg_max_y = np.where(poalr_cnts == np.max(poalr_cnts[:, 1]))
        arg_min_y = np.where(poalr_cnts == np.min(poalr_cnts[:, 1]))

        sample_points = find_largest_rectangle(poalr_cnts, arg_max_x, arg_min_x, arg_max_y, arg_min_y)

        sample_points = sample_points.squeeze(0)
        point_center = np.array([int((tlblX + trbrX) / 2), int((tltrY + blbrY) / 2)])
        sample_points = np.append(sample_points, point_center).reshape(5, 2)

        '''
        计算出轮廓的四个方向的最值点
        '''
        point_top = sample_points[0]
        point_right = sample_points[1]
        point_bottom = sample_points[2]
        point_left = sample_points[3]
        point_center = sample_points[4]

        cv2.circle(orig, point_top, 10, (0, 255, 0), -1)
        cv2.circle(orig, point_right, 10, (0, 255, 0), -1)
        cv2.circle(orig, point_bottom, 10, (0, 255, 0), -1)
        cv2.circle(orig, point_left, 10, (0, 255, 0), -1)
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

        chImage = cv2ImgAddText(orig, "长: {:.2f}cm".format(dimA), 30, 70, (255, 0, 0), 30)

        chImage = cv2ImgAddText(chImage, "宽: {:.2f}cm".format(dimB), 30, 100, (255, 0, 0), 30)

        if mask is None:
            # 计算外切矩形正中心位置处颜色的值
            (middleX, middleY) = midpoint([tlblX, tlblY], [trbrX, trbrY])
            color = orig[int(middleY), int(middleX)]
            color = color.tolist()  # 将numpy类型的数组转换为对应的python原生的list形式
        else:
            # 提取对应轮廓的颜色均值
            mean_color = cv2.mean(orig, mask)

        chImage = cv2ImgAddText(chImage, "弧长: {:.1f}cm".format(res_trans), 30, 130, (0, 0, 255), 30)

        chImage = cv2ImgAddText(chImage, "面积: {:.1f}cm2".format(area_trans), 30, 160, (0, 0, 255), 30)
        print("contour area {}: ".format(i), area_trans)

        cv2.rectangle(orig, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)

        cv2.rectangle(chImage, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)

        chImage = cv2ImgAddText(chImage,
                                "颜色: 蓝{}, 绿{}, 红{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
                                30, 190, (0, 0, 255), 30)

        cv2.imshow("Dati", chImage)
        # cv2.waitKey(0)
        return point_top, point_right, point_bottom, point_left


"""
计算对应掩码下的原始图像中的颜色的均值
image: 原始图像
mask: 目标掩码
"""


def process_contours_Color(image, mask):
    orig = image.copy()
    mean_color = cv2.mean(orig, mask)
    chImage = cv2ImgAddText(orig,
                            "颜色: 蓝{}, 绿{}, 红{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
                            30, 190, (0, 0, 255), 30)
    cv2.imshow("Dati", chImage)
    return mean_color


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    start_time = time.time()
    # pool = Pool(processes=4)
    # 定义对应的命令行参数
    args = get_args()
    width = args.width

    # 作为备用点计算，防止没有放置直尺。或者直尺没有捕捉到
    sample_A = (138, 681)
    sample_B = (179, 682)

    # 使用predict进行mask的提取
    # 大体的模型
    net = UNet(n_channels=3, n_classes=1)  # 加载样本提取模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置处理设备
    # device = torch.device('cpu')  # 设置处理设备
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))  # 加载模型

    # 直尺的模型
    net_zc = UNet(n_channels=3, n_classes=1)  # 加载直尺提取模型
    net_zc.to(device=device)
    net_zc.load_state_dict(torch.load(args.model_zc, map_location=device))  # 加载模型

    # 癌灶的模型
    net_cancer = UNet(n_channels=3, n_classes=1)
    net_cancer.to(device=device)
    net_cancer.load_state_dict(torch.load(args.model_cancer, map_location=device))

    input_files = glob.glob("./DaTi_images/*", recursive=True)

    for i, fn in enumerate(input_files):
        start = time.time()  # 开始的时间
        img = cv2.imread(fn)  # 读取文件夹中的图像
        img = cv2.resize(img, (1024, 768))
        image = img.copy()
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 构建两个网络，同时获得对应的样本轮廓的mask和直尺的mask, 来一副新的图才进行计算
        mask, mask_zc, mask_cancer = predict_img(net=net, net_zc=net_zc, net_cancer=net_cancer,
                                                 full_img=img,
                                                 scale_factor=args.scale,
                                                 out_threshold=args.mask_threshold,
                                                 device=device)

        '''
        1.对大体的mask进行处理
        '''
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

        '''
        2.对直尺的mask_zc进行处理
        '''
        # if mask_zc is not None:
        result_zc = mask_to_image(mask_zc)
        mask_zc = np.asarray(result_zc)
        backup_zc = np.zeros((mask_zc.shape[0] + 10, mask_zc.shape[1] + 10), dtype=np.uint8)  # backup_zc用于填充边缘的部分
        backup_zc[5:mask_zc.shape[0] + 5, 5: mask_zc.shape[1] + 5] = mask_zc[:, :]  # 用填充过的图等于对应的mask_zc中的信息

        # k = np.ones((21, 21), np.uint8)
        # backup_zc = cv2.morphologyEx(backup_zc, cv2.MORPH_OPEN, k)
        print(backup_zc.shape)
        edged_zc = cv2.Canny(backup_zc, 20, 150, apertureSize=3)  # 直尺边缘

        # 提取直尺对应的轮廓
        cnts_zc = cv2.findContours(edged_zc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_zc = cnts_zc[0] if imutils.is_cv2() else cnts_zc[1]

        '''
        3.对癌灶区的mask_cancer进行处理
        '''
        # 转换为对应的PIL.Image格式
        result_cancer = mask_to_image(mask_cancer)

        # 转为numpy.array的形式，使得之后的opencv能够进行调用
        mask_cancer = np.asarray(result_cancer)

        # 使用Canny进行相应的边缘检测
        edged_cancer = cv2.Canny(mask_cancer, 0, 100)  # 样本图像的边缘

        # 在经过边缘检测的图像中进行轮廓特征的提取
        cnts_cancer = cv2.findContours(edged_cancer.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # 找到对应的样本的轮廓
        cnts_cancer = cnts_cancer[0] if imutils.is_cv2() else cnts_cancer[1]  # 不同版本的opencv对应的contours的位置是不一致的

        # print("cnts: ", cnts)
        # exit()

        '''
        4.对求出的轮廓值进行处理
        '''
        # 对直尺边缘的轮廓进行捕捉
        pixelsPerMetric = process_contours_zc(edged_zc, cnts_zc, pixelsPerMetric, image)
        # 对最外面的大体轮廓进行处理
        # process_contours_DaTi(cnts, image, pixelsPerMetric, mask=mask)
        mean_color = process_contours_Color(image, mask)
        print("mean color: ", mean_color)
        print("mask: ", mask)
        print("mask type: ", type(mask))
        # 对癌灶区轮廓进行处理
        process_contours(cnts_cancer, image, pixelsPerMetric, mask=mask_cancer,
                         whole_mean_color=mean_color, whole_cnt=cnts)  # 对样本的轮廓进行相应的处理操作

        # print("time consume: ", end - start)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    end = time.time()  # 开始的时间
    # pool.close()
    print("multi time: {}".format(time.time() - start_time))
