#include<iostream>
#include<torch/torch.h>
#include<torch/script.h>
#include<memory>
#include<opencv2/opencv.hpp>
#include"predict.h"
#include<math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// 对直线段进行排序
bool compareLineIndex(cv::Vec4i lines1, cv::Vec4i lines2) {
	auto i = lines1[1];
	auto j = lines2[1];
	return (i < j);
}

// 对轮廓按照x的大小进行降序排序
bool compareContourIndex_x(Point point1, Point point2) {
	auto x1 = point1.x;
	auto x2 = point2.x;
	return(x1 > x2);
}

bool compareContourIndex_y(Point point1, Point point2) {
	auto y1 = point1.y;
	auto y2 = point2.y;
	return(y1 > y2);
}

Point2f compare_distance(vector<Point2f> candidate_points, vector<Point> contour, string selecte_model) {
	if (selecte_model == "close") {
		Point2f closest_point;
		auto min_dis = 1000000;
		for (int i = 0; i < candidate_points.size(); i++) {
			auto dis = abs(cv::pointPolygonTest(contour, candidate_points[i], true));
			if (dis < min_dis) {
				min_dis = dis;
				closest_point = candidate_points[i];
			}
		}
		return closest_point;
	}
	else if (selecte_model == "far") {
		Point2f far_point;
		auto max_dis = 0;
		for (int i = 0; i < candidate_points.size(); i++) {
			auto dis = abs(cv::pointPolygonTest(contour, candidate_points[i], true));
			if (dis > max_dis) {
				max_dis = dis;
				far_point = candidate_points[i];
			}
		}
		return far_point;
	}
}

// 计算点到直线的距离
double pointToLinesDistance(cv::Vec4i lines1, cv::Vec4i lines2) {
	int x1 = lines1[0];
	int y1 = lines1[1];
	int x2 = lines1[2];
	int y2 = lines1[3];

	int x3 = lines2[0];
	int y3 = lines2[1];
	auto distance = abs(((x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)) / sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)));
	return double(distance);
}

// 计算两个点之间的距离
double twoPointsDistance(cv::Point2f point1, cv::Point2f point2)
{
	auto x1 = point1.x;
	auto y1 = point1.y;
	auto x2 = point2.x;
	auto y2 = point2.y;

	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

double twoPointDistance_int(Point point1, Point point2) {
	auto x1 = point1.x;
	auto y1 = point1.y;
	auto x2 = point2.x;
	auto y2 = point2.y;

	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

void  process_contours(vector<Point> contour, cv::Mat image, int i, double pixelDistance, cv::Mat threshold) {
	// 最小外接矩阵 
	vector<vector<Point>> contours;
	contours.push_back(contour);
	vector<RotatedRect> minRect_local(1);
	Point2f rect_points_local[4];

	for (size_t i = 0; i < 1; i++) {
		minRect_local[i] = minAreaRect(contour);
	}

	for (size_t i = 0; i < 1; i++)
	{
		Scalar color = Scalar(0, 255, 0);
		// 对样本的轮廓进行绘制
		drawContours(image, contours, (int)i, Scalar(0, 0, 255), 2);
		// rotated rectangle
		//minRect_local[i].points(rect_points_local);
		//for (int j = 0; j < 4; j++)
		//{
			//line(image, rect_points_local[j], rect_points_local[(j + 1) % 4], color);
			//circle(image, rect_points_local[j], 5, Scalar(0, 0, 255), -1);
		//}
	}

	// 计算宽和高
	auto point_top_left = rect_points_local[0];
	auto point_top_right = rect_points_local[1];
	auto point_bottom_right = rect_points_local[2];
	auto point_bottom_left = rect_points_local[3];

	Point2f middle_top, middle_bottom, middle_left, middle_right;

	middle_top.x = (point_top_right.x + point_top_left.x) / 2;
	middle_top.y = (point_top_right.y + point_top_left.y) / 2;
	middle_bottom.x = (point_bottom_right.x + point_bottom_left.x) / 2;
	middle_bottom.y = (point_bottom_right.y + point_bottom_left.y) / 2;
	middle_left.x = (point_top_left.x + point_bottom_left.x) / 2;
	middle_left.y = (point_top_left.y + point_bottom_left.y) / 2;
	middle_right.x = (point_top_right.x + point_bottom_right.x) / 2;
	middle_right.y = (point_top_right.y + point_bottom_right.y) / 2;
	 

	//circle(image, middle_bottom, 5, Scalar(0, 0, 255), -1);
	//circle(image, middle_top, 5, Scalar(0, 0, 255), -1);
	//circle(image, middle_left, 5, Scalar(0, 0, 255), -1);
	//circle(image, middle_right, 5, Scalar(0, 0, 255), -1);
	//line(image, middle_bottom, middle_top, Scalar(0, 255, 0));
	//line(image, middle_left, middle_right, Scalar(0, 255, 0));

	double width = twoPointsDistance(point_top_left, point_top_right);
	double height = twoPointsDistance(middle_bottom, middle_top);

	cout << "height: " << height * pixelDistance << endl;
	cout << "width: " << width * pixelDistance << endl;

	// 计算轮廓的面积
	auto area = contourArea(contour, false);
	std::cout << "area: " << area * pixelDistance * pixelDistance << std::endl;

	// 计算轮廓的周长
	auto length = arcLength(contour, true);
	cout << "arclength: " << length * pixelDistance << endl;

	// 获取对应轮廓中的颜色的均值
	auto color = cv::mean(image, threshold);
	cout << "color: " << color << endl;

	// 展示最终结果
	//imshow("orig", image);
	//waitKey(0);
}

void  process_contours_cancer(vector<Point> contour, cv::Mat image, int i, double pixelDistance, cv::Mat threshold, Scalar whole_mean_color,vector< vector<Point>> whole_contour, String selected_model) {
	// 最小外接矩阵 
	vector<vector<Point>> contours;
	contours.push_back(contour);
	vector<RotatedRect> minRect_local(1);
	Point2f rect_points_local[4];

	for (size_t i = 0; i < 1; i++) {
		minRect_local[i] = minAreaRect(contour);  // 找到该轮廓对应的最小外接矩阵
	}

	for (size_t i = 0; i < 1; i++)
	{
		Scalar color = Scalar(0, 255, 0);
		// 对样本的轮廓进行绘制
		drawContours(image, contours, (int)i, Scalar(0, 0, 255), 2);
		// rotated rectangle
		minRect_local[i].points(rect_points_local);

	}

	// 计算宽和高
	auto point_bottom_left = rect_points_local[0];
	auto point_top_left = rect_points_local[1];
	auto point_top_right = rect_points_local[2];
	auto point_bottom_right = rect_points_local[3];

	Point2f middle_top, middle_bottom, middle_left, middle_right;

	middle_top.x = (point_top_right.x + point_top_left.x) / 2;
	middle_top.y = (point_top_right.y + point_top_left.y) / 2;
	middle_bottom.x = (point_bottom_right.x + point_bottom_left.x) / 2;
	middle_bottom.y = (point_bottom_right.y + point_bottom_left.y) / 2;
	middle_left.x = (point_top_left.x + point_bottom_left.x) / 2;
	middle_left.y = (point_top_left.y + point_bottom_left.y) / 2;
	middle_right.x = (point_top_right.x + point_bottom_right.x) / 2;
	middle_right.y = (point_top_right.y + point_bottom_right.y) / 2;
	cout << "point_top_left: " << point_top_left << "point_top_right: " << point_top_right << "point_bottom_right: " << point_bottom_right << "point_bottom_left: " << point_bottom_left << endl;
	// 计算该轮廓的直径是否超过3cm
	auto contour_width = twoPointsDistance(middle_left, middle_right) * pixelDistance;
	auto contour_height = twoPointsDistance(middle_top, middle_bottom) * pixelDistance;
	cout << "contour_width: " << contour_width << endl;
	cout << "contour_height: " << contour_height << endl;

	double width = twoPointsDistance(point_top_left, point_top_right);
	double height = twoPointsDistance(middle_bottom, middle_top);


	// 如果轮廓的宽和长都无法超过对应的3cm标准，则将其直接使用rotated rectangle来进行标注
	if (contour_width < 3 || contour_height < 3) {
		cout << "small area" << endl;
		Scalar color = Scalar(0, 255, 0);
		for (int j = 0; j < 4; j++)
		{
			line(image, rect_points_local[j], rect_points_local[(j + 1) % 4], color);
			circle(image, rect_points_local[j], 5, Scalar(0, 0, 255), -1);

		}
		circle(image, middle_bottom, 5, Scalar(255, 0, 0), -1);
		circle(image, middle_top, 5, Scalar(255, 0, 0), -1);
		circle(image, middle_left, 5, Scalar(255, 0, 0), -1);
		circle(image, middle_right, 5, Scalar(255, 0, 0), -1);
		line(image, middle_bottom, middle_top, Scalar(0, 255, 0));
		line(image, middle_left, middle_right, Scalar(0, 255, 0));

		Point first_point(int((middle_left.x + middle_top.x)/2), int((middle_left.y + middle_top.y)/2));
		Point second_point(int((middle_right.x + middle_top.x)/2), int((middle_right.y + middle_top.y)/2));
		Point third_point(int((middle_right.x + middle_bottom.x)/2), int((middle_right.y + middle_bottom.y)/2));
		Point fourth_point(int((middle_left.x + middle_bottom.x)/2), int((middle_left.y + middle_bottom.y)/2));


		cv::putText(image, "1", first_point, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "2", second_point, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "3", third_point, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "4", fourth_point, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		/*cv::putText(image, "2", Point(int(point_left.x - 0.4*(mark_rectangle_width / 2)), int(point_left.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "3", Point(int(point_right.x - 0.4*(mark_rectangle_width / 2)), int(point_right.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "4", Point(int(point_top.x - 0.4*(mark_rectangle_width / 2)), int(point_top.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "5", Point(int(center_point.x - 0.4*(mark_rectangle_width / 2)), int(center_point.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));*/
	}
	else {
		cout << "bigger area" << endl;
		// 找到轮廓中对应的最值点
		Point center_point;  // 轮廓中间的点
		//cout << "contour information1: " << contours << endl;
		std::sort(contours[0].begin(), contours[0].end(), compareContourIndex_x);
		//cout << "contour information2: " << contours << endl;
		auto point_right = contours[0][0];  // 轮廓最右边的点
		auto point_left = contours[0][contours[0].size() - 1];  // 轮廓最左边的点
		std::sort(contours[0].begin(), contours[0].end(), compareContourIndex_y);
		auto point_top = contours[0][0];  // 轮廓最顶部的点
		auto  point_bottom = contours[0][contours[0].size() - 1];  // 轮廓最底部的点
		//cout << "contour information3: " << contours << endl;
		cout << "point_right: " << "; " << point_right << "point_left" << point_left << "; " <<
			"point top: " << point_top << "; " << "point_bottom" << point_bottom << endl;
		//system("pause");
		center_point.x = int((middle_left.x + middle_right.x) / 2);
		center_point.y = int((middle_top.y + middle_bottom.y) / 2);

		auto mark_rectangle_width = 1.0 / pixelDistance;
		auto mark_rectangle_height = 0.5 / pixelDistance;

		rectangle(image, Point(int(point_right.x - (mark_rectangle_width / 2)), int(point_right.y - (mark_rectangle_height / 2))),
			Point(int(point_right.x + (mark_rectangle_width / 2)), int(point_right.y + (mark_rectangle_height / 2))), Scalar(255, 0, 0), 0, LINE_8);

		rectangle(image, Point(int(point_left.x - (mark_rectangle_width / 2)), 
			int(point_left.y - (mark_rectangle_height / 2))),
			Point(int(point_left.x + (mark_rectangle_width / 2)),
				int(point_left.y + (mark_rectangle_height / 2))), Scalar(255, 0, 0), 0, LINE_8);

		rectangle(image, Point(int(point_top.x - (mark_rectangle_width / 2)),
			int(point_top.y - (mark_rectangle_height / 2))),
			Point(int(point_top.x + (mark_rectangle_width / 2)),
				int(point_top.y + (mark_rectangle_height / 2))), Scalar(255, 0, 0), 0, LINE_8);

		rectangle(image, Point(int(point_bottom.x - (mark_rectangle_width / 2)), int(point_bottom.y - (mark_rectangle_height / 2))),
			Point(int(point_bottom.x + (mark_rectangle_width / 2)), int(point_bottom.y + (mark_rectangle_height / 2))), Scalar(255, 0, 0), 0, LINE_8);

		rectangle(image, Point(int(center_point.x - (mark_rectangle_width / 2)), 
			int(center_point.y - (mark_rectangle_height / 2))),
			Point(int(center_point.x + (mark_rectangle_width / 2)), 
				int(center_point.y + (mark_rectangle_height / 2))), Scalar(255, 0, 0), 0, LINE_8);

		cout << "point_right: " << "; " << point_right << "point_left" << point_left << "; " <<
			"point top: " << point_top << "; " << "point_bottom" << point_bottom << endl;

		cv::putText(image, "1", Point(int(point_bottom.x - 0.4*(mark_rectangle_width / 2)), int(point_bottom.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "2", Point(int(point_left.x - 0.4*(mark_rectangle_width / 2)),int(point_left.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "3", Point(int(point_right.x - 0.4*(mark_rectangle_width / 2)), int(point_right.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "4", Point(int(point_top.x - 0.4*(mark_rectangle_width / 2)),int(point_top.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		cv::putText(image, "5", Point(int(center_point.x - 0.4*(mark_rectangle_width / 2)),int(center_point.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		/*circle(image, point_bottom, 5, Scalar(255, 0, 0), -1);
		circle(image, point_left, 5, Scalar(255, 0, 0), -1);
		circle(image, point_right, 5, Scalar(255, 0, 0), -1);
		circle(image, point_top, 5, Scalar(255, 0, 0), -1);
		circle(image, center_point, 5, Scalar(255, 0, 0), -1);*/

		// 定义可能小于1cm的候选框
		Point2f top_right(point_right.x, point_top.y);
		Point2f bottom_right (point_right.x, point_bottom.y);
		Point2f top_left(point_left.x, point_top.y);
		Point2f bottom_left(point_left.x, point_bottom.y);
		vector<Point2f> candidate_points_closer;  // 建立vector用于存储所有的
		
		candidate_points_closer.push_back(top_right);
		candidate_points_closer.push_back(bottom_right);
		candidate_points_closer.push_back(top_left);
		candidate_points_closer.push_back(bottom_left);
		//cout << "candidate_points_closer: " << candidate_points_closer << endl;
		//std::cout << "whole contour: " << whole_contour << endl;

		// 定义大于1cm的候选框
		Point2f top_far_point(point_top.x, point_top.y - 2 / pixelDistance);
		Point2f right_far_point(point_right.x + 2 / pixelDistance, point_right.y);
		Point2f bottom_far_point(point_bottom.x, point_bottom.y + 2 / pixelDistance);
		Point2f left_far_point(point_left.x - 2 / pixelDistance, point_left.y);

		vector<Point2f> candidate_points_far;
		candidate_points_far.push_back(top_far_point);
		candidate_points_far.push_back(right_far_point);
		candidate_points_far.push_back(bottom_far_point);
		candidate_points_far.push_back(left_far_point);

		// 候选框必须在轮廓内部，且距离癌灶区轮廓不能超过1cm
		for (vector<Point2f>::iterator it = candidate_points_closer.begin(); it != candidate_points_closer.end(); ++it) {
			bool flag = false;
			for (int i = 0; i < whole_contour.size(); i++) {
				cout << "our sorted distance: " << cv::pointPolygonTest(whole_contour[i], *it, false) << endl;
				// 点落在任何一个大体轮廓上都是可以的，便设置flag为true
				if (cv::pointPolygonTest(whole_contour[i], *it, false) == 1) {
					flag = true;
				}
				cout << "inner +++++++++++++++" << endl;
			}
			if (!flag) {
				cout << "not erase candidate_points_closer: " << candidate_points_closer << endl;
				it = candidate_points_closer.erase(it);
				if (it == candidate_points_closer.end()) break; // 如果是到了候选框的最后一个点，则不要继续++了，直接break掉
				continue;
			}
			//system("pause");
			// 如果是选择对应的近癌旁的点
			auto sub_dis = abs(cv::pointPolygonTest(contour, *it, true)) * pixelDistance;
			cout << "sub dis: " << sub_dis << endl;
			if (sub_dis > 1) {
				cout << "**************" << endl;
				it = candidate_points_closer.erase(it);
				if (it == candidate_points_closer.end()) break;  // 如果是到了候选框的最后一个点，则不要继续++了，直接break掉
				//continue;
			}
			cout << "final candidate_points_closer: " << candidate_points_closer << endl;
			cout << "outer +++++++++++++++++++++++" << endl;
		}

		for (vector<Point2f>::iterator it = candidate_points_far.begin(); it != candidate_points_far.end(); ++it) {
			bool flag = false;
			for (int i = 0; i < whole_contour.size(); i++) {
				cout << "our sorted distance: " << cv::pointPolygonTest(whole_contour[i], *it, false) << endl;
				// 点落在任何一个大体轮廓上都是可以的，便设置flag为true
				if (cv::pointPolygonTest(whole_contour[i], *it, false) == 1) {
					flag = true;
				}
				cout << "inner +++++++++++++++" << endl;
			}
			if (!flag) {
				cout << "inner candidate_points_far: " << candidate_points_far << endl;
				it = candidate_points_far.erase(it);
				if (it == candidate_points_far.end()) break; // 如果是到了候选框的最后一个点，则不要继续++了，直接break掉
				continue;
			}
			//system("pause");
			// 如果是选择对应的近癌旁的点
			auto sub_dis = abs(cv::pointPolygonTest(contour, *it, true)) * pixelDistance;
			cout << "sub dis: " << sub_dis << endl;
			if (sub_dis < 1) {
				cout << "**************" << endl;
				it = candidate_points_far.erase(it);
				if (it == candidate_points_far.end()) break; // 如果是到了候选框的最后一个点，则不要继续++了，直接break掉
			}
			cout << "final candidate_points_far: " << candidate_points_far << endl;
			cout << "outer +++++++++++++++++++++++" << endl;
		}
		if (candidate_points_closer.size() != 0) {
			auto close_point = compare_distance(candidate_points_closer, contour, "close");
			Point close_point_int(int(close_point.x), int(close_point.y));
			rectangle(image, Point(int(close_point_int.x - (mark_rectangle_width / 2)),
				int(close_point_int.y - (mark_rectangle_height / 2))),
				Point(int(close_point_int.x + (mark_rectangle_width / 2)),
					int(close_point_int.y + (mark_rectangle_height / 2))), Scalar(0, 255, 0), 0, LINE_8);
			cv::putText(image, "6", Point(int(close_point_int.x - 0.4*(mark_rectangle_width / 2)), int(close_point_int.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		}
		
		if (candidate_points_far.size() != 0) {
			// 在对候选点的位置与距离关系选择完毕之后，就可以根据近或者远选择最近的点或者最远的点作为标注框
			auto far_point = compare_distance(candidate_points_far, contour, "far");

			Point far_point_int(int(far_point.x), int(far_point.y));

			rectangle(image, Point(int(far_point_int.x - (mark_rectangle_width / 2)),
				int(far_point_int.y - (mark_rectangle_height / 2))),
				Point(int(far_point_int.x + (mark_rectangle_width / 2)),
					int(far_point_int.y + (mark_rectangle_height / 2))), Scalar(0, 0, 255), 0, LINE_8);
			cv::putText(image, "7", Point(int(far_point_int.x - 0.4*(mark_rectangle_width / 2)), int(far_point_int.y + 0.5 * (mark_rectangle_height / 2))), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255));
		}
	}

	std::cout << "height: " << height * pixelDistance << endl;
	cout << "width: " << width * pixelDistance << endl;

	// 计算轮廓的面积
	auto area = contourArea(contour, false);
	std::cout << "area: " << area * pixelDistance * pixelDistance << std::endl;

	// 计算轮廓的周长
	auto length = arcLength(contour, true);
	cout << "arclength: " << length * pixelDistance << endl;

	// 获取对应轮廓中的颜色的均值
	auto color = cv::mean(image, threshold);
	cout << "color: " << color << endl;

	// 展示最终结果
	/*imshow("cancer", image);
	waitKey(0);*/
}

int main()
{	
	clock_t startTime, endTime;

	startTime = clock(); // 计时开始
	cout << torch::cuda::is_available() << endl;

	//读取图片
	auto image = cv::imread("B20210612.jpg");

	auto orig = image.clone();

	// 将图片大小转换为指定大小
	cv::resize(image, image, cv::Size(1024, 768));

	// 对图像进行预处理后方便后续直接输入模型
	auto result = preprocess(image, 0.5, "unet.pt"); // 大体形成的mask
	auto result_zc = preprocess(image, 0.5, "unet_zc.pt");  // 直尺的mask
	auto result_cancer = preprocess(image, 0.5, "unet_cancer.pt"); // 癌灶区的mask

	// 对直尺的结果进行二值化
	threshold(result_zc, result_zc, 100, 255, THRESH_BINARY);
	Mat dst, cdst, cdstP, dis;

	// 对直尺进行边缘检测
	Canny(result_zc, dst, 50, 200, 3);
	//cvtColor(dst, cdstP, COLOR_GRAY2BGR);
	cvtColor(dst, dis, COLOR_GRAY2BGR);

	// 基于统计的霍夫变换
	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection

	std::sort(linesP.begin(), linesP.end(), compareLineIndex);

	auto distance = pointToLinesDistance(linesP[1], linesP[linesP.size() - 1]);
	line(dis, Point(linesP[1][0], linesP[1][1]), Point(linesP[1][2], linesP[1][3]), Scalar(0, 255, 0), 3, LINE_AA);
	line(dis, Point(linesP[linesP.size() - 1][0], linesP[linesP.size() - 1][1]), Point(linesP[linesP.size() - 1][2], linesP[linesP.size() - 1][3]), Scalar(0, 255, 0), 3, LINE_AA);
	//imshow("dis", dis);
	//waitKey(0);

	// 得出对应的比例尺, 比例尺是十分关键的
	double pixeldistance = 3 / distance;

	/*
	计算出比例尺之后对大体轮廓信息进行处理
	*/

	// 二值化处理
	threshold(result, result, 120, 255, THRESH_BINARY);
	auto threshold = result;

	Mat edged;
	// 对边缘进行高斯模糊
	//GaussianBlur(result, result, Size(5, 5), 0);

	// 进行边缘检测
	Canny(result, edged, 50, 150);

	// 寻找对应的轮廓
	vector<vector<Point>> contours;
	vector<vector<Point>> whole_contours;

	findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));  // 将检测到的边缘信息保存到contours中
	whole_contours.assign(contours.begin(), contours.end());

	for (size_t i = 0; i < contours.size(); i++) {
		if (arcLength(contours[i], true) < 180) {
			continue;
		}
		process_contours(contours[i], image, i, pixeldistance, threshold);
	}

	// 计算整体轮廓对应的颜色均值
	auto whole_mean_color = cv::mean(orig, threshold);  // whole为大体轮廓对应的颜色均值
	cout << whole_mean_color[0] << endl;
	

	/*
	对癌灶区的mask进行处理
	*/
	// 对癌灶区的mask进行二值化处理
	cv::threshold(result_cancer, result_cancer, 120, 255, THRESH_BINARY);
	auto threshold_cancer = result_cancer;

	// 使用canny边缘检测对当前的二值化癌灶区轮廓进行处理
	cv::Mat edged_cancer;
	cv::Canny(result_cancer, edged_cancer, 50, 150);

	// 将癌灶区的轮廓信息存储在contours_cancer中
	std::vector<vector<Point>> contours_cancer;
	cv::findContours(edged_cancer, contours_cancer, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	for (int i = 0; i < contours_cancer.size(); i++) {
		if (arcLength(contours_cancer[i], true) < 100) {
			continue;
		}
		process_contours_cancer(contours_cancer[i], image, i, pixeldistance, threshold_cancer, whole_mean_color, whole_contours, "closer");
	}

	imshow("image", image);
	//imshow("orig", orig);
	waitKey(0);

	cout << "function success |||" << endl;

	endTime = clock();
	cout << "run time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	//system("pause");
	return 0;
}
