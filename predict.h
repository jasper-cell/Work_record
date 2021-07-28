#pragma once
#pragma once
#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<torch/script.h>

using namespace cv;
using namespace std;
Mat preprocess(Mat pil_img, float scale, String path, float out_threshold = 0.5)
{
	auto w = pil_img.cols;
	auto h = pil_img.rows;

	auto newW = int(scale * w);
	auto newH = int(scale * h);

	resize(pil_img, pil_img, Size(newW, newH));

	// ���Ƚ���ɫͨ������ת��
	cvtColor(pil_img, pil_img, CV_BGR2RGB);
	// ���������ͽ��д���
	pil_img.convertTo(pil_img, CV_32FC3, 1.0f / 255.0f);
	//opencv format H*W*C
	auto input_tensor = torch::from_blob(pil_img.data, { 1, newH, newW, 3 });
	//pytorch format N*C*H*W
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
	//input_tensor = input_tensor.to(at::kCUDA);  // ����������Ǩ�Ƶ�GPU��

	// create model
	auto model = torch::jit::load(path);
	//model.to(at::kCUDA);  // ��ģ��Ǩ�Ƶ�GPU��

	//model.eval();

	//ǰ�򴫲�
	
	//system("pause");
	auto output = model.forward({input_tensor}).toTensor().sigmoid();
	//auto probs = torch::sigmoid(output).to(at::kCPU);
	cout << "caculate sucess" << endl;
	

	// ��batch�����ѹ��
	//auto probs = output.to(at::kCPU).squeeze(0).detach().permute({ 1, 2, 0 });
	auto probs = output.squeeze(0).detach().permute({ 1, 2, 0 });

	//probs = probs.permute({ 1, 2, 0 });

	probs = probs.mul(255).clamp(0, 255).to(torch::kU8);
	cv::Mat resultImg(newH, newW, CV_8UC1);

	std::memcpy((void*)resultImg.data, probs.data_ptr(), sizeof(torch::kU8) * probs.numel());

	resize(resultImg, resultImg, Size(w, h));

	return resultImg;
}