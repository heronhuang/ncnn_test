// ncnn_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <ncnn/net.h>
#include <chrono>
int main()
{
	ncnn::Net mobilenetv3; 
	mobilenetv3.opt.use_vulkan_compute = true; 
	
 
	if (mobilenetv3.load_param("./models/resnet18.ncnn.param"))
		exit(-1);
	if (mobilenetv3.load_model("./models/resnet18.ncnn.bin"))
		exit(-1);
	auto start = std::chrono::high_resolution_clock::now();
	auto extractor = mobilenetv3.create_extractor();
	ncnn::Mat input(1,3,360,640); ncnn::Mat  output;
	for (int i = 0; i < 100; i++)
	{ 
		extractor.input("input", input); 
		extractor.extract("output", output);
	}
	auto end = std::chrono::high_resolution_clock::now();
	
	auto duration = end - start;
	float ecliped   = (float)std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()/100.0f;

	std::cout << "time:" << ecliped;
}

 
