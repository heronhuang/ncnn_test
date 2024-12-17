// ncnn_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <ncnn/net.h>
int main()
{
	ncnn::Net mobilenetv3; 
	mobilenetv3.opt.use_vulkan_compute = true; 

 
	if (mobilenetv3.load_param("./models/resnet18.ncnn.param"))
		exit(-1);
	if (mobilenetv3.load_model("./models/resnet18.ncnn.bin"))
		exit(-1);
	auto au = mobilenetv3.create_extractor();
	

}

 
