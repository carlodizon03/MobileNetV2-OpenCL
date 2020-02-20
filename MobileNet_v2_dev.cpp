#include "lib/clMobileNetV2.h"
#include <string.h>

int main()
{
	mobilenetv2 model(0, 0);
	const char* clPath = "mobilenetv2.cl";
	const char* weightPath = "weight.bin";
	const char* imgPath = "test_img/test1.bmp";
	model.clInitialize(clPath, weightPath);
	model.clImageLoader(imgPath);
	//model.test();
	/*if(argc > 2)
	{
		model.clInitialize("D:/Documents/Course Materials/Deep Learning Projects/Liscotech MobileNEtv2/MobileNetV2_IDO/MobileNetv2 -OpenCl Implementation/MobileNetv-OpenCL/mobilenetv2.cl", argv[1]);
		model.clImageLoader(argv[2]);
	}
		
	else if(argc > 1)
	{
		model.clInitialize("mobilenetv2.cl", "mobilenetv2_parameter.bin");
		model.clImageLoader(argv[1]);
	}*/
	
	model.clInference();
	model.clShowTimeProfile();
	model.clShowResult();
}
