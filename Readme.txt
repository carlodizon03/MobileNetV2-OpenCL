MobileNet V2 with OpenCL

class index:
Number 0 : cat
Number 1 : dog

File:
- weight.bin : MobileNetV2 weights for two classes classification
- lib/clMobileNetV2.cpp and lib/clMobileNetV2.h :
	> OpenCL initialization
	> MobileNetV2 building
	> Inference...
- mobilenetv2.cl : kernel code for MobileNetV2
- Jupyter Notebook/MobileNetv2.ipynb : python code (training with keras/tensorflow)
- Jupyter Notebook/MobileNetV2_dogcat.h5 : weights file



Compile on IDO:

File needed:
 - lib/clMobileNetV2.cpp and lib/clMobileNetV2.h
 - test_img
 - MobileNet_v2_dev.cpp
 - weight.bin
 - mobilenetv2.cl
 
Compiling:
 >> sudo apt update
 >> sudo apt install build-essential linux-headers-$(uname -r) libelf-dev
 >> sudo apt-get install libclc-amdgcn mesa-opencl-icd ocl-icd-opencl-dev opencl-headers clinfo
 ## install opencl on IDO
 
 >> g++ MobileNet_v2_dev.cpp lib/clMobileNetV2.cpp -o MobileNetV2 -lOpenCL
 ## compile on IDO
 
Execute example:
 >> ./MobileNetV2 "weight.bin" "test_img/test1.bmp"
 >> ./MobileNetV2 "weight.bin" "test_img/test2.bmp"
 >> ./MobileNetV2 "weight.bin" "test_img/test3.bmp"
 >> ./MobileNetV2 "weight.bin" "test_img/test4.bmp"