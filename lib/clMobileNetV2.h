#define _CRT_SECURE_NO_WARNINGS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <assert.h>
#include <fstream>
#include <CL/cl.h>

#define FLOAT sizeof(float)

struct BMP {
	unsigned char header[54];
	unsigned char data[224][224][3];
};

struct memList {
	cl_mem OUTPUT;
	cl_mem WEIGHT;
	cl_mem BIAS;
	cl_mem BN;
	cl_kernel KERNEL;
	size_t DIM;
	size_t GWI[3];
	size_t LWI[3];
};

class mobilenetv2 {

public:
	/* OpenCL Initialize Function */
	mobilenetv2(unsigned int, unsigned int);
	void clInitialize(const char*, const char*);
	void clInference();
	void clShowResult();
	void clShowTimeProfile();
	void clLayerData(int, int);
	void test();
	/* Tool Member Function */
	void clErrorCheck(cl_int, int, bool);
	void clImageLoader(const char*);
	

	/* Visible Member Variable*/
	cl_uint Ndevice;
	cl_uint Nplatform;
	cl_platform_id PLATFORM;
	cl_device_id DEVICE;
	int PlatformIndex;
	int DeviceIndex;
	int ParameterSize;
	float* Image;
	float* Result;
	cl_event* EVs;
	memList* MEMLIST;

private:
	short Classes;
	cl_platform_id* PLATFORMS;
	cl_device_id* DEVICES;
	cl_context CONTEXT;
	cl_command_queue QUEUE;
	cl_program PROGRAM;
};
