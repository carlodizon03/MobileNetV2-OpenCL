#include "clMobileNetV2.h"

mobilenetv2::mobilenetv2(unsigned int platformIndex, unsigned int deviceIndex)
{
	cl_int RET;
	RET = clGetPlatformIDs(0, NULL, &Nplatform);
	clErrorCheck(RET, __LINE__, true);
	PLATFORMS = new cl_platform_id[Nplatform];
	RET = clGetPlatformIDs(Nplatform, PLATFORMS, NULL);
	clErrorCheck(RET, __LINE__, true);
	if (Nplatform == 0)
	{
		printf("No platform.\n");
		assert(0);
	}
	if (platformIndex >= Nplatform)
	{
		printf("Invaild platform index.\n");
		assert(0);
	}
	PlatformIndex = platformIndex;
	PLATFORM = PLATFORMS[PlatformIndex];
	/* choose one platform you will use */
	

	RET = clGetDeviceIDs(PLATFORM, CL_DEVICE_TYPE_ALL, 0, NULL, &Ndevice);
	clErrorCheck(RET, __LINE__, true);

	DEVICES = new cl_device_id[Ndevice];
	RET = clGetDeviceIDs(PLATFORM, CL_DEVICE_TYPE_ALL, Ndevice, DEVICES, NULL);
	clErrorCheck(RET, __LINE__, true);
	if (Ndevice == 0)
	{
		printf("No device.\n");
		assert(0);
	}
	if (deviceIndex >= Ndevice)
	{
		printf("Invaild device index.\n");
		assert(0);
	}

	cl_context_properties ContextProp[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)PLATFORM, 0 };
	CONTEXT = clCreateContext(ContextProp, Ndevice, DEVICES, NULL, NULL, &RET);
	clErrorCheck(RET, __LINE__, true);
	/* Create context on platform */

	DeviceIndex = deviceIndex;
	DEVICE = DEVICES[DeviceIndex];
	/* choose one device you will use */

	QUEUE = clCreateCommandQueue(CONTEXT, DEVICE, CL_QUEUE_PROFILING_ENABLE, &RET);
	clErrorCheck(RET, __LINE__, true);
	/* Create a command queue object on device */

	printf("\n");
	size_t Nchar;
	RET = clGetPlatformInfo(PLATFORM, CL_PLATFORM_NAME, 0, NULL, &Nchar);
	clErrorCheck(RET, __LINE__, false);
	char* PlatformName = new char[Nchar];
	RET = clGetPlatformInfo(PLATFORM, CL_PLATFORM_NAME, Nchar, PlatformName, NULL);
	clErrorCheck(RET, __LINE__, false);
	printf("Platform %d : %s\n\n", PlatformIndex, PlatformName);

	RET = clGetDeviceInfo(DEVICE, CL_DEVICE_NAME, 0, NULL, &Nchar);
	clErrorCheck(RET, __LINE__, false);
	char* DeviceName = new char[Nchar];
	RET = clGetDeviceInfo(DEVICE, CL_DEVICE_NAME, Nchar, DeviceName, NULL);
	clErrorCheck(RET, __LINE__, false);
	printf("Device   %d : %s\n\n", DeviceIndex, DeviceName);

	delete[] PlatformName;
	delete[] DeviceName;

	printf("Context and command queue created.\n\n");
	printf("------------------------------------------------------------------------\n\n");
}

void mobilenetv2::clInitialize(const char* KernelFileName, const char* ParaFileName)
{
	Classes = 2;
	Result = new float[Classes];
	FILE* programHandle;
	char* programBuffer;
	size_t programSize;
	programHandle = fopen(KernelFileName, "rb");

	if (programHandle != NULL)
	{
		fseek(programHandle, 0, SEEK_END);
		programSize = ftell(programHandle);
		rewind(programHandle);
		programBuffer = (char*)malloc(programSize + 1);
		fread(programBuffer, sizeof(char), programSize, programHandle);
		*(programBuffer + programSize) = '\0';
		fclose(programHandle);
	}
	else
	{
		printf("CL file cannot open.\n");
		assert(0);
	}
	/* read kernel code from external file */
	printf("Kernel code reading completed.\n\n");

	cl_int RET;
	PROGRAM = clCreateProgramWithSource(CONTEXT, 1, (const char**)& programBuffer, &programSize, &RET); 
	/* create kernel code program object */
	clErrorCheck(RET, __LINE__, true);
	

	RET = clBuildProgram(PROGRAM, 1, &DEVICE, "-cl-std=CL1.1", 0, 0);
	/* compile kernel program */
	clErrorCheck(RET, __LINE__, false);
	if (RET != CL_SUCCESS)
	{
		int Nchar = 65535;
		char* Message = new char[Nchar];
		clGetProgramBuildInfo(PROGRAM, DEVICE, CL_PROGRAM_BUILD_LOG, Nchar, Message, NULL);
		printf("\nCompile Fail Message :\n");
		printf("%s\n", Message);
		assert(0);
	}
	
	printf("Kernel code compiling success.\n\n");
	printf("------------------------------------------------------------------------\n\n");

	FILE* fp;
	fp = fopen(ParaFileName, "rb"); /* read weights file */
	fseek(fp, 0, SEEK_END);
	int FileSize = ftell(fp);
	ParameterSize = FileSize / sizeof(float);
	int ReadCounter = ParameterSize;
	printf("File size : %d Bytes\n\n", FileSize);
	printf("Number of parameter : %d\n\n", ParameterSize);
	rewind(fp);
	

	float* WeightTemp;
	float* BiasTemp;
	float* BNTemp;
	int OutputSize, WeightSize, BiasSize, BNSize;
	MEMLIST = new memList[66];


	/* Input Layer */
	MEMLIST[0].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * 224 * 224 * 3, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	
	/* ZeroPadding + Convolution(3x3, /2) + BN + ReLU6 */
	OutputSize = 112 * 112 * 32;
	WeightSize = 32 * 3 * 3 * 3;
	BNSize = 32 * 4;
	WeightTemp = new float[WeightSize]; /* buffer on host */
	BNTemp = new float[BNSize];         /* buffer on host */
	MEMLIST[1].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[1].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[1].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	/* read weight from bin file*/
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[1].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	/* write weight into buffer on Deivce(GPU) */
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[1].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	/* write weight into buffer on Deivce(GPU) */
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[1].KERNEL = clCreateKernel(PROGRAM, "clConvBNReLU_A0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[1].KERNEL, 0, sizeof(cl_mem), &MEMLIST[0].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[1].KERNEL, 1, sizeof(cl_mem), &MEMLIST[1].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[1].KERNEL, 2, sizeof(cl_mem), &MEMLIST[1].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[1].KERNEL, 3, sizeof(cl_mem), &MEMLIST[1].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[1].DIM = 3;
	MEMLIST[1].GWI[0] = 112;
	MEMLIST[1].GWI[1] = 112;
	MEMLIST[1].GWI[2] = 32;
	MEMLIST[1].LWI[0] = 16;
	MEMLIST[1].LWI[1] = 16;
	MEMLIST[1].LWI[2] = 1;

	/* Group 1 : 1 block, 1 stride */
	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 112 * 112 * 32;
	WeightSize = 32 * 3 * 3;
	BNSize = 32 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[2].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[2].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[2].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[2].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[2].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[2].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_B0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[2].KERNEL, 0, sizeof(cl_mem), &MEMLIST[1].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[2].KERNEL, 1, sizeof(cl_mem), &MEMLIST[2].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[2].KERNEL, 2, sizeof(cl_mem), &MEMLIST[2].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[2].KERNEL, 3, sizeof(cl_mem), &MEMLIST[2].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[2].DIM = 3;
	MEMLIST[2].GWI[0] = 112;
	MEMLIST[2].GWI[1] = 112;
	MEMLIST[2].GWI[2] = 32;
	MEMLIST[2].LWI[0] = 16;
	MEMLIST[2].LWI[1] = 16;
	MEMLIST[2].LWI[2] = 1;


	/* Point Wise Convolution(1x1) + BN */
	OutputSize = 112 * 112 * 16;
	WeightSize = 16 * 32 * 1 * 1;
	BNSize = 16 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[3].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[3].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[3].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[3].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[3].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[3].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_B1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[3].KERNEL, 0, sizeof(cl_mem), &MEMLIST[2].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[3].KERNEL, 1, sizeof(cl_mem), &MEMLIST[3].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[3].KERNEL, 2, sizeof(cl_mem), &MEMLIST[3].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[3].KERNEL, 3, sizeof(cl_mem), &MEMLIST[3].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[3].DIM = 3;
	MEMLIST[3].GWI[0] = 112;
	MEMLIST[3].GWI[1] = 112;
	MEMLIST[3].GWI[2] = 16;
	MEMLIST[3].LWI[0] = 8;
	MEMLIST[3].LWI[1] = 8;
	MEMLIST[3].LWI[2] = 4;


	/* Group 2 : 2 block, 2 stride */

	/* 1st block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 112 * 112 * 96;
	WeightSize = 96 * 16 * 1 * 1;
	BNSize = 96 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[4].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[4].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[4].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[4].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[4].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[4].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_C0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[4].KERNEL, 0, sizeof(cl_mem), &MEMLIST[3].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[4].KERNEL, 1, sizeof(cl_mem), &MEMLIST[4].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[4].KERNEL, 2, sizeof(cl_mem), &MEMLIST[4].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[4].KERNEL, 3, sizeof(cl_mem), &MEMLIST[4].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[4].DIM = 3;
	MEMLIST[4].GWI[0] = 112;
	MEMLIST[4].GWI[1] = 112;
	MEMLIST[4].GWI[2] = 96;
	MEMLIST[4].LWI[0] = 8;
	MEMLIST[4].LWI[1] = 8;
	MEMLIST[4].LWI[2] = 4;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 56 * 56 * 96;
	WeightSize = 96 * 3 * 3;
	BNSize = 96 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[5].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[5].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[5].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[5].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[5].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[5].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_C1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[5].KERNEL, 0, sizeof(cl_mem), &MEMLIST[4].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[5].KERNEL, 1, sizeof(cl_mem), &MEMLIST[5].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[5].KERNEL, 2, sizeof(cl_mem), &MEMLIST[5].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[5].KERNEL, 3, sizeof(cl_mem), &MEMLIST[5].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[5].DIM = 3;
	MEMLIST[5].GWI[0] = 56;
	MEMLIST[5].GWI[1] = 56;
	MEMLIST[5].GWI[2] = 96;
	MEMLIST[5].LWI[0] = 8;
	MEMLIST[5].LWI[1] = 8;
	MEMLIST[5].LWI[2] = 4;

	/* Point Wise Convolution(1x1) + BN (Layer 6)*/
	OutputSize = 56 * 56 * 24;
	WeightSize = 24 * 96 * 1 * 1;
	BNSize = 24 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[6].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[6].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[6].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[6].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[6].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[6].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_C2", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[6].KERNEL, 0, sizeof(cl_mem), &MEMLIST[5].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[6].KERNEL, 1, sizeof(cl_mem), &MEMLIST[6].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[6].KERNEL, 2, sizeof(cl_mem), &MEMLIST[6].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[6].KERNEL, 3, sizeof(cl_mem), &MEMLIST[6].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[6].DIM = 3;
	MEMLIST[6].GWI[0] = 56;
	MEMLIST[6].GWI[1] = 56;
	MEMLIST[6].GWI[2] = 24;
	MEMLIST[6].LWI[0] = 8;
	MEMLIST[6].LWI[1] = 8;
	MEMLIST[6].LWI[2] = 4;

	/* 2ed block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 56 * 56 * 144;
	WeightSize = 144 * 24 * 1 * 1;
	BNSize = 144 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[7].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[7].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[7].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[7].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[7].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[7].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_C3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[7].KERNEL, 0, sizeof(cl_mem), &MEMLIST[6].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[7].KERNEL, 1, sizeof(cl_mem), &MEMLIST[7].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[7].KERNEL, 2, sizeof(cl_mem), &MEMLIST[7].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[7].KERNEL, 3, sizeof(cl_mem), &MEMLIST[7].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[7].DIM = 3;
	MEMLIST[7].GWI[0] = 56;
	MEMLIST[7].GWI[1] = 56;
	MEMLIST[7].GWI[2] = 144;
	MEMLIST[7].LWI[0] = 8;
	MEMLIST[7].LWI[1] = 8;
	MEMLIST[7].LWI[2] = 4;

	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 56 * 56 * 144;
	WeightSize = 144 * 3 * 3;
	BNSize = 144 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[8].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[8].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[8].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[8].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[8].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[8].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_C4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[8].KERNEL, 0, sizeof(cl_mem), &MEMLIST[7].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[8].KERNEL, 1, sizeof(cl_mem), &MEMLIST[8].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[8].KERNEL, 2, sizeof(cl_mem), &MEMLIST[8].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[8].KERNEL, 3, sizeof(cl_mem), &MEMLIST[8].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[8].DIM = 3;
	MEMLIST[8].GWI[0] = 56;
	MEMLIST[8].GWI[1] = 56;
	MEMLIST[8].GWI[2] = 144;
	MEMLIST[8].LWI[0] = 8;
	MEMLIST[8].LWI[1] = 8;
	MEMLIST[8].LWI[2] = 4;

	/* Point Wise Convolution(1x1) + BN (Layer 9)*/
	OutputSize = 56 * 56 * 24;
	WeightSize = 24 * 144 * 1 * 1;
	BNSize = 24 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[9].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[9].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[9].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[9].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[9].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[9].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_C5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[9].KERNEL, 0, sizeof(cl_mem), &MEMLIST[8].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[9].KERNEL, 1, sizeof(cl_mem), &MEMLIST[9].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[9].KERNEL, 2, sizeof(cl_mem), &MEMLIST[9].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[9].KERNEL, 3, sizeof(cl_mem), &MEMLIST[9].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[9].DIM = 3;
	MEMLIST[9].GWI[0] = 56;
	MEMLIST[9].GWI[1] = 56;
	MEMLIST[9].GWI[2] = 24;
	MEMLIST[9].LWI[0] = 8;
	MEMLIST[9].LWI[1] = 8;
	MEMLIST[9].LWI[2] = 4;

	/* Add Layer 9 and Layer 6 (Layer 10)*/
	OutputSize = 56 * 56 * 24;
	MEMLIST[10].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[10].KERNEL = clCreateKernel(PROGRAM, "clAdd_C6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[10].KERNEL, 0, sizeof(cl_mem), &MEMLIST[9].OUTPUT);   clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[10].KERNEL, 1, sizeof(cl_mem), &MEMLIST[6].OUTPUT);   clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[10].KERNEL, 2, sizeof(cl_mem), &MEMLIST[10].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[10].DIM = 3;
	MEMLIST[10].GWI[0] = 56;
	MEMLIST[10].GWI[1] = 56;
	MEMLIST[10].GWI[2] = 24;
	MEMLIST[10].LWI[0] = 8;
	MEMLIST[10].LWI[1] = 8;
	MEMLIST[10].LWI[2] = 4;


	/* Group 3 : 3 block, 2 stride */

	/* 1st block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 56 * 56 * 144;
	WeightSize = 144 * 24 * 1 * 1;
	BNSize = 144 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[11].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[11].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[11].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[11].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[11].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[11].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_D0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[11].KERNEL, 0, sizeof(cl_mem), &MEMLIST[10].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[11].KERNEL, 1, sizeof(cl_mem), &MEMLIST[11].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[11].KERNEL, 2, sizeof(cl_mem), &MEMLIST[11].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[11].KERNEL, 3, sizeof(cl_mem), &MEMLIST[11].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[11].DIM = 3;
	MEMLIST[11].GWI[0] = 56;
	MEMLIST[11].GWI[1] = 56;
	MEMLIST[11].GWI[2] = 144;
	MEMLIST[11].LWI[0] = 8;
	MEMLIST[11].LWI[1] = 8;
	MEMLIST[11].LWI[2] = 4;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 28 * 28 * 144;
	WeightSize = 144 * 3 * 3;
	BNSize = 144 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[12].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[12].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[12].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[12].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[12].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[12].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_D1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[12].KERNEL, 0, sizeof(cl_mem), &MEMLIST[11].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[12].KERNEL, 1, sizeof(cl_mem), &MEMLIST[12].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[12].KERNEL, 2, sizeof(cl_mem), &MEMLIST[12].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[12].KERNEL, 3, sizeof(cl_mem), &MEMLIST[12].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[12].DIM = 3;
	MEMLIST[12].GWI[0] = 28;
	MEMLIST[12].GWI[1] = 28;
	MEMLIST[12].GWI[2] = 144;
	MEMLIST[12].LWI[0] = 14;
	MEMLIST[12].LWI[1] = 14;
	MEMLIST[12].LWI[2] = 1;

	/* Point Wise Convolution(1x1) + BN (Layer 13)*/
	OutputSize = 28 * 28 * 32;
	WeightSize = 32 * 144 * 1 * 1;
	BNSize = 32 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[13].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[13].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[13].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[13].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[13].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[13].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_D2", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[13].KERNEL, 0, sizeof(cl_mem), &MEMLIST[12].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[13].KERNEL, 1, sizeof(cl_mem), &MEMLIST[13].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[13].KERNEL, 2, sizeof(cl_mem), &MEMLIST[13].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[13].KERNEL, 3, sizeof(cl_mem), &MEMLIST[13].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[13].DIM = 3;
	MEMLIST[13].GWI[0] = 28;
	MEMLIST[13].GWI[1] = 28;
	MEMLIST[13].GWI[2] = 32;
	MEMLIST[13].LWI[0] = 4;
	MEMLIST[13].LWI[1] = 4;
	MEMLIST[13].LWI[2] = 16;

	/* 2ed block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 28 * 28 * 192;
	WeightSize = 192 * 32 * 1 * 1;
	BNSize = 192 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[14].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[14].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[14].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[14].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[14].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[14].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_D3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[14].KERNEL, 0, sizeof(cl_mem), &MEMLIST[13].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[14].KERNEL, 1, sizeof(cl_mem), &MEMLIST[14].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[14].KERNEL, 2, sizeof(cl_mem), &MEMLIST[14].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[14].KERNEL, 3, sizeof(cl_mem), &MEMLIST[14].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[14].DIM = 3;
	MEMLIST[14].GWI[0] = 28;
	MEMLIST[14].GWI[1] = 28;
	MEMLIST[14].GWI[2] = 192;
	MEMLIST[14].LWI[0] = 4;
	MEMLIST[14].LWI[1] = 4;
	MEMLIST[14].LWI[2] = 16;


	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 28 * 28 * 192;
	WeightSize = 192 * 3 * 3;
	BNSize = 192 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[15].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[15].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[15].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[15].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[15].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[15].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_D4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[15].KERNEL, 0, sizeof(cl_mem), &MEMLIST[14].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[15].KERNEL, 1, sizeof(cl_mem), &MEMLIST[15].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[15].KERNEL, 2, sizeof(cl_mem), &MEMLIST[15].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[15].KERNEL, 3, sizeof(cl_mem), &MEMLIST[15].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[15].DIM = 3;
	MEMLIST[15].GWI[0] = 28;
	MEMLIST[15].GWI[1] = 28;
	MEMLIST[15].GWI[2] = 192;
	MEMLIST[15].LWI[0] = 4;
	MEMLIST[15].LWI[1] = 4;
	MEMLIST[15].LWI[2] = 12;


	/* Point Wise Convolution(1x1) + BN (Layer 16)*/
	OutputSize = 28 * 28 * 32;
	WeightSize = 32 * 192 * 1 * 1;
	BNSize = 32 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[16].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[16].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[16].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[16].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[16].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[16].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_D5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[16].KERNEL, 0, sizeof(cl_mem), &MEMLIST[15].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[16].KERNEL, 1, sizeof(cl_mem), &MEMLIST[16].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[16].KERNEL, 2, sizeof(cl_mem), &MEMLIST[16].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[16].KERNEL, 3, sizeof(cl_mem), &MEMLIST[16].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[16].DIM = 3;
	MEMLIST[16].GWI[0] = 28;
	MEMLIST[16].GWI[1] = 28;
	MEMLIST[16].GWI[2] = 32;
	MEMLIST[16].LWI[0] = 4;
	MEMLIST[16].LWI[1] = 4;
	MEMLIST[16].LWI[2] = 16;


	/* Add Layer 13 and Layer 16 (Layer 17)*/
	OutputSize = 28 * 28 * 32;
	MEMLIST[17].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[17].KERNEL = clCreateKernel(PROGRAM, "clAdd_D6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[17].KERNEL, 0, sizeof(cl_mem), &MEMLIST[13].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[17].KERNEL, 1, sizeof(cl_mem), &MEMLIST[16].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[17].KERNEL, 2, sizeof(cl_mem), &MEMLIST[17].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[17].DIM = 3;
	MEMLIST[17].GWI[0] = 28;
	MEMLIST[17].GWI[1] = 28;
	MEMLIST[17].GWI[2] = 32;
	MEMLIST[17].LWI[0] = 4;
	MEMLIST[17].LWI[1] = 4;
	MEMLIST[17].LWI[2] = 16;

	/* 3rd block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 28 * 28 * 192;
	WeightSize = 192 * 32 * 1 * 1;
	BNSize = 192 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[18].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[18].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[18].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[18].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[18].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[18].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_D3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[18].KERNEL, 0, sizeof(cl_mem), &MEMLIST[17].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[18].KERNEL, 1, sizeof(cl_mem), &MEMLIST[18].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[18].KERNEL, 2, sizeof(cl_mem), &MEMLIST[18].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[18].KERNEL, 3, sizeof(cl_mem), &MEMLIST[18].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[18].DIM = 3;
	MEMLIST[18].GWI[0] = 28;
	MEMLIST[18].GWI[1] = 28;
	MEMLIST[18].GWI[2] = 192;
	MEMLIST[18].LWI[0] = 4;
	MEMLIST[18].LWI[1] = 4;
	MEMLIST[18].LWI[2] = 16;


	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 28 * 28 * 192;
	WeightSize = 192 * 3 * 3;
	BNSize = 192 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[19].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[19].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[19].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[19].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[19].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[19].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_D4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[19].KERNEL, 0, sizeof(cl_mem), &MEMLIST[18].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[19].KERNEL, 1, sizeof(cl_mem), &MEMLIST[19].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[19].KERNEL, 2, sizeof(cl_mem), &MEMLIST[19].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[19].KERNEL, 3, sizeof(cl_mem), &MEMLIST[19].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[19].DIM = 3;
	MEMLIST[19].GWI[0] = 28;
	MEMLIST[19].GWI[1] = 28;
	MEMLIST[19].GWI[2] = 192;
	MEMLIST[19].LWI[0] = 4;
	MEMLIST[19].LWI[1] = 4;
	MEMLIST[19].LWI[2] = 12;

	/* Point Wise Convolution(1x1) + BN (Layer 20)*/
	OutputSize = 28 * 28 * 32;
	WeightSize = 32 * 192 * 1 * 1;
	BNSize = 32 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[20].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[20].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[20].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[20].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[20].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[20].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_D5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[20].KERNEL, 0, sizeof(cl_mem), &MEMLIST[19].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[20].KERNEL, 1, sizeof(cl_mem), &MEMLIST[20].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[20].KERNEL, 2, sizeof(cl_mem), &MEMLIST[20].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[20].KERNEL, 3, sizeof(cl_mem), &MEMLIST[20].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[20].DIM = 3;
	MEMLIST[20].GWI[0] = 28;
	MEMLIST[20].GWI[1] = 28;
	MEMLIST[20].GWI[2] = 32;
	MEMLIST[20].LWI[0] = 4;
	MEMLIST[20].LWI[1] = 4;
	MEMLIST[20].LWI[2] = 16;

	/* Add Layer 17 and Layer 20 (Layer 21)*/
	OutputSize = 28 * 28 * 32;
	MEMLIST[21].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[21].KERNEL = clCreateKernel(PROGRAM, "clAdd_D6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[21].KERNEL, 0, sizeof(cl_mem), &MEMLIST[20].OUTPUT);      clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[21].KERNEL, 1, sizeof(cl_mem), &MEMLIST[17].OUTPUT);      clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[21].KERNEL, 2, sizeof(cl_mem), &MEMLIST[21].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[21].DIM = 3;
	MEMLIST[21].GWI[0] = 28;
	MEMLIST[21].GWI[1] = 28;
	MEMLIST[21].GWI[2] = 32;
	MEMLIST[21].LWI[0] = 4;
	MEMLIST[21].LWI[1] = 4;
	MEMLIST[21].LWI[2] = 16;


	/* Group 4 : 4 block, 2 stride */

	/* 1st block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 28 * 28 * 192;
	WeightSize = 192 * 32 * 1 * 1;
	BNSize = 192 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[22].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[22].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[22].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[22].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[22].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[22].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_E0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[22].KERNEL, 0, sizeof(cl_mem), &MEMLIST[21].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[22].KERNEL, 1, sizeof(cl_mem), &MEMLIST[22].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[22].KERNEL, 2, sizeof(cl_mem), &MEMLIST[22].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[22].KERNEL, 3, sizeof(cl_mem), &MEMLIST[22].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[22].DIM = 3;
	MEMLIST[22].GWI[0] = 28;
	MEMLIST[22].GWI[1] = 28;
	MEMLIST[22].GWI[2] = 192;
	MEMLIST[22].LWI[0] = 4;
	MEMLIST[22].LWI[1] = 4;
	MEMLIST[22].LWI[2] = 16;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 192;
	WeightSize = 192 * 3 * 3;
	BNSize = 192 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[23].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[23].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[23].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[23].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[23].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[23].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_E1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[23].KERNEL, 0, sizeof(cl_mem), &MEMLIST[22].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[23].KERNEL, 1, sizeof(cl_mem), &MEMLIST[23].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[23].KERNEL, 2, sizeof(cl_mem), &MEMLIST[23].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[23].KERNEL, 3, sizeof(cl_mem), &MEMLIST[23].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[23].DIM = 3;
	MEMLIST[23].GWI[0] = 14;
	MEMLIST[23].GWI[1] = 14;
	MEMLIST[23].GWI[2] = 192;
	MEMLIST[23].LWI[0] = 14;
	MEMLIST[23].LWI[1] = 14;
	MEMLIST[23].LWI[2] = 1;


	/* Point Wise Convolution(1x1) + BN (Layer 24)*/
	OutputSize = 14 * 14 * 64;
	WeightSize = 64 * 192 * 1 * 1;
	BNSize = 64 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[24].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[24].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[24].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[24].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[24].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[24].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_E2", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[24].KERNEL, 0, sizeof(cl_mem), &MEMLIST[23].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[24].KERNEL, 1, sizeof(cl_mem), &MEMLIST[24].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[24].KERNEL, 2, sizeof(cl_mem), &MEMLIST[24].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[24].KERNEL, 3, sizeof(cl_mem), &MEMLIST[24].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[24].DIM = 3;
	MEMLIST[24].GWI[0] = 14;
	MEMLIST[24].GWI[1] = 14;
	MEMLIST[24].GWI[2] = 64;
	MEMLIST[24].LWI[0] = 14;
	MEMLIST[24].LWI[1] = 14;
	MEMLIST[24].LWI[2] = 1;


	/* 2ed block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 64 * 1 * 1;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[25].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[25].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[25].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[25].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[25].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[25].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_E3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[25].KERNEL, 0, sizeof(cl_mem), &MEMLIST[24].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[25].KERNEL, 1, sizeof(cl_mem), &MEMLIST[25].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[25].KERNEL, 2, sizeof(cl_mem), &MEMLIST[25].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[25].KERNEL, 3, sizeof(cl_mem), &MEMLIST[25].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[25].DIM = 3;
	MEMLIST[25].GWI[0] = 14;
	MEMLIST[25].GWI[1] = 14;
	MEMLIST[25].GWI[2] = 384;
	MEMLIST[25].LWI[0] = 14;
	MEMLIST[25].LWI[1] = 14;
	MEMLIST[25].LWI[2] = 1;

	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 3 * 3;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[26].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[26].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[26].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[26].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[26].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[26].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_E4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[26].KERNEL, 0, sizeof(cl_mem), &MEMLIST[25].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[26].KERNEL, 1, sizeof(cl_mem), &MEMLIST[26].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[26].KERNEL, 2, sizeof(cl_mem), &MEMLIST[26].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[26].KERNEL, 3, sizeof(cl_mem), &MEMLIST[26].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[26].DIM = 3;
	MEMLIST[26].GWI[0] = 14;
	MEMLIST[26].GWI[1] = 14;
	MEMLIST[26].GWI[2] = 384;
	MEMLIST[26].LWI[0] = 14;
	MEMLIST[26].LWI[1] = 14;
	MEMLIST[26].LWI[2] = 1;


	/* Point Wise Convolution(1x1) + BN (Layer 27)*/
	OutputSize = 14 * 14 * 64;
	WeightSize = 64 * 384 * 1 * 1;
	BNSize = 64 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[27].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[27].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[27].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[27].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[27].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[27].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_E5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[27].KERNEL, 0, sizeof(cl_mem), &MEMLIST[26].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[27].KERNEL, 1, sizeof(cl_mem), &MEMLIST[27].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[27].KERNEL, 2, sizeof(cl_mem), &MEMLIST[27].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[27].KERNEL, 3, sizeof(cl_mem), &MEMLIST[27].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[27].DIM = 3;
	MEMLIST[27].GWI[0] = 14;
	MEMLIST[27].GWI[1] = 14;
	MEMLIST[27].GWI[2] = 64;
	MEMLIST[27].LWI[0] = 14;
	MEMLIST[27].LWI[1] = 14;
	MEMLIST[27].LWI[2] = 1;

	/* Add Layer 24 and Layer 27 (Layer 28)*/
	OutputSize = 14 * 14 * 64;
	MEMLIST[28].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[28].KERNEL = clCreateKernel(PROGRAM, "clAdd_E6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[28].KERNEL, 0, sizeof(cl_mem), &MEMLIST[27].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[28].KERNEL, 1, sizeof(cl_mem), &MEMLIST[24].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[28].KERNEL, 2, sizeof(cl_mem), &MEMLIST[28].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[28].DIM = 3;
	MEMLIST[28].GWI[0] = 14;
	MEMLIST[28].GWI[1] = 14;
	MEMLIST[28].GWI[2] = 64;
	MEMLIST[28].LWI[0] = 14;
	MEMLIST[28].LWI[1] = 14;
	MEMLIST[28].LWI[2] = 1;


	/* 3rd block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 64 * 1 * 1;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[29].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[29].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[29].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[29].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[29].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[29].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_E3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[29].KERNEL, 0, sizeof(cl_mem), &MEMLIST[28].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[29].KERNEL, 1, sizeof(cl_mem), &MEMLIST[29].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[29].KERNEL, 2, sizeof(cl_mem), &MEMLIST[29].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[29].KERNEL, 3, sizeof(cl_mem), &MEMLIST[29].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[29].DIM = 3;
	MEMLIST[29].GWI[0] = 14;
	MEMLIST[29].GWI[1] = 14;
	MEMLIST[29].GWI[2] = 384;
	MEMLIST[29].LWI[0] = 14;
	MEMLIST[29].LWI[1] = 14;
	MEMLIST[29].LWI[2] = 1;


	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 3 * 3;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[30].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[30].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[30].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[30].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[30].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[30].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_E4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[30].KERNEL, 0, sizeof(cl_mem), &MEMLIST[29].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[30].KERNEL, 1, sizeof(cl_mem), &MEMLIST[30].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[30].KERNEL, 2, sizeof(cl_mem), &MEMLIST[30].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[30].KERNEL, 3, sizeof(cl_mem), &MEMLIST[30].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[30].DIM = 3;
	MEMLIST[30].GWI[0] = 14;
	MEMLIST[30].GWI[1] = 14;
	MEMLIST[30].GWI[2] = 384;
	MEMLIST[30].LWI[0] = 14;
	MEMLIST[30].LWI[1] = 14;
	MEMLIST[30].LWI[2] = 1;


	/* Point Wise Convolution(1x1) + BN (Layer 31)*/
	OutputSize = 14 * 14 * 64;
	WeightSize = 64 * 384 * 1 * 1;
	BNSize = 64 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[31].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[31].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[31].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[31].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[31].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[31].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_E5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[31].KERNEL, 0, sizeof(cl_mem), &MEMLIST[30].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[31].KERNEL, 1, sizeof(cl_mem), &MEMLIST[31].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[31].KERNEL, 2, sizeof(cl_mem), &MEMLIST[31].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[31].KERNEL, 3, sizeof(cl_mem), &MEMLIST[31].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[31].DIM = 3;
	MEMLIST[31].GWI[0] = 14;
	MEMLIST[31].GWI[1] = 14;
	MEMLIST[31].GWI[2] = 64;
	MEMLIST[31].LWI[0] = 14;
	MEMLIST[31].LWI[1] = 14;
	MEMLIST[31].LWI[2] = 1;

	/* Add Layer 28 and Layer 31 (Layer 32)*/
	OutputSize = 14 * 14 * 64;
	MEMLIST[32].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[32].KERNEL = clCreateKernel(PROGRAM, "clAdd_E6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[32].KERNEL, 0, sizeof(cl_mem), &MEMLIST[31].OUTPUT);      clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[32].KERNEL, 1, sizeof(cl_mem), &MEMLIST[28].OUTPUT);      clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[32].KERNEL, 2, sizeof(cl_mem), &MEMLIST[32].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[32].DIM = 3;
	MEMLIST[32].GWI[0] = 14;
	MEMLIST[32].GWI[1] = 14;
	MEMLIST[32].GWI[2] = 64;
	MEMLIST[32].LWI[0] = 14;
	MEMLIST[32].LWI[1] = 14;
	MEMLIST[32].LWI[2] = 1;

	/* 4th block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 64 * 1 * 1;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[33].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[33].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[33].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[33].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[33].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[33].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_E3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[33].KERNEL, 0, sizeof(cl_mem), &MEMLIST[32].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[33].KERNEL, 1, sizeof(cl_mem), &MEMLIST[33].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[33].KERNEL, 2, sizeof(cl_mem), &MEMLIST[33].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[33].KERNEL, 3, sizeof(cl_mem), &MEMLIST[33].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[33].DIM = 3;
	MEMLIST[33].GWI[0] = 14;
	MEMLIST[33].GWI[1] = 14;
	MEMLIST[33].GWI[2] = 384;
	MEMLIST[33].LWI[0] = 14;
	MEMLIST[33].LWI[1] = 14;
	MEMLIST[33].LWI[2] = 1;


	/* Depth Wise Convolution(3x3, same)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 3 * 3;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[34].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[34].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[34].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[34].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[34].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[34].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_E4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[34].KERNEL, 0, sizeof(cl_mem), &MEMLIST[33].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[34].KERNEL, 1, sizeof(cl_mem), &MEMLIST[34].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[34].KERNEL, 2, sizeof(cl_mem), &MEMLIST[34].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[34].KERNEL, 3, sizeof(cl_mem), &MEMLIST[34].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[34].DIM = 3;
	MEMLIST[34].GWI[0] = 14;
	MEMLIST[34].GWI[1] = 14;
	MEMLIST[34].GWI[2] = 384;
	MEMLIST[34].LWI[0] = 14;
	MEMLIST[34].LWI[1] = 14;
	MEMLIST[34].LWI[2] = 1;

	/* Point Wise Convolution(1x1) + BN (Layer 35)*/
	OutputSize = 14 * 14 * 64;
	WeightSize = 64 * 384 * 1 * 1;
	BNSize = 64 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[35].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[35].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[35].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[35].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[35].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[35].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_E5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[35].KERNEL, 0, sizeof(cl_mem), &MEMLIST[34].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[35].KERNEL, 1, sizeof(cl_mem), &MEMLIST[35].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[35].KERNEL, 2, sizeof(cl_mem), &MEMLIST[35].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[35].KERNEL, 3, sizeof(cl_mem), &MEMLIST[35].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[35].DIM = 3;
	MEMLIST[35].GWI[0] = 14;
	MEMLIST[35].GWI[1] = 14;
	MEMLIST[35].GWI[2] = 64;
	MEMLIST[35].LWI[0] = 14;
	MEMLIST[35].LWI[1] = 14;
	MEMLIST[35].LWI[2] = 1;


	/* Add Layer 32 and Layer 35 (Layer 36)*/
	OutputSize = 14 * 14 * 64;
	MEMLIST[36].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[36].KERNEL = clCreateKernel(PROGRAM, "clAdd_E6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[36].KERNEL, 0, sizeof(cl_mem), &MEMLIST[35].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[36].KERNEL, 1, sizeof(cl_mem), &MEMLIST[32].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[36].KERNEL, 2, sizeof(cl_mem), &MEMLIST[36].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[36].DIM = 3;
	MEMLIST[36].GWI[0] = 14;
	MEMLIST[36].GWI[1] = 14;
	MEMLIST[36].GWI[2] = 64;
	MEMLIST[36].LWI[0] = 14;
	MEMLIST[36].LWI[1] = 14;
	MEMLIST[36].LWI[2] = 1;


	/* Group 5 : 3 block, 1 stride */

    /* 1st block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 64 * 1 * 1;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[37].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[37].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[37].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[37].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[37].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[37].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_F0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[37].KERNEL, 0, sizeof(cl_mem), &MEMLIST[36].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[37].KERNEL, 1, sizeof(cl_mem), &MEMLIST[37].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[37].KERNEL, 2, sizeof(cl_mem), &MEMLIST[37].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[37].KERNEL, 3, sizeof(cl_mem), &MEMLIST[37].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[37].DIM = 3;
	MEMLIST[37].GWI[0] = 14;
	MEMLIST[37].GWI[1] = 14;
	MEMLIST[37].GWI[2] = 384;
	MEMLIST[37].LWI[0] = 14;
	MEMLIST[37].LWI[1] = 14;
	MEMLIST[37].LWI[2] = 1;


	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 384;
	WeightSize = 384 * 3 * 3;
	BNSize = 384 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[38].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[38].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[38].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[38].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[38].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[38].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_F1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[38].KERNEL, 0, sizeof(cl_mem), &MEMLIST[37].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[38].KERNEL, 1, sizeof(cl_mem), &MEMLIST[38].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[38].KERNEL, 2, sizeof(cl_mem), &MEMLIST[38].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[38].KERNEL, 3, sizeof(cl_mem), &MEMLIST[38].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[38].DIM = 3;
	MEMLIST[38].GWI[0] = 14;
	MEMLIST[38].GWI[1] = 14;
	MEMLIST[38].GWI[2] = 384;
	MEMLIST[38].LWI[0] = 14;
	MEMLIST[38].LWI[1] = 14;
	MEMLIST[38].LWI[2] = 1;

	/* Point Wise Convolution(1x1) + BN (Layer 39)*/
	OutputSize = 14 * 14 * 96;
	WeightSize = 96 * 384 * 1 * 1;
	BNSize = 96 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[39].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[39].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[39].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[39].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[39].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[39].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_F2", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[39].KERNEL, 0, sizeof(cl_mem), &MEMLIST[38].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[39].KERNEL, 1, sizeof(cl_mem), &MEMLIST[39].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[39].KERNEL, 2, sizeof(cl_mem), &MEMLIST[39].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[39].KERNEL, 3, sizeof(cl_mem), &MEMLIST[39].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[39].DIM = 3;
	MEMLIST[39].GWI[0] = 14;
	MEMLIST[39].GWI[1] = 14;
	MEMLIST[39].GWI[2] = 96;
	MEMLIST[39].LWI[0] = 14;
	MEMLIST[39].LWI[1] = 14;
	MEMLIST[39].LWI[2] = 1;


	/* 2ed block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 576;
	WeightSize = 576 * 96 * 1 * 1;
	BNSize = 576 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[40].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[40].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[40].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[40].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[40].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[40].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_F3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[40].KERNEL, 0, sizeof(cl_mem), &MEMLIST[39].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[40].KERNEL, 1, sizeof(cl_mem), &MEMLIST[40].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[40].KERNEL, 2, sizeof(cl_mem), &MEMLIST[40].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[40].KERNEL, 3, sizeof(cl_mem), &MEMLIST[40].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[40].DIM = 3;
	MEMLIST[40].GWI[0] = 14;
	MEMLIST[40].GWI[1] = 14;
	MEMLIST[40].GWI[2] = 576;
	MEMLIST[40].LWI[0] = 14;
	MEMLIST[40].LWI[1] = 14;
	MEMLIST[40].LWI[2] = 1;


	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 576;
	WeightSize = 576 * 3 * 3;
	BNSize = 576 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[41].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[41].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[41].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[41].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[41].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[41].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_F4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[41].KERNEL, 0, sizeof(cl_mem), &MEMLIST[40].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[41].KERNEL, 1, sizeof(cl_mem), &MEMLIST[41].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[41].KERNEL, 2, sizeof(cl_mem), &MEMLIST[41].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[41].KERNEL, 3, sizeof(cl_mem), &MEMLIST[41].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[41].DIM = 3;
	MEMLIST[41].GWI[0] = 14;
	MEMLIST[41].GWI[1] = 14;
	MEMLIST[41].GWI[2] = 576;
	MEMLIST[41].LWI[0] = 14;
	MEMLIST[41].LWI[1] = 14;
	MEMLIST[41].LWI[2] = 1;


	/* Point Wise Convolution(1x1) + BN (Layer 42)*/
	OutputSize = 14 * 14 * 96;
	WeightSize = 96 * 576 * 1 * 1;
	BNSize = 96 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[42].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[42].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[42].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[42].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[42].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[42].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_F5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[42].KERNEL, 0, sizeof(cl_mem), &MEMLIST[41].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[42].KERNEL, 1, sizeof(cl_mem), &MEMLIST[42].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[42].KERNEL, 2, sizeof(cl_mem), &MEMLIST[42].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[42].KERNEL, 3, sizeof(cl_mem), &MEMLIST[42].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[42].DIM = 3;
	MEMLIST[42].GWI[0] = 14;
	MEMLIST[42].GWI[1] = 14;
	MEMLIST[42].GWI[2] = 96;
	MEMLIST[42].LWI[0] = 14;
	MEMLIST[42].LWI[1] = 14;
	MEMLIST[42].LWI[2] = 1;


	/* Add Layer 42 and Layer 39 (Layer 43)*/
	OutputSize = 14 * 14 * 96;
	MEMLIST[43].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[43].KERNEL = clCreateKernel(PROGRAM, "clAdd_F6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[43].KERNEL, 0, sizeof(cl_mem), &MEMLIST[42].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[43].KERNEL, 1, sizeof(cl_mem), &MEMLIST[39].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[43].KERNEL, 2, sizeof(cl_mem), &MEMLIST[43].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[43].DIM = 3;
	MEMLIST[43].GWI[0] = 14;
	MEMLIST[43].GWI[1] = 14;
	MEMLIST[43].GWI[2] = 96;
	MEMLIST[43].LWI[0] = 14;
	MEMLIST[43].LWI[1] = 14;
	MEMLIST[43].LWI[2] = 1;

	/* 3rd block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 576;
	WeightSize = 576 * 96 * 1 * 1;
	BNSize = 576 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[44].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[44].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[44].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[44].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[44].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[44].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_F3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[44].KERNEL, 0, sizeof(cl_mem), &MEMLIST[43].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[44].KERNEL, 1, sizeof(cl_mem), &MEMLIST[44].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[44].KERNEL, 2, sizeof(cl_mem), &MEMLIST[44].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[44].KERNEL, 3, sizeof(cl_mem), &MEMLIST[44].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[44].DIM = 3;
	MEMLIST[44].GWI[0] = 14;
	MEMLIST[44].GWI[1] = 14;
	MEMLIST[44].GWI[2] = 576;
	MEMLIST[44].LWI[0] = 14;
	MEMLIST[44].LWI[1] = 14;
	MEMLIST[44].LWI[2] = 1;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 14 * 14 * 576;
	WeightSize = 576 * 3 * 3;
	BNSize = 576 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[45].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[45].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[45].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[45].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[45].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[45].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_F4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[45].KERNEL, 0, sizeof(cl_mem), &MEMLIST[44].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[45].KERNEL, 1, sizeof(cl_mem), &MEMLIST[45].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[45].KERNEL, 2, sizeof(cl_mem), &MEMLIST[45].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[45].KERNEL, 3, sizeof(cl_mem), &MEMLIST[45].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[45].DIM = 3;
	MEMLIST[45].GWI[0] = 14;
	MEMLIST[45].GWI[1] = 14;
	MEMLIST[45].GWI[2] = 576;
	MEMLIST[45].LWI[0] = 14;
	MEMLIST[45].LWI[1] = 14;
	MEMLIST[45].LWI[2] = 1;

	/* Point Wise Convolution(1x1) + BN (Layer 46)*/
	OutputSize = 14 * 14 * 96;
	WeightSize = 96 * 576 * 1 * 1;
	BNSize = 96 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[46].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[46].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[46].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[46].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[46].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[46].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_F5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[46].KERNEL, 0, sizeof(cl_mem), &MEMLIST[45].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[46].KERNEL, 1, sizeof(cl_mem), &MEMLIST[46].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[46].KERNEL, 2, sizeof(cl_mem), &MEMLIST[46].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[46].KERNEL, 3, sizeof(cl_mem), &MEMLIST[46].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[46].DIM = 3;
	MEMLIST[46].GWI[0] = 14;
	MEMLIST[46].GWI[1] = 14;
	MEMLIST[46].GWI[2] = 96;
	MEMLIST[46].LWI[0] = 14;
	MEMLIST[46].LWI[1] = 14;
	MEMLIST[46].LWI[2] = 1;

	/* Add Layer 46 and Layer 43 (Layer 47)*/
	OutputSize = 14 * 14 * 96;
	MEMLIST[47].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[47].KERNEL = clCreateKernel(PROGRAM, "clAdd_F6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[47].KERNEL, 0, sizeof(cl_mem), &MEMLIST[46].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[47].KERNEL, 1, sizeof(cl_mem), &MEMLIST[43].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[47].KERNEL, 2, sizeof(cl_mem), &MEMLIST[47].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[47].DIM = 3;
	MEMLIST[47].GWI[0] = 14;
	MEMLIST[47].GWI[1] = 14;
	MEMLIST[47].GWI[2] = 96;
	MEMLIST[47].LWI[0] = 14;
	MEMLIST[47].LWI[1] = 14;
	MEMLIST[47].LWI[2] = 1;





	/* Group 6 : 3 block, 2 stride */

	/* 1st block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 14 * 14 * 576;
	WeightSize = 576 * 96 * 1 * 1;
	BNSize = 576 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[48].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[48].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[48].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[48].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[48].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[48].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_G0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[48].KERNEL, 0, sizeof(cl_mem), &MEMLIST[47].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[48].KERNEL, 1, sizeof(cl_mem), &MEMLIST[48].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[48].KERNEL, 2, sizeof(cl_mem), &MEMLIST[48].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[48].KERNEL, 3, sizeof(cl_mem), &MEMLIST[48].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[48].DIM = 3;
	MEMLIST[48].GWI[0] = 14;
	MEMLIST[48].GWI[1] = 14;
	MEMLIST[48].GWI[2] = 576;
	MEMLIST[48].LWI[0] = 14;
	MEMLIST[48].LWI[1] = 14;
	MEMLIST[48].LWI[2] = 1;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 7 * 7 * 576;
	WeightSize = 576 * 3 * 3;
	BNSize = 576 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[49].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[49].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[49].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[49].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[49].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[49].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_G1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[49].KERNEL, 0, sizeof(cl_mem), &MEMLIST[48].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[49].KERNEL, 1, sizeof(cl_mem), &MEMLIST[49].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[49].KERNEL, 2, sizeof(cl_mem), &MEMLIST[49].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[49].KERNEL, 3, sizeof(cl_mem), &MEMLIST[49].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[49].DIM = 3;
	MEMLIST[49].GWI[0] = 7;
	MEMLIST[49].GWI[1] = 7;
	MEMLIST[49].GWI[2] = 576;
	MEMLIST[49].LWI[0] = 7;
	MEMLIST[49].LWI[1] = 7;
	MEMLIST[49].LWI[2] = 4;

	/* Point Wise Convolution(1x1) + BN (Layer 50)*/
	OutputSize = 7 * 7 * 160;
	WeightSize = 160 * 576 * 1 * 1;
	BNSize = 160 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[50].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[50].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[50].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[50].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[50].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[50].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_G2", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[50].KERNEL, 0, sizeof(cl_mem), &MEMLIST[49].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[50].KERNEL, 1, sizeof(cl_mem), &MEMLIST[50].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[50].KERNEL, 2, sizeof(cl_mem), &MEMLIST[50].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[50].KERNEL, 3, sizeof(cl_mem), &MEMLIST[50].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[50].DIM = 3;
	MEMLIST[50].GWI[0] = 7;
	MEMLIST[50].GWI[1] = 7;
	MEMLIST[50].GWI[2] = 160;
	MEMLIST[50].LWI[0] = 7;
	MEMLIST[50].LWI[1] = 7;
	MEMLIST[50].LWI[2] = 5;


	/* 2ed block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 7 * 7 * 960;
	WeightSize = 960 * 160 * 1 * 1;
	BNSize = 960 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[51].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[51].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[51].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[51].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[51].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[51].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_G3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[51].KERNEL, 0, sizeof(cl_mem), &MEMLIST[50].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[51].KERNEL, 1, sizeof(cl_mem), &MEMLIST[51].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[51].KERNEL, 2, sizeof(cl_mem), &MEMLIST[51].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[51].KERNEL, 3, sizeof(cl_mem), &MEMLIST[51].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[51].DIM = 3;
	MEMLIST[51].GWI[0] = 7;
	MEMLIST[51].GWI[1] = 7;
	MEMLIST[51].GWI[2] = 960;
	MEMLIST[51].LWI[0] = 7;
	MEMLIST[51].LWI[1] = 7;
	MEMLIST[51].LWI[2] = 5;
	

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 7 * 7 * 960;
	WeightSize = 960 * 3 * 3;
	BNSize = 960 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[52].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[52].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[52].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[52].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[52].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[52].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_G4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[52].KERNEL, 0, sizeof(cl_mem), &MEMLIST[51].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[52].KERNEL, 1, sizeof(cl_mem), &MEMLIST[52].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[52].KERNEL, 2, sizeof(cl_mem), &MEMLIST[52].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[52].KERNEL, 3, sizeof(cl_mem), &MEMLIST[52].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[52].DIM = 3;
	MEMLIST[52].GWI[0] = 7;
	MEMLIST[52].GWI[1] = 7;
	MEMLIST[52].GWI[2] = 960;
	MEMLIST[52].LWI[0] = 7;
	MEMLIST[52].LWI[1] = 7;
	MEMLIST[52].LWI[2] = 5;

	/* Point Wise Convolution(1x1) + BN (Layer 53)*/
	OutputSize = 7 * 7 * 160;
	WeightSize = 160 * 960 * 1 * 1;
	BNSize = 160 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[53].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[53].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[53].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[53].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[53].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[53].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_G5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[53].KERNEL, 0, sizeof(cl_mem), &MEMLIST[52].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[53].KERNEL, 1, sizeof(cl_mem), &MEMLIST[53].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[53].KERNEL, 2, sizeof(cl_mem), &MEMLIST[53].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[53].KERNEL, 3, sizeof(cl_mem), &MEMLIST[53].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[53].DIM = 3;
	MEMLIST[53].GWI[0] = 7;
	MEMLIST[53].GWI[1] = 7;
	MEMLIST[53].GWI[2] = 160;
	MEMLIST[53].LWI[0] = 7;
	MEMLIST[53].LWI[1] = 7;
	MEMLIST[53].LWI[2] = 5;

	/* Add Layer 53 and Layer 50 (Layer 54)*/
	OutputSize = 7 * 7 * 160;
	MEMLIST[54].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[54].KERNEL = clCreateKernel(PROGRAM, "clAdd_G6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[54].KERNEL, 0, sizeof(cl_mem), &MEMLIST[53].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[54].KERNEL, 1, sizeof(cl_mem), &MEMLIST[50].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[54].KERNEL, 2, sizeof(cl_mem), &MEMLIST[54].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[54].DIM = 3;
	MEMLIST[54].GWI[0] = 7;
	MEMLIST[54].GWI[1] = 7;
	MEMLIST[54].GWI[2] = 160;
	MEMLIST[54].LWI[0] = 7;
	MEMLIST[54].LWI[1] = 7;
	MEMLIST[54].LWI[2] = 5;

	/* 3rd block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 7 * 7 * 960;
	WeightSize = 960 * 160 * 1 * 1;
	BNSize = 960 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[55].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[55].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[55].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[55].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[55].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[55].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_G3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[55].KERNEL, 0, sizeof(cl_mem), &MEMLIST[54].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[55].KERNEL, 1, sizeof(cl_mem), &MEMLIST[55].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[55].KERNEL, 2, sizeof(cl_mem), &MEMLIST[55].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[55].KERNEL, 3, sizeof(cl_mem), &MEMLIST[55].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[55].DIM = 3;
	MEMLIST[55].GWI[0] = 7;
	MEMLIST[55].GWI[1] = 7;
	MEMLIST[55].GWI[2] = 960;
	MEMLIST[55].LWI[0] = 7;
	MEMLIST[55].LWI[1] = 7;
	MEMLIST[55].LWI[2] = 5;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 7 * 7 * 960;
	WeightSize = 960 * 3 * 3;
	BNSize = 960 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[56].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[56].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[56].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[56].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[56].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[56].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_G4", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[56].KERNEL, 0, sizeof(cl_mem), &MEMLIST[55].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[56].KERNEL, 1, sizeof(cl_mem), &MEMLIST[56].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[56].KERNEL, 2, sizeof(cl_mem), &MEMLIST[56].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[56].KERNEL, 3, sizeof(cl_mem), &MEMLIST[56].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[56].DIM = 3;
	MEMLIST[56].GWI[0] = 7;
	MEMLIST[56].GWI[1] = 7;
	MEMLIST[56].GWI[2] = 960;
	MEMLIST[56].LWI[0] = 7;
	MEMLIST[56].LWI[1] = 7;
	MEMLIST[56].LWI[2] = 5;

	/* Point Wise Convolution(1x1) + BN (Layer 57)*/
	OutputSize = 7 * 7 * 160;
	WeightSize = 160 * 960 * 1 * 1;
	BNSize = 160 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[57].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[57].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[57].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[57].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[57].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[57].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_G5", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[57].KERNEL, 0, sizeof(cl_mem), &MEMLIST[56].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[57].KERNEL, 1, sizeof(cl_mem), &MEMLIST[57].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[57].KERNEL, 2, sizeof(cl_mem), &MEMLIST[57].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[57].KERNEL, 3, sizeof(cl_mem), &MEMLIST[57].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[57].DIM = 3;
	MEMLIST[57].GWI[0] = 7;
	MEMLIST[57].GWI[1] = 7;
	MEMLIST[57].GWI[2] = 160;
	MEMLIST[57].LWI[0] = 7;
	MEMLIST[57].LWI[1] = 7;
	MEMLIST[57].LWI[2] = 5;

	/* Add Layer 57 and Layer 54 (Layer 58)*/
	OutputSize = 7 * 7 * 160;
	MEMLIST[58].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[58].KERNEL = clCreateKernel(PROGRAM, "clAdd_G6", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[58].KERNEL, 0, sizeof(cl_mem), &MEMLIST[57].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[58].KERNEL, 1, sizeof(cl_mem), &MEMLIST[54].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[58].KERNEL, 2, sizeof(cl_mem), &MEMLIST[58].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[58].DIM = 3;
	MEMLIST[58].GWI[0] = 7;
	MEMLIST[58].GWI[1] = 7;
	MEMLIST[58].GWI[2] = 160;
	MEMLIST[58].LWI[0] = 7;
	MEMLIST[58].LWI[1] = 7;
	MEMLIST[58].LWI[2] = 5;




	/* Group 7 : 1 block, 1 stride */

	/* 1st block */
	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 7 * 7 * 960;
	WeightSize = 960 * 160 * 1 * 1;
	BNSize = 960 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[59].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[59].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[59].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[59].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[59].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[59].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_H0", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[59].KERNEL, 0, sizeof(cl_mem), &MEMLIST[58].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[59].KERNEL, 1, sizeof(cl_mem), &MEMLIST[59].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[59].KERNEL, 2, sizeof(cl_mem), &MEMLIST[59].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[59].KERNEL, 3, sizeof(cl_mem), &MEMLIST[59].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[59].DIM = 3;
	MEMLIST[59].GWI[0] = 7;
	MEMLIST[59].GWI[1] = 7;
	MEMLIST[59].GWI[2] = 960;
	MEMLIST[59].LWI[0] = 7;
	MEMLIST[59].LWI[1] = 7;
	MEMLIST[59].LWI[2] = 5;

	/* Depth Wise Convolution(3x3, /2)  + BN + ReLU6 */
	OutputSize = 7 * 7 * 960;
	WeightSize = 960 * 3 * 3;
	BNSize = 960 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[60].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[60].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[60].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[60].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[60].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[60].KERNEL = clCreateKernel(PROGRAM, "clDWConvBNReLU_H1", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[60].KERNEL, 0, sizeof(cl_mem), &MEMLIST[59].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[60].KERNEL, 1, sizeof(cl_mem), &MEMLIST[60].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[60].KERNEL, 2, sizeof(cl_mem), &MEMLIST[60].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[60].KERNEL, 3, sizeof(cl_mem), &MEMLIST[60].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[60].DIM = 3;
	MEMLIST[60].GWI[0] = 7;
	MEMLIST[60].GWI[1] = 7;
	MEMLIST[60].GWI[2] = 960;
	MEMLIST[60].LWI[0] = 7;
	MEMLIST[60].LWI[1] = 7;
	MEMLIST[60].LWI[2] = 5;

	/* Point Wise Convolution(1x1) + BN (Layer 61)*/
	OutputSize = 7 * 7 * 320;
	WeightSize = 320 * 960 * 1 * 1;
	BNSize = 320 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[61].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[61].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[61].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[61].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[61].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[61].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNLinear_H2", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[61].KERNEL, 0, sizeof(cl_mem), &MEMLIST[60].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[61].KERNEL, 1, sizeof(cl_mem), &MEMLIST[61].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[61].KERNEL, 2, sizeof(cl_mem), &MEMLIST[61].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[61].KERNEL, 3, sizeof(cl_mem), &MEMLIST[61].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[61].DIM = 3;
	MEMLIST[61].GWI[0] = 7;
	MEMLIST[61].GWI[1] = 7;
	MEMLIST[61].GWI[2] = 320;
	MEMLIST[61].LWI[0] = 7;
	MEMLIST[61].LWI[1] = 7;
	MEMLIST[61].LWI[2] = 5;

	/* Point Wise Convolution(1x1) + BN + ReLU6 */
	OutputSize = 7 * 7 * 1280;
	WeightSize = 1280 * 320 * 1 * 1;
	BNSize = 1280 * 4;
	WeightTemp = new float[WeightSize];
	BNTemp = new float[BNSize];
	MEMLIST[62].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[62].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[62].BN = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BNSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[62].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < BNSize; n++)
	{
		fread(&BNTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[62].BN, CL_TRUE, 0, FLOAT * BNSize, BNTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BNTemp;
	MEMLIST[62].KERNEL = clCreateKernel(PROGRAM, "clPWConvBNReLU_H3", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[62].KERNEL, 0, sizeof(cl_mem), &MEMLIST[61].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[62].KERNEL, 1, sizeof(cl_mem), &MEMLIST[62].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[62].KERNEL, 2, sizeof(cl_mem), &MEMLIST[62].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[62].KERNEL, 3, sizeof(cl_mem), &MEMLIST[62].BN);      clErrorCheck(RET, __LINE__, true);
	MEMLIST[62].DIM = 3;
	MEMLIST[62].GWI[0] = 7;
	MEMLIST[62].GWI[1] = 7;
	MEMLIST[62].GWI[2] = 1280;
	MEMLIST[62].LWI[0] = 7;
	MEMLIST[62].LWI[1] = 7;
	MEMLIST[62].LWI[2] = 5;

	/* Global average pooling */
	OutputSize = 1 * 1 * 1280;
	MEMLIST[63].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[63].KERNEL = clCreateKernel(PROGRAM, "clGlobalAveragePool", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[63].KERNEL, 0, sizeof(cl_mem), &MEMLIST[62].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[63].KERNEL, 1, sizeof(cl_mem), &MEMLIST[63].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[63].DIM = 1;
	MEMLIST[63].GWI[0] = 1280;
	MEMLIST[63].GWI[1] = 1;
	MEMLIST[63].GWI[2] = 1;
	MEMLIST[63].LWI[0] = 256;
	MEMLIST[63].LWI[1] = 1;
	MEMLIST[63].LWI[2] = 1;


	/* Dense */
	OutputSize = 1 * 1 * 2;
	WeightSize = 1280 * 2;
	BiasSize = 2;
	BiasTemp = new float[BiasSize];
	WeightTemp = new float[WeightSize];
	MEMLIST[64].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[64].BIAS = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * BiasSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[64].WEIGHT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * WeightSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	for (int n = 0; n < BiasSize; n++)
	{
		fread(&BiasTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[64].BIAS, CL_TRUE, 0, FLOAT * BiasSize, BiasTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	for (int n = 0; n < WeightSize; n++)
	{
		fread(&WeightTemp[n], FLOAT, 1, fp);
		ReadCounter--;
	}
	RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[64].WEIGHT, CL_TRUE, 0, FLOAT * WeightSize, WeightTemp, 0, NULL, NULL);
	clFinish(QUEUE);
	delete[] WeightTemp;
	delete[] BiasTemp;
	MEMLIST[64].KERNEL = clCreateKernel(PROGRAM, "clDense", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[64].KERNEL, 0, sizeof(cl_mem), &MEMLIST[63].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[64].KERNEL, 1, sizeof(cl_mem), &MEMLIST[64].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[64].KERNEL, 2, sizeof(cl_mem), &MEMLIST[64].WEIGHT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[64].KERNEL, 3, sizeof(cl_mem), &MEMLIST[64].BIAS);    clErrorCheck(RET, __LINE__, true);
	MEMLIST[64].DIM = 1;
	MEMLIST[64].GWI[0] = 2;
	MEMLIST[64].GWI[1] = 1;
	MEMLIST[64].GWI[2] = 1;
	MEMLIST[64].LWI[0] = 2;
	MEMLIST[64].LWI[1] = 1;
	MEMLIST[64].LWI[2] = 1;

	/* Softmax */
	OutputSize = 2;
	MEMLIST[65].OUTPUT = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE, FLOAT * OutputSize, 0, &RET);
	clErrorCheck(RET, __LINE__, true);
	MEMLIST[65].KERNEL = clCreateKernel(PROGRAM, "clSoftmax", &RET);
	clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[65].KERNEL, 0, sizeof(cl_mem), &MEMLIST[64].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	RET = clSetKernelArg(MEMLIST[65].KERNEL, 1, sizeof(cl_mem), &MEMLIST[65].OUTPUT);  clErrorCheck(RET, __LINE__, true);
	MEMLIST[65].DIM = 1;
	MEMLIST[65].GWI[0] = 2;
	MEMLIST[65].GWI[1] = 1;
	MEMLIST[65].GWI[2] = 1;
	MEMLIST[65].LWI[0] = 2;
	MEMLIST[65].LWI[1] = 1;
	MEMLIST[65].LWI[2] = 1;
	fclose(fp);
	if (ReadCounter != 0)
	{
		printf("Parameter load in not complete.\n\n");
		assert(0);
	}
	else
	{
		printf("MobileNet V2 completed.\n\n");
	}
	printf("------------------------------------------------------------------------\n\n");
}

void mobilenetv2::clInference()
{
	cl_int RET;
	EVs = new cl_event[67];

	for (int idx = 0; idx < 66; idx++)
	{
		if (idx == 0)
		{
			RET = clEnqueueWriteBuffer(QUEUE, MEMLIST[idx].OUTPUT, CL_TRUE, 0, FLOAT * 224 * 224 * 3, Image, 0, NULL, &EVs[idx]);
			clErrorCheck(RET, __LINE__, true);
			clFinish(QUEUE);
		}
		else
		{
			RET = clEnqueueNDRangeKernel(QUEUE, MEMLIST[idx].KERNEL, MEMLIST[idx].DIM, 0, MEMLIST[idx].GWI, MEMLIST[idx].LWI, 0, NULL, &EVs[idx]);
			if (RET != CL_SUCCESS) printf("Trouble at %d and RET = %d\n", idx, RET);
			clErrorCheck(RET, __LINE__, true);
			clFinish(QUEUE);
		}
	}
	RET = clEnqueueReadBuffer(QUEUE, MEMLIST[65].OUTPUT, CL_TRUE, 0, FLOAT * Classes, Result, 0, NULL, &EVs[66]);
	clFinish(QUEUE);
}

void mobilenetv2::clShowResult()
{
	for (int idx = 0; idx < Classes; idx++)
	{		
		if ((idx+1) % 5 == 0) printf("Number %3d : %5.6f %%\n", idx, Result[idx] * 100);
		else printf("Number %3d : %5.6f %%  ", idx, Result[idx] * 100);
	}
	printf("\n");
	printf("------------------------------------------------------------------------\n\n");
}

void mobilenetv2::clShowTimeProfile()
{
	float sum = 0;
	cl_ulong startTime, endTime, kernelExecTimeNs;
	for (int idx = 0; idx < 67; idx++)
	{
		clGetEventProfilingInfo(EVs[idx], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
		clGetEventProfilingInfo(EVs[idx], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
		kernelExecTimeNs = endTime - startTime;
		sum += (float)kernelExecTimeNs;

		printf("Layer %2d - [%6.3f]\n", idx, (float)kernelExecTimeNs * 1e-6);
	}
	printf("========================================================================\n");
	printf("Inference Execution Time : \t\t\t     [%8.3f ms]\n", sum * 1e-6);


}

void mobilenetv2::clLayerData(int Layer, int Size)
{
	cl_int RET;
	float* data = new float[Size];
	RET = clEnqueueReadBuffer(QUEUE, MEMLIST[Layer].WEIGHT, CL_TRUE, 0, FLOAT * Size, data, 0, NULL, NULL);

	for (int i = 0; i < (int)(Size*0.01); i++)
	{
		printf("%5.3f  ",data[i]);
		if (i % 32 == 0) printf("\n");
	}
	/* Debug use */

}


void mobilenetv2::clErrorCheck(cl_int RET, int Line, bool useInterrupt)
{
	if (RET)
		printf("Line %d - Error ", Line - 1);
	switch (RET) {
	case CL_DEVICE_NOT_FOUND:
		printf("%3d : Device not found.\n", RET);
		break;
	case CL_DEVICE_NOT_AVAILABLE:
		printf("%3d : Device not available.\n", RET);
		break;
	case CL_COMPILER_NOT_AVAILABLE:
		printf("%3d : Compiler not available.\n", RET);
		break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		printf("%3d : Memory object allocation failed.\n", RET);
		break;
	case CL_OUT_OF_RESOURCES:
		printf("%3d : Out of resource.\n", RET);
		break;
	case CL_OUT_OF_HOST_MEMORY:
		printf("%3d : Out of host memory.\n", RET);
		break;
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		printf("%3d : Profiling information not available.\n", RET);
		break;
	case CL_MEM_COPY_OVERLAP:
		printf("%3d : Memory copy overlap.\n", RET);
		break;
	case CL_IMAGE_FORMAT_MISMATCH:
		printf("%3d : Image format mismatch.\n", RET);
		break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		printf("%3d : Image format not supported.\n", RET);
		break;
	case CL_BUILD_PROGRAM_FAILURE:
		printf("%3d : Build program failed.\n", RET);
		break;
	case CL_MAP_FAILURE:
		printf("%3d : Maping failed.\n", RET);
		break;
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		printf("%3d : Misaligned sub-buffer offset.\n", RET);
		break;
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		printf("%3d : Execution status error for events in wait list.\n", RET);
		break;
	case CL_COMPILE_PROGRAM_FAILURE:
		printf("%3d : Compile program failed.\n", RET);
		break;
	case CL_LINKER_NOT_AVAILABLE:
		printf("%3d : Linker not available.\n", RET);
		break;
	case CL_LINK_PROGRAM_FAILURE:
		printf("%3d : Link program failed.\n", RET);
		break;
	case CL_DEVICE_PARTITION_FAILED:
		printf("%3d : Device partition failed.\n", RET);
		break;
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
		printf("%3d : Kernel argument information not available.\n", RET);
		break;
	case CL_INVALID_VALUE:
		printf("%3d : Invalid value.\n", RET);
		break;
	case CL_INVALID_DEVICE_TYPE:
		printf("%3d : Invalid device type.\n", RET);
		break;
	case CL_INVALID_PLATFORM:
		printf("%3d : Invalid platform.\n", RET);
		break;
	case CL_INVALID_DEVICE:
		printf("%3d : Invalid device.\n", RET);
		break;
	case CL_INVALID_CONTEXT:
		printf("%3d : Invalid context.\n", RET);
		break;
	case CL_INVALID_QUEUE_PROPERTIES:
		printf("%3d : Invalid queue properties.\n", RET);
		break;
	case CL_INVALID_COMMAND_QUEUE:
		printf("%3d : Invalid command queue.\n", RET);
		break;
	case CL_INVALID_HOST_PTR:
		printf("%3d : Invalid host pointer.\n", RET);
		break;
	case CL_INVALID_MEM_OBJECT:
		printf("%3d : Invalid memory object.\n", RET);
		break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		printf("%3d : Invalid image format descriptor.\n", RET);
		break;
	case CL_INVALID_IMAGE_SIZE:
		printf("%3d : Invalid image size.\n", RET);
		break;
	case CL_INVALID_SAMPLER:
		printf("%3d : Invalid sampler.\n", RET);
		break;
	case CL_INVALID_BINARY:
		printf("%3d : Invalid binary.\n", RET);
		break;
	case CL_INVALID_BUILD_OPTIONS:
		printf("%3d : Invalid build options.\n", RET);
		break;
	case CL_INVALID_PROGRAM:
		printf("%3d : Invalid program.\n", RET);
		break;
	case CL_INVALID_PROGRAM_EXECUTABLE:
		printf("%3d : Invalid program executable.\n", RET);
		break;
	case CL_INVALID_KERNEL_NAME:
		printf("%3d : Invalid kernel name.\n", RET);
		break;
	case CL_INVALID_KERNEL_DEFINITION:
		printf("%3d : Invalid kernel definition.\n", RET);
		break;
	case CL_INVALID_KERNEL:
		printf("%3d : Invalid kernel.\n", RET);
		break;
	case CL_INVALID_ARG_INDEX:
		printf("%3d : Invalid argument index.\n", RET);
		break;
	case CL_INVALID_ARG_VALUE:
		printf("%3d : Invalid argument value.\n", RET);
		break;
	case CL_INVALID_ARG_SIZE:
		printf("%3d : Invalid argument size.\n", RET);
		break;
	case CL_INVALID_KERNEL_ARGS:
		printf("%3d : Invalid kernel arguments.\n", RET);
		break;
	case CL_INVALID_WORK_DIMENSION:
		printf("%3d : Invalid work dimension.\n", RET);
		break;
	case CL_INVALID_WORK_GROUP_SIZE:
		printf("%3d : Invalid work group size.\n", RET);
		break;
	case CL_INVALID_WORK_ITEM_SIZE:
		printf("%3d : Invalid work time size.\n", RET);
		break;
	case CL_INVALID_GLOBAL_OFFSET:
		printf("%3d : Invalid global offset.\n", RET);
		break;
	case CL_INVALID_EVENT_WAIT_LIST:
		printf("%3d : Invalid event wait list.\n", RET);
		break;
	case CL_INVALID_EVENT:
		printf("%3d : Invalid event.\n", RET);
		break;
	case CL_INVALID_OPERATION:
		printf("%3d : Invalid operation.\n", RET);
		break;
	case CL_INVALID_GL_OBJECT:
		printf("%3d : Invalid GL object.\n", RET);
		break;
	case CL_INVALID_BUFFER_SIZE:
		printf("%3d : Invalid buffer size.\n", RET);
		break;
	case CL_INVALID_MIP_LEVEL:
		printf("%3d : Invalid map level.\n", RET);
		break;
	case CL_INVALID_GLOBAL_WORK_SIZE:
		printf("%3d : Invalid global work size.\n", RET);
		break;
	case CL_INVALID_PROPERTY:
		printf("%3d : Invalid property.\n", RET);
		break;
	case CL_INVALID_IMAGE_DESCRIPTOR:
		printf("%3d : Invalid image descriptor.\n", RET);
		break;
	case CL_INVALID_COMPILER_OPTIONS:
		printf("%3d : Invalid compiler options.\n", RET);
		break;
	case CL_INVALID_LINKER_OPTIONS:
		printf("%3d : Invalid linker options.\n", RET);
		break;
	case CL_INVALID_DEVICE_PARTITION_COUNT:
		printf("%3d : Invalid device partition count.\n", RET);
		break;
	case CL_INVALID_PIPE_SIZE:
		printf("%3d : Invalid pipe size.\n", RET);
		break;
	case CL_INVALID_DEVICE_QUEUE:
		printf("%3d : Invalid device queue.\n", RET);
		break;
	default:
		break;
	}
	if (useInterrupt) assert(RET == CL_SUCCESS);
}

void mobilenetv2::clImageLoader(const char* ImageFileName)
{
	//delete[] Image;
	std::ifstream InputImage;
	InputImage.open(ImageFileName, std::ios::binary);
	Image = new float[224 * 224 * 3];
	if (!InputImage.is_open())
		printf("Open image file \"%s\"error.\n", ImageFileName);
	else
	{
		BMP img;
		InputImage.read((char*)& img, sizeof(img));
		for (int d = 0; d < 3; d++)
		{
			for (int h = 0; h < 224; h++)
			{
				for (int w = 0; w < 224; w++)
				{
					float data = img.data[224 - h - 1][w][d];
					Image[(2-d) * 224 * 224 + h * 224 + w] = data / 255.0;
					/*RGB*/
				}
			}
		}
		
		InputImage.close();
	}
}
