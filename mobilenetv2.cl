/* First layer convolution [224,224,3] (3x3,/2,RB pad) [112,112,32] */
kernel void clConvBNReLU_A0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	int fi = 0;
	#pragma unroll
	for (int din = 0; din < 3; din++)
	{
		#pragma unroll
		for (int fh = 0; fh < 3; fh++)
		{
			#pragma unroll
			for (int fw = 0; fw < 3; fw++,fi++)
			{
				int Hp = H * 2 + fh;
				int Wp = W * 2 + fw;
				
				float F = Filter[D * 27 + fi];
				float I = 0;
				if((Hp < 224) && (Wp < 224))
					I = Input[din * 224 * 224 + Hp * 224 + Wp];
				SUM += I * F;
			}
		}
	}
	//SUM = GOS * (SIUM - MEAN) +  BETA;
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 112 * 112 + H * 112 + W] = SUM;
}



/*==================================================================================================*/
/* Block layer 1 : 1 block, stride 1 */ 
/* depth-wise convolution [112,112,32] (3x3,same pad) [112,112,32] */
kernel void clDWConvBNReLU_B0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
				
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >=0 && Hp < 112) && (Wp >=0 && Wp < 112))
				I = Input[D * 112 * 112 + Hp * 112 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 112 * 112 + H * 112 + W] = SUM;
}

/* point-wise convolution [112,112,32] (1x1) [112,112,16] */
kernel void clPWConvBNLinear_B1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 32; din++)
	{
		float F = Filter[D * 32 + din ];
		float I = Input[din * 112 * 112 + H * 112 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 112 * 112 + H * 112 + W] = SUM;
}





/*==================================================================================================*/
/* Block layer 2 : 2 blocks, stride 2 */ 
/* point-wise convolution [112,112,16] (1x1) [112,112,96] */
kernel void clPWConvBNReLU_C0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 16; din++)
	{
		float F = Filter[D * 16 + din ];
		float I = Input[din * 112 * 112 + H * 112 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 112 * 112 + H * 112 + W] = SUM;
}

/* depth-wise convolution [112,112,96] (3x3, /2, RB pad) [56,56,96] */
kernel void clDWConvBNReLU_C1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 2 + fh;
			int Wp = W * 2 + fw;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp < 112) && (Wp < 112))
				I = Input[D * 112 * 112 + Hp * 112 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = SUM;
}

/* point-wise convolution [56,56,96] (1x1) [56,56,24] */
kernel void clPWConvBNLinear_C2(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 96; din++)
	{
		float F = Filter[D * 96 + din ];
		float I = Input[din * 56 * 56 + H * 56 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = SUM;
}

/* point-wise convolution [56,56,24] (1x1) [56,56,144] */
kernel void clPWConvBNReLU_C3(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 24; din++)
	{
		float F = Filter[D * 24 + din ];
		float I = Input[din * 56 * 56 + H * 56 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = SUM;
}

/* depth-wise convolution [56,56,144] (3x3, same pad) [56,56,144] */
kernel void clDWConvBNReLU_C4(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 56) && (Wp >= 0 && Wp < 56))
				I = Input[D * 56 * 56 + Hp * 56 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = SUM;
}

/* point-wise convolution [56,56,144] (1x1) [56,56,24] */
kernel void clPWConvBNLinear_C5(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 144; din++)
	{
		float F = Filter[D * 144 + din ];
		float I = Input[din * 56 * 56 + H * 56 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = SUM;
}

/* Adding merge [56,56,24] */
kernel void clAdd_C6(
global float* InputA,
global float* InputB,
global float* Output)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = InputA[D * 56 * 56 + H * 56 + W] + InputB[D * 56 * 56 + H * 56 + W];
}





/*==================================================================================================*/
/* Block layer 3 : 3 blocks, stride 2 */ 
/* point-wise convolution [56,56,24] (1x1) [56,56,144] */
kernel void clPWConvBNReLU_D0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 24; din++)
	{
		float F = Filter[D * 24 + din ];
		float I = Input[din * 56 * 56 + H * 56 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 56 * 56 + H * 56 + W] = SUM;
}

/* depth-wise convolution [56,56,144] (3x3, /2, RB pad) [28,28,144] */
kernel void clDWConvBNReLU_D1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 2 + fh;
			int Wp = W * 2 + fw;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp < 56) && (Wp < 56))
				I = Input[D * 56 * 56 + Hp * 56 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = SUM;
}

/* point-wise convolution [28,28,144] (1x1) [28,28,32] */
kernel void clPWConvBNLinear_D2(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 144; din++)
	{
		float F = Filter[D * 144 + din ];
		float I = Input[din * 28 * 28 + H * 28 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = SUM;
}

/* point-wise convolution [28,28,32] (1x1) [28,28,192] */
kernel void clPWConvBNReLU_D3(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 32; din++)
	{
		float F = Filter[D * 32 + din ];
		float I = Input[din * 28 * 28 + H * 28 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = SUM;
}

/* depth-wise convolution [28,28,192] (3x3, same pad) [28,28,192] */
kernel void clDWConvBNReLU_D4(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 28) && (Wp >= 0 && Wp < 28))
				I = Input[D * 28 * 28 + Hp * 28 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = SUM;
}

/* point-wise convolution [28,28,192] (1x1) [28,28,32] */
kernel void clPWConvBNLinear_D5(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 192; din++)
	{
		float F = Filter[D * 192 + din ];
		float I = Input[din * 28 * 28 + H * 28 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = SUM;
}

/* Adding merge [28,28,32] */
kernel void clAdd_D6(
global float* InputA,
global float* InputB,
global float* Output)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = InputA[D * 28 * 28 + H * 28 + W] + InputB[D * 28 * 28 + H * 28 + W];
}




/*==================================================================================================*/
/* Block layer 4 : 4 blocks, stride 2 */ 
/* point-wise convolution [28,28,32] (1x1) [28,28,192] */
kernel void clPWConvBNReLU_E0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 32; din++)
	{
		float F = Filter[D * 32 + din ];
		float I = Input[din * 28 * 28 + H * 28 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 28 * 28 + H * 28 + W] = SUM;
}

/* depth-wise convolution [28,28,192] (3x3, /2, RB pad) [14,14,192] */
kernel void clDWConvBNReLU_E1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 2 + fh;
			int Wp = W * 2 + fw;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp < 28) && (Wp < 28))
				I = Input[D * 28 * 28 + Hp * 28 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* point-wise convolution [14,14,192] (1x1) [14,14,64] */
kernel void clPWConvBNLinear_E2(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 192; din++)
	{
		float F = Filter[D * 192 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* point-wise convolution [14,14,64] (1x1) [14,14,384] */
kernel void clPWConvBNReLU_E3(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 64; din++)
	{
		float F = Filter[D * 64 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* depth-wise convolution [14,14,384] (3x3, same pad) [14,14,384] */
kernel void clDWConvBNReLU_E4(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 14) && (Wp >= 0 && Wp < 14))
				I = Input[D * 14 * 14 + Hp * 14 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* point-wise convolution [14,14,384] (1x1) [14,14,64] */
kernel void clPWConvBNLinear_E5(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 384; din++)
	{
		float F = Filter[D * 384 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* Adding merge [14,14,64] */
kernel void clAdd_E6(
global float* InputA,
global float* InputB,
global float* Output)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = InputA[D * 14 * 14 + H * 14 + W] + InputB[D * 14 * 14 + H * 14 + W];
}





/*==================================================================================================*/
/* Block layer 5 : 3 blocks, stride 1 */ 
/* point-wise convolution [14,14,64] (1x1) [14,14,384] */
kernel void clPWConvBNReLU_F0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 64; din++)
	{
		float F = Filter[D * 64 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* depth-wise convolution [14,14,384] (3x3, same pad) [14,14,384] */
kernel void clDWConvBNReLU_F1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 14) && (Wp >= 0 && Wp < 14))
				I = Input[D * 14 * 14 + Hp * 14 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* point-wise convolution [14,14,384] (1x1) [14,14,96] */
kernel void clPWConvBNLinear_F2(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 384; din++)
	{
		float F = Filter[D * 384 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* point-wise convolution [14,14,96] (1x1) [14,14,576] */
kernel void clPWConvBNReLU_F3(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 96; din++)
	{
		float F = Filter[D * 96 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* depth-wise convolution [14,14,576] (3x3, same pad) [14,14,576] */
kernel void clDWConvBNReLU_F4(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 14) && (Wp >= 0 && Wp < 14))
				I = Input[D * 14 * 14 + Hp * 14 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* point-wise convolution [14,14,576] (1x1) [14,14,96] */
kernel void clPWConvBNLinear_F5(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 576; din++)
	{
		float F = Filter[D * 576 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* Adding merge [14,14,96] */
kernel void clAdd_F6(
global float* InputA,
global float* InputB,
global float* Output)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = InputA[D * 14 * 14 + H * 14 + W] + InputB[D * 14 * 14 + H * 14 + W];
}





/*==================================================================================================*/
/* Block layer 6 : 3 blocks, stride 2 */ 
/* point-wise convolution [14,14,96] (1x1) [14,14,576] */
kernel void clPWConvBNReLU_G0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 96; din++)
	{
		float F = Filter[D * 96 + din ];
		float I = Input[din * 14 * 14 + H * 14 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 14 * 14 + H * 14 + W] = SUM;
}

/* depth-wise convolution [14,14,576] (3x3, /2, RB pad) [7,7,576] */
kernel void clDWConvBNReLU_G1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 2 + fh;
			int Wp = W * 2 + fw;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp < 14) && (Wp < 14))
				I = Input[D * 14 * 14 + Hp * 14 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* point-wise convolution [7,7,576] (1x1) [7,7,160] */
kernel void clPWConvBNLinear_G2(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 576; din++)
	{
		float F = Filter[D * 576 + din ];
		float I = Input[din * 7 * 7 + H * 7 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* point-wise convolution [7,7,160] (1x1) [7,7,960] */
kernel void clPWConvBNReLU_G3(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 160; din++)
	{
		float F = Filter[D * 160 + din ];
		float I = Input[din * 7 * 7 + H * 7 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* depth-wise convolution [7,7,960] (3x3, same pad) [7,7,960] */
kernel void clDWConvBNReLU_G4(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 7) && (Wp >= 0 && Wp < 7))
				I = Input[D * 7 * 7 + Hp * 7 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* point-wise convolution [7,7,960] (1x1) [7,7,160] */
kernel void clPWConvBNLinear_G5(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 960; din++)
	{
		float F = Filter[D * 960 + din ];
		float I = Input[din * 7 * 7 + H * 7 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* Adding merge [7,7,160] */
kernel void clAdd_G6(
global float* InputA,
global float* InputB,
global float* Output)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = InputA[D * 7 * 7 + H * 7 + W] + InputB[D * 7 * 7 + H * 7 + W];
}





/*==================================================================================================*/
/* Block layer 7 : 1 blocks, stride 1 */ 
/* point-wise convolution [7,7,160] (1x1) [7,7,960] */
kernel void clPWConvBNReLU_H0(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 160; din++)
	{
		float F = Filter[D * 160 + din ];
		float I = Input[din * 7 * 7 + H * 7 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* depth-wise convolution [7,7,960] (3x3, same pad) [7,7,960] */
kernel void clDWConvBNReLU_H1(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		#pragma unroll
		for (int fw = 0; fw < 3; fw++)
		{
			int Hp = H * 1 + fh - 1;
			int Wp = W * 1 + fw - 1;
			float F = Filter[D * 9 + fh * 3 + fw];
			float I = 0;
			if((Hp >= 0 && Hp < 7) && (Wp >= 0 && Wp < 7))
				I = Input[D * 7 * 7 + Hp * 7 + Wp];
			SUM += I * F;
		}
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* point-wise convolution [7,7,960] (1x1) [7,7,320] */
kernel void clPWConvBNLinear_H2(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 960; din++)
	{
		float F = Filter[D * 960 + din ];
		float I = Input[din * 7 * 7 + H * 7 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}

/* point-wise convolution [7,7,320] (1x1) [7,7,1280] */
kernel void clPWConvBNReLU_H3(
global float* Input,
global float* Output,
global float* Filter,
global float* BN)
{
	int W = get_global_id(0);
	int H = get_global_id(1);
	int D = get_global_id(2);
	float GOS  = BN[D * 4 + 0];
	float BETA = BN[D * 4 + 1];
	float MEAN = BN[D * 4 + 2];
	float SIGMA2 = BN[D * 4 + 3];
	float SUM = 0;
	#pragma unroll (4)
	for (int din = 0; din < 320; din++)
	{
		float F = Filter[D * 320 + din ];
		float I = Input[din * 7 * 7 + H * 7 + W];
		SUM += I * F;
	}
	SUM = GOS * (SUM - MEAN)/sqrt(SIGMA2+0.001)+BETA;
	if (SUM < 0) SUM = 0;
	if (SUM > 6) SUM = 6;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D * 7 * 7 + H * 7 + W] = SUM;
}





/*==================================================================================================*/
/* Global average pooling [7,7,1280] (7x7) [1,1,1280] */
kernel void clGlobalAveragePool(
global float* Input,
global float* Output)
{
	int D = get_global_id(0);
	float SUM = 0;
	#pragma unroll
	for (int p = 0; p < 49; p++)
		SUM += Input[D * 49 + p];
	
	SUM /= 49;
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[D] = SUM;
}



/*==================================================================================================*/
/* Dense [1,1,1280] (1280 x Class) [1,1,Class]*/
kernel void clDense(
global float* Input,
global float* Output,
global float* Weight,
global float* Bias)
{
	int V = get_global_id(0);
	int Units = get_global_size(0);
	float SUM = 0;
	#pragma unroll (4)
	for (int vin = 0; vin < 1280; vin++)
	{
		SUM += Input[vin] * Weight[vin * Units + V];
	}
	SUM += Bias[V];
	Output[V] = SUM;
}





/*==================================================================================================*/
/* Softmax [Class] */
kernel void clSoftmax(
global float* Input, 
global float* Output)
{
	int C = get_global_id(0);
	int Classes = get_global_size(0);
	float SUM = 0;
		
	#pragma unroll
	for (int idx = 0; idx < Classes; idx++)
		SUM += exp(Input[idx]);
	barrier(CLK_GLOBAL_MEM_FENCE);
	Output[C] = exp(Input[C]) / SUM;
}

