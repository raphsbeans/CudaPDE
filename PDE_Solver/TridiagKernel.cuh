#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define EPS (0.0000001f)
#define M (4 * 1048576)
__device__ int gl[M];


namespace TridiagKernel {

	void thomas_wrapper(int nbBlocks, int blockSize, float *d_a, float* d_b, float* d_c, float* d_y, int dim);
	__global__ void thom_k(float* a, float* b, float* c, float* y, int n);

	void pcr_wrapper(int nbBlocks, int blockSize, int sharedMem, float* d_a, float* d_b, float* d_c, float* d_y, int dim);
	__global__ void pcr_k(float* a, float* b, float* c, float* y, int n);

	void pcr_wrapper(int nbBlocks, int blockSize, int sharedMem, float d_a, float d_b, float d_c, float* d_y, int dim);
	__global__ void pcr_k(float a, float b, float c, float* y, int n);

}