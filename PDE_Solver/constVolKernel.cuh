#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace ConstVolKernel {

	void Pxu(int dim, int size, float p_d, float p_m, float p_u, float* u);

	__global__ void Pxu_k(float p_d, float p_m, float p_u, float* u, int n);

}

