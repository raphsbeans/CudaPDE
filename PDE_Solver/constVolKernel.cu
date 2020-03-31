#include "constVolKernel.cuh"

namespace ConstVolKernel {

	// dim = dimension of P
	// size = number of vectors u_i
	void Pxu(int dim, int size, float p_d, float p_m, float p_u, float* u) {
		// one block per multiplication
		int nbBlocks = size;
		int blockSize = dim;
		int sharedMemSize = dim * sizeof(float);
		Pxu_k<<<nbBlocks, blockSize, sharedMemSize>>> (p_d, p_m, p_u, u, dim);
	}

	// calculate P * u_i for many i's and store answer in u (the size of u must be a multiple of the dimension of P)
	// blockSize = k * (dimension of P) ---> (each block does k matrix multiplications)
	// nbBlocks = (size of u) / (dim * k) 
	__global__ void Pxu_k(float p_d, float p_m, float p_u, float* u, int n)
	{
		// Identifies the thread working within a group
		int tidx = threadIdx.x % n; 
		// Identifies the data concerned by the computations
		int Qt = (threadIdx.x - tidx) / n;

		extern __shared__ float sAds[];
		float* su = (float*)&sAds[Qt * n];
		su[threadIdx.x] = u[blockIdx.x * blockDim.x + threadIdx.x];
		__syncthreads();

		float temp;
		if (tidx > 0 && tidx < n - 1)
			temp = p_d * su[tidx - 1] + p_m * su[tidx] + p_u * su[tidx + 1];
		else if (tidx == 0)
			temp = p_m * su[tidx] + p_u * su[tidx + 1];
		else
			temp = p_d * su[tidx - 1] + p_m * su[tidx];

		u[blockIdx.x * blockDim.x + threadIdx.x] = temp;
	}
}


