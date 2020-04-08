#include "constVolKernel.cuh"

namespace ConstVolKernel {

	void tridiag_x_matrix_GPU(int dim, int size, float* p_d, float* p_m, float* p_u, float* u) {
		// one block per multiplication
		int nbBlocks = size;
		int blockSize = dim;
		int sharedMemSize = 4 * blockSize * sizeof(float);
		tridiag_x_matrix_k <<< nbBlocks, blockSize, sharedMemSize >>> (p_d, p_m, p_u, u, dim);
		cudaDeviceSynchronize();
	}

	void tridiag_x_matrix_GPU(int dim, int size, float p_d, float p_m, float p_u, float* u) {
		// one block per multiplication
		int nbBlocks = size;
		int blockSize = dim;
		int sharedMemSize = blockSize * sizeof(float);
		tridiag_x_matrix_k <<< nbBlocks, blockSize, sharedMemSize >>> (p_d, p_m, p_u, u, dim);
		cudaDeviceSynchronize();
	}

	__global__ void tridiag_x_matrix_k(float p_d, float p_m, float p_u, float* u, int n)
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

	__global__ void tridiag_x_matrix_k(float* p_d, float* p_m, float* p_u, float* u, int n)
	{
		// Identifies the thread working within a group
		int tidx = threadIdx.x % n;
		// Identifies the data concerned by the computations
		int Qt = (threadIdx.x - tidx) / n;

		extern __shared__ float sAds[];
		float* su, * sp_d, * sp_m, * sp_u;
		su = (float*)&sAds[4 * Qt * n];
		sp_d = (float*)&su[n];
		sp_m = (float*)&sp_d[n];
		sp_u = (float*)&sp_m[n];

		su[threadIdx.x] = u[blockIdx.x * blockDim.x + threadIdx.x];
		sp_d[threadIdx.x] = p_d[tidx];
		sp_m[threadIdx.x] = p_m[tidx];;
		sp_u[threadIdx.x] = p_u[tidx];;
		__syncthreads();

		float temp;
		if (tidx > 0 && tidx < n - 1)
			temp = sp_d[tidx] * su[tidx - 1] + sp_m[tidx] * su[tidx] + sp_u[tidx] * su[tidx + 1];
		else if (tidx == 0)
			temp = sp_m[tidx] * su[tidx] + sp_u[tidx] * su[tidx + 1];
		else
			temp = sp_d[tidx] * su[tidx - 1] + sp_m[tidx] * su[tidx];

		u[blockIdx.x * blockDim.x + threadIdx.x] = temp;
	}

}

