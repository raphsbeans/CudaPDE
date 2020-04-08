#include "BulletOptionKernel.cuh"
#include "cuda_definitions.cuh"
#include <math.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
void dumpMatrix(size_t size1, size_t size2, float* matrix_d, char* path)
{
	float* matrix = new float[size1 * size2];
	cudaMemcpy(matrix, matrix_d, size1 * size2 * sizeof(float), cudaMemcpyDeviceToHost);

	std::ofstream file(path);

	for (size_t i = 0; i < size1; i++) {
		for (size_t j = 0; j < size2; j++) {
			file << matrix[i + j * size1];
			if (j < size2 - 1)
				file << ",";
			else
				file << "\n";
		}
	}

	file.close();

	delete[] matrix;
}

namespace BulletOptionKernel {

	void boundaryConditionGPU(size_t spotGridSize, size_t stateSize, float* payoff, float strike) {
		boundaryCondition_k <<< 1, stateSize >>> (payoff, spotGridSize, strike);
		dumpMatrix(spotGridSize, stateSize, payoff, "C:\\Users\\erik\\Desktop\\bound.csv");
	}

	__global__ void boundaryCondition_k(float* payoff, size_t spotSize, float strike) {
		size_t state_idx = threadIdx.x;
		payoff[spotSize - 1 + state_idx * spotSize] = 2 * strike;
		payoff[0 + state_idx * spotSize] = 0.0;
	}


	void initPayoffGPU(size_t spotSize, size_t stateSize, float* payoff, float dx, float Smin, float strike, size_t P1, size_t P2) {
		initPayoff_k <<< stateSize, spotSize >> > (payoff, dx, Smin, strike, P1, P2);
		dumpMatrix(spotSize, stateSize, payoff, "C:\\Users\\erik\\Desktop\\initPayoff.csv");
	}

	__global__ void initPayoff_k(float* payoff, float dx, float Smin, float strike, size_t P1, size_t P2) {
		size_t spot_idx = threadIdx.x;
		size_t state_idx = blockIdx.x;

		float spot = Smin * expf(spot_idx * dx);
		size_t idx = spot_idx + state_idx * blockDim.x;

		// !! state grid value is equal to state index !!
		payoff[idx] = fmaxf(0.0f, spot - strike);// *((state_idx <= P2) && (state_idx >= P1));
	}

	void interStepGPU(size_t spotSize, size_t stateSize, float* payoff, size_t scheduleCounter, float dx, float Smin, size_t P1, size_t P2, float barrier)
	{
		// schedule size = state size
		interStep_k <<< spotSize, stateSize >>> (payoff, scheduleCounter, dx, Smin, P1, P2, barrier);
		dumpMatrix(spotSize, stateSize, payoff, "C:\\Users\\erik\\Desktop\\interStep.csv");
	}

	__global__ void interStep_k(float* payoff, size_t scheduleCounter, float dx, float Smin, size_t P1, size_t P2, float barrier)
	{
		size_t spot_idx = blockIdx.x;
		size_t state_idx = threadIdx.x;

		// shared memory for the payoff for a fixed spot (each block corresponds to a spot_dx)
		//extern __shared__ float shared_payoff_x[]; 
		//shared_payoff_x[state_idx] = payoff[spot_idx + state_idx * gridDim.x];
		//__syncthreads();

		float temp = 0.0;
		size_t P1_k = P1 > scheduleCounter ? P1 - scheduleCounter : 0;
		float spot = Smin * expf(spot_idx * dx);
		
		if (state_idx == P2)
		{
			temp = payoff[spot_idx + P2 * gridDim.x] * (spot >= barrier);
		}
		else if (P1_k <= state_idx && state_idx < P2) 
		{
			temp = payoff[spot_idx + state_idx * gridDim.x] * (spot >= barrier)
				+ payoff[spot_idx + (state_idx + 1) * gridDim.x] * (spot < barrier);
		}
		else if (state_idx == P1_k) 
		{
			temp = payoff[spot_idx + P1_k * gridDim.x] * (spot < barrier);
		}

		__syncthreads();
		payoff[spot_idx + state_idx * gridDim.x] = temp;
	}
}