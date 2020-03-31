#include "TridiagSolver.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

TridiagSolver::TridiagSolver(int dim, int size, Method type) : paramsLocation(ParamsLocation::HOST)
{
	this->dim = dim;
	this->size = size;
	this->method = type;
}

void TridiagSolver::setParamsLocation(ParamsLocation location)
{
	this->paramsLocation = location;
}

void TridiagSolver::solve(float* a, float* b, float* c, float* y)
{
	float* d_a, * d_b, * d_c, * d_y;

	if (this->paramsLocation == ParamsLocation::DEVICE) {
		d_a = a; d_b = b; d_c = c; d_y = y;
	}
	else {
		cudaMalloc(&d_a, size * dim * sizeof(float));
		cudaMalloc(&d_b, size * dim * sizeof(float));
		cudaMalloc(&d_c, size * dim * sizeof(float));
		cudaMalloc(&d_y, size * dim * sizeof(float));

		cudaMemcpy(d_a, a, size * dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size * dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, c, size * dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, size * dim * sizeof(float), cudaMemcpyHostToDevice);
	}

	if (this->method == Method::THOMAS) {		
		solveThomas(d_a, d_b, d_c, d_y);
	}
	else {
		solvePCR(d_a, d_b, d_c, d_y);
	}
	cudaDeviceSynchronize();

	if (this->paramsLocation == ParamsLocation::HOST) {
		cudaMemcpy(a, d_a, size * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(b, d_b, size * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(c, d_c, size * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(y, d_y, size * dim * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		cudaFree(d_y);
	}
}

void TridiagSolver::solveThomas(float* d_a, float* d_b, float* d_c, float* d_y)
{
	int nbBlocks = (int) (size + 255) / 256;
	int blockSize = 256;

	TridiagKernel::thomas_wrapper(nbBlocks, blockSize, d_a, d_b, d_c, d_y, dim);
}

void TridiagSolver::solvePCR(float* d_a, float* d_b, float* d_c, float* d_y)
{
	int minTB = (dim > 255) + 4 * (dim > 63 && dim < 256) + 16 * (dim > 15 && dim < 64) + 64 * (dim > 3 && dim < 16);
	int nbBlocks = (size + minTB - 1) / minTB;
	int blockSize = dim * minTB;
	int sharedMem = 5 * minTB * dim * sizeof(float);

	TridiagKernel::pcr_wrapper(nbBlocks, blockSize, sharedMem, d_a, d_b, d_c, d_y, dim);
}