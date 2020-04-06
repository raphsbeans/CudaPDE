#include "TridiagSolver.h"
#include "TridiagKernel.cuh"
#include <stdexcept>

TridiagSolver::TridiagSolver(int dim, int size, Method type) :
	paramsLocation(ParamsLocation::HOST),
	lastElapsedTime(0),
	avgElapsedTime(0),
	nbCalls(0)
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

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	if (this->method == Method::THOMAS) {
		TridiagSolverImpl::thomasGPU(size, dim, d_a, d_b, d_c, d_y);
	}
	else {
		TridiagSolverImpl::pcr(size, dim, d_a, d_b, d_c, d_y);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&lastElapsedTime, start, stop);
	avgElapsedTime = avgElapsedTime * nbCalls + lastElapsedTime;
	nbCalls++;
	avgElapsedTime /= nbCalls;

	//cudaDeviceSynchronize();

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

void TridiagSolver::solve(float a, float b, float c, float* y)
{
	float* d_y;

	if (this->paramsLocation == ParamsLocation::DEVICE) {
		d_y = y;
	}
	else {
		cudaMalloc(&d_y, size * dim * sizeof(float));
		cudaMemcpy(d_y, y, size * dim * sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	if (this->method == Method::THOMAS) {
		throw std::invalid_argument("Thomas method for constant diagonals not yet implemented.");
	}
	else {
		TridiagSolverImpl::pcr(size, dim, a, b, c, d_y);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&lastElapsedTime, start, stop);
	avgElapsedTime = avgElapsedTime * nbCalls + lastElapsedTime;
	nbCalls++;
	avgElapsedTime /= nbCalls;

	if (this->paramsLocation == ParamsLocation::HOST) {
		cudaMemcpy(y, d_y, size * dim * sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_y);
	}
}