
#define HIDE
#ifndef HIDE

#include "cuda_definitions.cuh"
#include "TridiagSolver.h"
#include "global_definitions.h"

#include <iostream>
using namespace std;


// Produces tridiagonal symmetric diagonally dominant matrices 
// Tri_k <<<NB,HM*Dim>>>(da, db, dc, 10.0f, i * Dim * NB, Dim, Dim * NB);
__global__ void Tri_k(float* a, float* b, float* c, float norm, int n)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x % n;
	// Identifies the data concerned by the computations
	int Qt = (threadIdx.x - tidx) / n;
	// The global memory access index
	int gb_index_x = Qt + blockIdx.x * (blockDim.x / n);

	b[gb_index_x * n + tidx] = ((float)tidx + 1.0f) / (norm);
	if (tidx > 0 && tidx < n - 1) {
		a[gb_index_x * n + tidx] = ((float)tidx + 1.0f) / (norm * 3);
		c[gb_index_x * n + tidx] = ((float)tidx + 1.0f) / (norm * 3);
	}
	else if (tidx == 0) {
		a[gb_index_x * n + tidx] = 0.0f;
	}
	else {
		c[gb_index_x * n + tidx] = 0.0f;
	}
}

int main() {
	// The rank of the matrix
	int Dim = 64;
	// The number of blocks
	int NB = M / Dim;
	// The number of matrices to invert
	int size = NB;

	// The diagonal elements
	float *y;
	float *d_a, *d_b, *d_c, *d_y;

	y = (float*) calloc(size * Dim, sizeof(float));

	cudaMalloc(&d_a, size * Dim * sizeof(float));
	cudaMalloc(&d_b, size * Dim * sizeof(float));
	cudaMalloc(&d_c, size * Dim * sizeof(float));
	cudaMalloc(&d_y, size * Dim * sizeof(float));

	// Tridiagonal elements
	Tri_k << <NB, Dim >> > (d_a, d_b, d_c, 10.0f, Dim);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < Dim; j++) {
			y[j + i * Dim] = 0.5f * j;
		}
	}

	cudaMemcpy(d_y, y, size * Dim * sizeof(float), cudaMemcpyHostToDevice);

	TridiagSolver solver(Dim, size, TridiagSolver::Method::PCR);
	solver.setParamsLocation(ParamsLocation::DEVICE);
	solver.solve(d_a, d_b, d_c, d_y);

	cudaMemcpy(y, d_y, size * Dim * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t k = 0; k < 100; ++k)
	{
		printf("\n%.3e", y[k]);
	}

	free(y);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_y);

	return 0;
}

#endif