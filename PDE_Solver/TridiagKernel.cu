#include "TridiagKernel.cuh"

#include <stdio.h>
#include <math.h>

namespace TridiagSolverImpl {

	void thomasGPU(size_t size, size_t dim, float* d_a, float* d_b, float* d_c, float* d_y)
	{
		size_t nbBlocks = (int)(size + 255) / 256;
		size_t blockSize = 256;

		thom_k <<< nbBlocks, blockSize >>> (d_a, d_b, d_c, d_y, dim);
	}

	void pcr(size_t size, size_t dim, float* d_a, float* d_b, float* d_c, float* d_y)
	{
		size_t minTB = (dim > 255) + 4 * (dim > 63 && dim < 256) + 16 * (dim > 15 && dim < 64) + 64 * (dim > 3 && dim < 16);
		size_t nbBlocks = (size + minTB - 1) / minTB;
		size_t blockSize = dim * minTB;
		size_t sharedMem = 5 * minTB * dim * sizeof(float);

		pcr_k <<< nbBlocks, blockSize, sharedMem >>> (d_a, d_b, d_c, d_y, dim);
	}

	void pcr(size_t size, size_t dim, float a, float b, float c, float* d_y)
	{
		size_t minTB = (dim > 255) + 4 * (dim > 63 && dim < 256) + 16 * (dim > 15 && dim < 64) + 64 * (dim > 3 && dim < 16);
		size_t nbBlocks = (size + minTB - 1) / minTB;
		size_t blockSize = dim * minTB;
		size_t sharedMem = 5 * minTB * dim * sizeof(float);

		pcr_k <<< nbBlocks, blockSize, sharedMem >>> (a, b, c, d_y, dim);
	}

	// Thomas
	__global__ void thom_k(float* a, float* b, float* c, float* y, int n) {
		// The global memory access index
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		float bet, * gam;
		gam = &gl.float_t[idx * n];

		if (fabs(b[idx * n]) < EPS)
			printf("Error 1 in tridiag");

		// If this happens then you should rewrite your equations as a set of order N - 1
		bet = b[idx * n];
		y[idx * n] = y[idx * n] / bet;
		for (int j = 1; j < n; ++j) {
			gam[j] = c[idx * n + j - 1] / bet;
			bet = b[idx * n + j] - a[idx * n + j] * gam[j];

			if (fabsf(bet) < EPS)
				printf("Error 2 in tridag"); //Algorithm fails;

			y[idx * n + j] = (y[idx * n + j] - a[idx * n + j] * y[idx * n + j - 1]) / bet;
		}

		for (int j = n - 2; j >= 0; --j) {
			y[idx * n + j] -= gam[j + 1] * y[idx * n + j + 1]; // Backsubstitution
		}

	}

	// PCR 
	__global__ void pcr_k(float* a, float* b, float* c, float* y, int n) {
		// Identifies the thread working within a group
		int tidx = threadIdx.x % n;
		// Identifies the data concerned by the computations
		int Qt = (threadIdx.x - tidx) / n;
		// The global memory access index
		int gb_index_x = Qt + blockIdx.x * (blockDim.x / n);
		// Local integers
		int i, nt, lL, d, tL, tR;
		// Local floats
		float aL, bL, cL, yL, aLp, bLp, cLp, yLp;
		// Shared memory
		extern __shared__ float sAds[];

		nt = 5 * Qt * n;
		d = (n / 2 + (n % 2)) * (tidx % 2) + (int)tidx / 2;
		float* sa = (float*)&sAds[nt];
		float* sb = (float*)&sa[n];
		float* sc = (float*)&sb[n];
		float* sy = (float*)&sc[n];
		int* sl = (int*)&sy[n];

		sa[tidx] = a[gb_index_x * n + tidx];
		sb[tidx] = b[gb_index_x * n + tidx];
		sc[tidx] = c[gb_index_x * n + tidx];
		sy[tidx] = y[gb_index_x * n + tidx];
		sl[tidx] = tidx;
		__syncthreads();

		//Left/Right indices of the reduction
		tL = tidx - 1;
		if (tL < 0) tL = 0;
		tR = tidx + 1;
		if (tR >= n) tR = 0;

		for (i = 0; i < (int)log2((float)n) + 1; i++) {
			lL = (int)sl[tidx];
			aL = sa[tidx];
			bL = sb[tidx];
			cL = sc[tidx];
			yL = sy[tidx];

			bLp = sb[tL];
			//Reduction phase
			if (fabsf(aL) > EPS) {
				aLp = sa[tL];
				cLp = sc[tL];
				yLp = sy[tL];
				float temp1 = aL / bLp;
				bL -= cLp * temp1;
				yL -= yLp * temp1;
				aL = -aLp * temp1;
			}

			aLp = sa[tR];
			bLp = sb[tR];
			cLp = sc[tR];
			float temp2 = cL / bLp;
			if (fabsf(aLp) > EPS) {
				yLp = sy[tR];
				bL -= aLp * temp2;
				yL -= yLp * temp2;
				cL = -cLp * temp2;
			}
			__syncthreads();
			//Permutation phase
			if (i < (int)log2((float)n)) {
				sa[d] = aL;
				sb[d] = bL;
				sc[d] = cL;
				sy[d] = yL;
				sl[d] = (int)lL;
				__syncthreads();
			}
		}

		sy[(int)tidx] = yL / bL;
		__syncthreads();
		y[gb_index_x * n + sl[tidx]] = sy[tidx];
	}

	// PCR with constant diagonals
	__global__ void pcr_k(float a, float b, float c, float* y, int n) {
		// Identifies the thread working within a group
		int tidx = threadIdx.x % n;
		// Identifies the data concerned by the computations
		int Qt = (threadIdx.x - tidx) / n;
		// The global memory access index
		int gb_index_x = Qt + blockIdx.x * (blockDim.x / n);
		// Local integers
		int i, nt, lL, d, tL, tR;
		// Local floats
		float aL, bL, cL, yL, aLp, bLp, cLp, yLp;
		// Shared memory
		extern __shared__ float sAds[];

		nt = 5 * Qt * n;
		d = (n / 2 + (n % 2)) * (tidx % 2) + (int)tidx / 2;
		float* sa = (float*)&sAds[nt];
		float* sb = (float*)&sa[n];
		float* sc = (float*)&sb[n];
		float* sy = (float*)&sc[n];
		int* sl = (int*)&sy[n];

		sa[tidx] = a;
		sb[tidx] = b;
		sc[tidx] = c;
		sy[tidx] = y[gb_index_x * n + tidx];
		sl[tidx] = tidx;
		__syncthreads();

		//Left/Right indices of the reduction
		tL = tidx - 1;
		if (tL < 0) tL = 0;
		tR = tidx + 1;
		if (tR >= n) tR = 0;

		for (i = 0; i < (int)log2((float)n) + 1; i++) {
			lL = (int)sl[tidx];
			aL = sa[tidx];
			bL = sb[tidx];
			cL = sc[tidx];
			yL = sy[tidx];

			bLp = sb[tL];
			//Reduction phase
			if (fabsf(aL) > EPS) {
				aLp = sa[tL];
				cLp = sc[tL];
				yLp = sy[tL];

				//bL = b[tidx] - a[tidx]*c[tidx]/b[tidx-1];
				bL -= aL * cL / bLp;
				//yL = y[tidx] - a[tidx]*y[tidx-1]/b[tidx-1];
				yL -= aL * yLp / bLp;
				//aL = -a[tidx]*a[tidx-1]/b[tidx-1];
				aL = -aL * aLp / bLp;

				//aL = -aL * aLp / bLp;
			}

			aLp = sa[tR];
			bLp = sb[tR];
			cLp = sc[tR];
			if (fabsf(aLp) > EPS) {
				yLp = sy[tR];
				//bL -= c[tidx+1]*a[tidx+1]/b[tidx+1];
				bL -= cLp * aLp / bLp;
				//yL -= c[tidx+1]*y[tidx+1]/b[tidx+1];
				yL -= cLp * yLp / bLp;

				cL = -cL * cLp / bLp;
			}
			__syncthreads();
			//Permutation phase
			if (i < (int)log2((float)n)) {
				sa[d] = aL;
				sb[d] = bL;
				sc[d] = cL;
				sy[d] = yL;
				sl[d] = (int)lL;
				__syncthreads();
			}
		}

		sy[(int)tidx] = yL / bL;
		__syncthreads();
		y[gb_index_x * n + sl[tidx]] = sy[tidx];
	}

}