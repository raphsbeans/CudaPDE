#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define EPS (0.0000001f)

#define M (4 * 1048576)
__device__ int gl[M];

//===================================
std::vector<int> obs_dates;
const size_t spot_dim = 50;
const size_t state_dim = 50;
const float B = 110;
const float rate = 0.05;
const float sigma = 0.2;
const float mu = rate - (sigma * sigma) / 2.0;
const size_t N = 200;
const float T = N / 365;
const float dt = T / N;
const int P1 = 3;
const int P2 = 8;
const float dx = 2 * 4 * sigma * sqrtf(T) / spot_dim;
//===================================


///////////////////////////////////////////////////////////////////////
// Thomas resolution for tridiagonal symmetric matrices
///////////////////////////////////////////////////////////////////////
__global__ void thom_sym_k(float *a, float *b, float *y, int n) {

	// The global memory access index
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int j;
	float bet, *gam;
	gam = (float *) &gl[idx*n];
	if (fabsf(b[idx*n]) < EPS) printf("Error 1 in tridag");
	//If this happens then you should rewrite your equations as a set of order N − 1
	bet=b[idx*n];
	y[idx*n]=y[idx*n]/(bet);
	for (j=1;j<n;j++) { //Decomposition and forward substitution.
		gam[j]=a[idx*n+j]/bet;
		bet=b[idx*n+j]-a[idx*n+j]*gam[j];
	if (fabsf(bet) < EPS) printf("Error 2 in tridag"); //Algorithm fails;
		y[idx*n+j]=(y[idx*n+j]-a[idx*n+j]*y[idx*n+j-1])/bet;
	}
	for (j=(n-2);j>=0;j--)
		y[idx*n+j] -= gam[j+1]*y[idx*n+j+1]; //Backsubstitution.
}

///////////////////////////////////////////////////////////////////////
// Thomas resolution for tridiagonal matrices
///////////////////////////////////////////////////////////////////////
__global__ void thom_k(float* a, float* b, float* c, float* y, int n) {
	// The global memory access index
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float bet, *gam;
	gam = (float*) &gl[idx * n];
	
	if (fabs(b[idx * n]) < EPS) 
		printf("Error 1 in tridiag");
	
	// If this happens then you should rewrite your equations as a set of order N - 1
	bet = b[idx * n];
	y[idx * n] = y[idx * n] / bet;
	for (int j = 1; j < n; ++j) {
		gam[j] = c[idx * n + j] / bet;
		bet = b[idx * n + j] - a[idx * n + j] * gam[j];
		
		if (fabsf(bet) < EPS) 
			printf("Error 2 in tridag"); //Algorithm fails;
		
		y[idx * n + j] = (y[idx * n + j] - a[idx * n + j] * y[idx * n + j - 1]) / bet;
	}
	
	for (int j = n - 2; j >= 0; --j) {
		y[idx * n + j] -= gam[j + 1] * y[idx * n + j + 1]; // Backsubstitution
	}

}

///////////////////////////////////////////////////////////////////////
// Parallel cyclic reduction for tridiagonal symmetric matrices
///////////////////////////////////////////////////////////////////////
__global__ void pcr_sym_k(float *a, float *b, float *y, int n) {
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n;
	// Identifies the data concerned by the computations
	int Qt = (threadIdx.x - tidx) / n;
	// The global memory access index
	int gb_index_x = Qt + blockIdx.x*(blockDim.x / n);
	// Local integers
	int i, nt, lL, d, tL, tR;
	// Local floats
	float aL, bL, yL, aLp, bLp, yLp;
	// Shared memory
	extern __shared__ float sAds[];


	nt = 4*Qt*n;
	d = (n / 2 + (n % 2))*(tidx % 2) + (int)tidx / 2;
	float* sa = (float*)&sAds[nt];
	float* sb = (float*)&sa[n];
	float* sy = (float*)&sb[n];	
	int* sl = (int*)&sy[n];

	sa[tidx] = a[gb_index_x*n + tidx];
	sb[tidx] = b[gb_index_x*n + tidx];
	sy[tidx] = y[gb_index_x*n + tidx];
	sl[tidx] = tidx;
	__syncthreads();

	//Left/Right indices of the reduction
	tL = tidx - 1;
	if (tL < 0) tL = 0;
	tR = tidx + 1;
	if (tR >= n) tR = 0;

	for (i = 0; i < (int)log2((float)n) + 1; i++){
		lL = (int)sl[tidx];
		aL = sa[tidx];
		bL = sb[tidx];
		yL = sy[tidx];

		bLp = sb[tL];
		//Reduction phase
		if (fabsf(aL) > EPS){
			aLp = sa[tL];
			yLp = sy[tL];
			//bL = b[tidx] - a[tidx]*c[tidx]/b[tidx-1];
			bL -= aL*aL / bLp;
			//yL = y[tidx] - a[tidx]*y[tidx-1]/b[tidx-1];
			yL -= aL*yLp / bLp;
			//aL = -a[tidx]*a[tidx-1]/b[tidx-1];
			aL = -aL*aLp / bLp;
		}

		aLp = sa[tR];
		bLp = sb[tR];
		if (fabsf(aLp) > EPS){
			yLp = sy[tR];
			//bL -= c[tidx+1]*a[tidx+1]/b[tidx+1];
			bL -= aLp*aLp / bLp;
			//yL -= c[tidx+1]*y[tidx+1]/b[tidx+1];
			yL -= aLp*yLp / bLp;
		}
		__syncthreads();
		//Permutation phase
		if (i < (int)log2((float)n)){
			sa[d] = aL;
			sb[d] = bL;
			sy[d] = yL;
			sl[d] = (int)lL;
			__syncthreads();
		}
	}

	sy[(int)tidx] =yL / bL;
	__syncthreads();
	y[gb_index_x*n + sl[tidx]] = sy[tidx];
}


///////////////////////////////////////////////////////////////////////
// Parallel cyclic reduction for tridiagonal matrices
///////////////////////////////////////////////////////////////////////
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

	nt = 4 * Qt * n;
	d = (n / 2 + (n % 2)) * (tidx % 2) + (int)tidx / 2;
	float* sa = (float*)& sAds[nt];
	float* sb = (float*)& sa[n];
	float* sc = (float*)& sb[n];
	float* sy = (float*)& sc[n];
	int* sl = (int*)& sy[n];

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
			bL = b[tidx] - a[tidx]*c[tidx]/b[tidx-1];
			//bL -= aL * aL / bLp;
			yL = y[tidx] - a[tidx]*y[tidx-1]/b[tidx-1];
			//yL -= aL * yLp / bLp;
			aL = -a[tidx]*a[tidx-1]/b[tidx-1];
			//aL = -aL * aLp / bLp;
		}

		aLp = sa[tR];
		bLp = sb[tR];
		if (fabsf(aLp) > EPS) {
			yLp = sy[tR];
			bL -= c[tidx+1]*a[tidx+1]/b[tidx+1];
			//bL -= aLp * aLp / bLp;
			yL -= c[tidx+1]*y[tidx+1]/b[tidx+1];
			//yL -= aLp * yLp / bLp;
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

// Produces tridiagonal symmetric diagonally dominant matrices 
__global__ void Tri_k(float *S1, float *D, float *S2, float norm, int i, 
						   int n, int L)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n;
	// Identifies the data concerned by the computations
	int Qt = (threadIdx.x - tidx) / n;
	// The global memory access index
	int gb_index_x = Qt + blockIdx.x*(blockDim.x / n);

	if(gb_index_x*n + tidx + i < L){
		D[gb_index_x * n + tidx + i] = ((float)tidx + 1.0f) / (norm);
		if (tidx > 0) {
			S1[gb_index_x * n + tidx + i] = ((float)tidx + 1.0f) / (norm * 6);
			S2[gb_index_x * n + tidx + i] = ((float)tidx + 0.5f) / (norm * 6);
		}
		else {
			S1[gb_index_x * n + tidx + i] = 0.0f;
			S2[gb_index_x * n + tidx + i] = 0.0f;
		}
	}
}

__global__ void back_step_k(float* payoff, float sigma, float r)
{
	float q_u = -(sigma * sigma * dt) / (4 * dx * dx) - mu * dt / (4 * dx);
	float q_m = 1 + (sigma * sigma * dt) / (2 * dx * dx);
	float q_d = mu * dt / (4 * dx) - (sigma * sigma * dt) / (4 * dx * dx);

	float p_u = (sigma * sigma * dt) / (4 * dx * dx) + mu * dt / (4 * dx);
	float p_m = 1 - (sigma * sigma * dt) / (2 * dx * dx);
	float p_d = (sigma * sigma * dt) / (4 * dx * dx) - mu * dt / (4 * dx);

	float *D = new float[]


}

int main(){

	int i, j;

	// The rank of the matrix
	int Dim = 64;
	// The number of blocks
	int NB = M/Dim;
	// The number of matrices to invert
	int size = NB;

	// The diagonal elements
	float *D, *DGPU;
	// The subdiagonal elements
	float *S1, *S2, *SGPU1, *SGPU2;
	// The system vector
	float *Y1, * Y2, *YGPU1, * YGPU2;

	float TimerV;					// GPU timer instructions
	cudaEvent_t start, stop;		// GPU timer instructions
	cudaEventCreate(&start);		// GPU timer instructions
	cudaEventCreate(&stop);			// GPU timer instructions

	// Memory allocation
	D = (float*)calloc(size * Dim, sizeof(float));
	S1 = (float*)calloc(size * Dim, sizeof(float));
	S2 = (float*)calloc(size * Dim, sizeof(float));
	Y1 = (float*)calloc(size * Dim, sizeof(float));
	Y2 = (float*)calloc(size * Dim, sizeof(float));
	cudaMalloc(&DGPU, size * Dim * sizeof(float));
	cudaMalloc(&SGPU1, size * Dim * sizeof(float));
	cudaMalloc(&SGPU2, size * Dim * sizeof(float));
	cudaMalloc(&YGPU1, size * Dim * sizeof(float));
	cudaMalloc(&YGPU2, size * Dim * sizeof(float));

	// Tridiagonal elements
	int HM = M/(NB*Dim); // =~ 1
	for (i=0; i*Dim*NB<M; i++){
		Tri_k <<<NB,HM*Dim>>>(SGPU1, DGPU, SGPU2, 10.0f, i*Dim*NB, Dim, 
										  Dim*NB);
	}
	cudaMemcpy(D, DGPU, size*Dim*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(S1, SGPU1, size * Dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(S2, SGPU2, size*Dim*sizeof(float), cudaMemcpyDeviceToHost);

	// Second member
	for (i=0; i<size; i++){
		for (j=0; j<Dim; j++){
			Y1[j + i * Dim] = 0.5f * j;
			Y2[j + i * Dim] = 0.5f * j;
		}
	}
	cudaMemcpy(YGPU1, Y1, size * Dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(YGPU2, Y2, size * Dim * sizeof(float), cudaMemcpyHostToDevice);

	// Resolution part
	cudaEventRecord(start,0);

	// The minimum number of threads per block for PCR
	int minTB = (Dim>255) + 4*(Dim>63 && Dim<256) + 16*(Dim>15 && Dim<64) + 64*(Dim>3 && Dim<16);
	printf("minTB: %i \n", minTB);
	pcr_k<<<NB/minTB, Dim*minTB, 4*minTB*Dim*sizeof(float)>>>(SGPU1, DGPU, SGPU2, YGPU1, Dim);
	thom_k<<<NB/256,256>>>(SGPU1, DGPU, SGPU2, YGPU2, Dim);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&TimerV,start, stop);

	cudaMemcpy(Y1, YGPU1, size * Dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Y2, YGPU2, size * Dim * sizeof(float), cudaMemcpyDeviceToHost);
	
	double diff = 0.0;
	for (size_t k = 0; k < 100/*size * Dim*/; ++k)
	{
		printf("\n%.3e   %.3e", Y1[k], Y2[k]);
		//diff += (Y1[k] - Y2[k]) * (Y1[k] - Y2[k]);
	}
	
	//printf("\nASE = %.5e", diff / (size * Dim));

	/*
	for (i=0; i<size; i++){
	        if(i==573){
			printf("\n\n");
			for (j=0; j<Dim; j++){
				printf("%.5e, ",Y[j+i*Dim]);
			}
		} 
	}
	*/


	printf("Execution time: %f ms\n", TimerV);

	// Memory free for other arrays
	free(D);
	cudaFree(DGPU);
	free(S1);
	free(S2);
	cudaFree(SGPU1);
	cudaFree(SGPU2);
	free(Y1);
	free(Y2);
	cudaFree(YGPU1);
	cudaFree(YGPU2);

	cudaEventDestroy(start);		// GPU timer instructions
	cudaEventDestroy(stop);			// GPU timer instructions

	return 0;
}
