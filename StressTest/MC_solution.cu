/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/
#include "rng.h"


// Generate uniformly distributed random variables
__device__ void CMRG_d(int *a0, int *a1, int *a2, int *a3, int *a4, 
			     int *a5, float *g0, float *g1, int nb){

 const int m1 = 2147483647;// Requested for the simulation
 const int m2 = 2145483479;// Requested for the simulation
 int h, p12, p13, p21, p23, k, loc;// Requested local parameters

 for(k=0; k<nb; k++){
	 // First Component 
	 h = *a0/q13; 
	 p13 = a13*(h*q13-*a0)-h*r13;
	 h = *a1/q12; 
	 p12 = a12*(*a1-h*q12)-h*r12;

	 if (p13 < 0) {
	   p13 = p13 + m1;
	 }
	 if (p12 < 0) {
	   p12 = p12 + m1;
	 }
	 *a0 = *a1;
	 *a1 = *a2;
	 if( (p12 - p13) < 0){
	   *a2 = p12 - p13 + m1;  
	 } else {
	   *a2 = p12 - p13;
	 }
  
	 // Second Component 
	 h = *a3/q23; 
	 p23 = a23*(h*q23-*a3)-h*r23;
	 h = *a5/q21; 
	 p21 = a21*(*a5-h*q21)-h*r21;

	 if (p23 < 0){
	   p23 = p23 + m2;
	 }
	 if (p12 < 0){
	   p21 = p21 + m2;
	 }
	 *a3 = *a4;
	 *a4 = *a5;
	 if ( (p21 - p23) < 0) {
	   *a5 = p21 - p23 + m2;  
	 } else {
	   *a5 = p21 - p23;
	 }

	 // Combines the two MRGs
	 if(*a2 < *a5){
		loc = *a2 - *a5 + m1;
	 }else{loc = *a2 - *a5;} 

	 if(k){
		if(loc == 0){
			*g1 = Invmp*m1;
		}else{*g1 = Invmp*loc;}
	 }else{
		*g1 = 0.0f; 
		if(loc == 0){
			*g0 = Invmp*m1;
		}else{*g0 = Invmp*loc;}
	 }
  }
}

// Genrates Gaussian distribution from a uniform one (Box-Muller)
__device__ void BoxMuller_d(float *g0, float *g1){

  float loc;
  if (*g1 < 1.45e-6f){
    loc = sqrtf(-2.0f*logf(0.00001f))*cosf(*g0*2.0f*MoPI);
  } else {
    if (*g1 > 0.99999f){
      loc = 0.0f;
    } else {loc = sqrtf(-2.0f*logf(*g1))*cosf(*g0*2.0f*MoPI);}
  }
  *g0 = loc;
}

// Black & Scholes model
__device__ void BS_d(float *S2, float S1, float r0,
					 float sigma, float dt, float e){

  *S2 = S1*expf((r0-0.5f*sigma*sigma)*dt*dt + sigma*dt*e);
}
		
__device__ void CMRG_get_d(int *a0, int *a1, int *a2, 
							int *a3, int *a4, int *a5, 
							int *CMRG_In){
	*a0 = CMRG_In[0];
	*a1 = CMRG_In[1];
	*a2 = CMRG_In[2];
	*a3 = CMRG_In[3];
	*a4 = CMRG_In[4];
	*a5 = CMRG_In[5];
}

__global__ void MC_k(float x_0, float r0, // S pour Stock
					 float sigma, float dt, int P1, int P2,
					 float K, float *R1, float *R2, // U pour Rando
					 float B, int size, int M,	// P pour Ppt 
					 TabSeedCMRG_t *pt_cmrg){

   int idx = threadIdx.x + blockIdx.x*blockDim.x;

   int a0, a1, a2, a3, a4, a5, P;
   float U1, U2;

   CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx]);


   extern __shared__ float A[];

   float *Ssh, *R1sh, *R2sh;

   Ssh = A;
   R1sh = Ssh + 2*blockDim.x;
   R2sh = R1sh + blockDim.x;
	

   Ssh[threadIdx.x] = x_0;
   //P[idx] = 0;
   P = 0;	
   for (int k=1; k<=M; k++){
	   CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &U1, &U2, 2);

	   BoxMuller_d(&U1, &U2);
	   BS_d(Ssh+threadIdx.x+(k%2)*512, Ssh[threadIdx.x+((k+1)%2)*512], r0, sigma, dt, U1);

	   //P[idx] += (S[idx+(k%2)*size]<B);
	   P += (Ssh[threadIdx.x+(k%2)*512]<B);
		
   }

   //R1[idx] = expf(-r0*dt*dt*M)*fmaxf(0.0f, S[idx+(M%2)*size]-K)*((P[idx]<=P2)&&(P[idx]>=P1));
   R1sh[threadIdx.x] = expf(-r0*dt*dt*M)*fmaxf(0.0f, Ssh[threadIdx.x+(M%2)*512]-K)*((P<=P2)&&(P>=P1))/size;
   R2sh[threadIdx.x] = R1sh[threadIdx.x]*R1sh[threadIdx.x]*size;

   __syncthreads();

   int i = blockDim.x/2;

   while(i != 0){
	  if(threadIdx.x < i){
		R1sh[threadIdx.x] += R1sh[threadIdx.x + i]; 
		R2sh[threadIdx.x] += R2sh[threadIdx.x + i];
		//__syncthreads();
	  }
	  __syncthreads();
	  i /= 2;
   }

	
   if(threadIdx.x == 0){
	 atomicAdd(R1, R1sh[0]);
	 atomicAdd(R2, R2sh[0]);
   }
}


int main()
{

	float T = 1.0f;
	float K = 100.0f;
	float x_0 = 100.0f;
	float vol = 0.2f;
	float tau = 0.1f;
	float B = 120.0f;
	int M = 100;
	int P1 = 10;
	int P2 = 49;
	float dt = sqrtf(T/M);
	float sum = 0.0f;	
	float sum2 = 0.0f;
	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
	float *res1, *res2;//, *res1C, *res2C;
    //float *Rando, *Stock;				// Random variables and stock
	//int *Ppt;							// Crossing the threshold
	int Ntraj = 512*512;

	cudaMalloc(&res1, sizeof(float));
	cudaMalloc(&res2, sizeof(float));
	cudaMemset(res1, 0, sizeof(float));
	cudaMemset(res2, 0, sizeof(float));


   /************************************************************
	Step 1:
	-------
		Allocate appropriately Rando, Stock and Ppt

   *************************************************************/
	

	PostInitDataCMRG();

	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start,0);			// GPU timer instructions

	// Step 3:
	//--------
	// Uncomment after memory allocation and free
	 MC_k<<<512,512,4*512*sizeof(float)>>>(x_0, tau, vol, dt, 
										   P1, P2, K, res1, res2, 
										   B, Ntraj, M, CMRG);

	cudaEventRecord(stop,0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
			 start, stop);				// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	//cudaMemcpy(res1C, res1, Ntraj*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(res2C, res2, Ntraj*sizeof(float), cudaMemcpyDeviceToHost);

   /************************************************************
	Step 2:
	-------
		Free appropriately Rando, Stock and Ppt

   *************************************************************/


	/*for0(int i=0; i<Ntraj; i++){
		sum  += res1C[i]/Ntraj;
		sum2  += res2C[i]/Ntraj;
	}*/

	//free(res1C);
	//free(res2C);
	
	cudaMemcpy(&sum, res1, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&sum2, res2, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(res1);
	cudaFree(res2);	

	printf("The price is equal to %f\n", sum);

	printf("error associated to a confidence interval of 95%% = %f\n", 
		   1.96*sqrt((double)(1.0f/(Ntraj-1))*(Ntraj*sum2 - (sum*sum)))/
		   sqrt((double)Ntraj));
	printf("Execution time %f ms\n", Tim);

	FreeCMRG();
	
	return 0;
}


