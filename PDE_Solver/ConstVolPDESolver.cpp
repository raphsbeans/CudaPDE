#include "ConstVolPDESolver.h"
#include "cuda_runtime.h"
#include <cmath>
#include "constVolKernel.cuh"
#include "TridiagSolver.h"

ConstVolPDESolver::ConstVolPDESolver(float volatility, float rate)
	: mVolatility(volatility), mRate(rate), pPayoff(nullptr), mSpotGridSize(0), mTimeGridSize(0), mStateGridSize(0)
{
}

void ConstVolPDESolver::setPayoff(Payoff* payoff)
{
	pPayoff = payoff;
	pPayoff->getGridSizes(mSpotGridSize, mTimeGridSize, mStateGridSize);
}

void ConstVolPDESolver::solveGPU()
{
	// get grid data ===============================================================
	float maturtiy = pPayoff->getTimeToMaturity();
	float dt = maturtiy / mTimeGridSize;
	
	float Smin, Smax;
	pPayoff->getSpotRange(Smin, Smax);
	float dx = (std::log(Smax) - std::log(Smin)) / mSpotGridSize;

	// create tridiagonal matrices =================================================
	float p_u, p_m, p_d;
	float q_u, q_m, q_d;
	float temp1, temp2;
	
	temp1 = ((double) mVolatility * mVolatility) * dt / (4.0 * dx * dx);
	temp2 = (mRate - (double) mVolatility * mVolatility / 2.0) * dt / (4.0 * dx);

	p_u = temp1 + temp2;
	q_u = -p_u;
	p_m = 1 + 2 * temp1;
	q_m = 1 - 2 * temp1;
	p_d = temp2 - temp1;
	q_d = -p_d;

	// Initialized Tridiagonal Solver
	TridiagSolver tridiagSolver(mSpotGridSize, mStateGridSize, TridiagSolver::Method::PCR);
	tridiagSolver.setParamsLocation(TridiagSolver::ParamsLocation::DEVICE);

	// initalize payoff on GPU =====================================================
	float* payoff = nullptr;
	cudaMalloc(&pPayoff, mSpotGridSize * mStateGridSize * sizeof(float));
	pPayoff->initPayoff(payoff);

	for (int i = mTimeGridSize; i >= 0; i--) {
		// payoff = P * payoff
		ConstVolKernel::tridiag_x_matrix_GPU(mSpotGridSize, mStateGridSize, p_d, p_m, p_u, payoff);

		// payoff = Q^-1 * payoff
		tridiagSolver.solve(q_d, q_m, q_u, payoff);

		// Call interstep payoff
		pPayoff->interStep(payoff, i);
	}

	pPayoff->applySolution(payoff);
	cudaFree(payoff);
}
