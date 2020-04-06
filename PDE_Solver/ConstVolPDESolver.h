#pragma once

#include "Payoff.h"

class ConstVolPDESolver
{
public:
	ConstVolPDESolver(float volatility, float rate);

	void setPayoff(Payoff* payoff);

	void solveGPU();

private:

	Payoff* pPayoff;

	// Model Parameters
	float mVolatility, mRate;

	// Numerical Parameters
	size_t mSpotGridSize;
	size_t mTimeGridSize;
	size_t mStateGridSize;
};

