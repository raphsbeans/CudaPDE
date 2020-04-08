#pragma once

#include "global_definitions.h"

class Payoff
{
public:
	Payoff(size_t daysToMaturity) : daysToMaturity(daysToMaturity) {}

	virtual void initPayoff(float* payoff, ParamsLocation paramsLocation) = 0;
	virtual void interStep(float* payoff, size_t timeIdx) = 0;

	virtual float applySolution(float* sol) = 0;

	virtual void getSpotRange(float& sMin, float& sMax) = 0;
	virtual void getGridSizes(size_t& spotGridSize, size_t& timeGridSize, size_t& stateSize) = 0;

	// return time to maturity in years
	virtual float getTimeToMaturity() { return daysToMaturity / 252.0; };

private:
	size_t daysToMaturity; // only business days
};

