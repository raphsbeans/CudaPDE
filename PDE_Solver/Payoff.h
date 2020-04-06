#pragma once
class Payoff
{
public:
	virtual void initPayoff(float* payoff) = 0;
	virtual void interStep(float* payoff, size_t timeIdx) = 0;

	virtual float applySolution(float* sol) = 0;

	virtual void getSpotRange(float& sMin, float& sMax) = 0;
	virtual size_t getGridSizes(size_t& spotGridSize, size_t& timeGridSize, size_t& stateSize) = 0;

	// return time to maturity in years
	virtual float getTimeToMaturity() { return daysToMaturity / 252.0; };

private:
	size_t daysToMaturity; // only business days
};

