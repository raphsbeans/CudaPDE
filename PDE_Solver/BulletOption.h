#pragma once
#include "Payoff.h"
class BulletOption : public Payoff {
public:
	BulletOption();
	void initPayoff(float* payoff);
	void interStep(float* payoff, size_t timeIdx);
	void applySolution(float* sol);
};

