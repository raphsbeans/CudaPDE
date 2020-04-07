#pragma once
#include "Payoff.h"
class BulletOption : public Payoff {
public:
	BulletOption(float Strike, size_t daysToMaturity, float P1, float P2, float B, size_t * pre_schedule);
	void initPayoff(float* payoff);
	void interStep(float* payoff, size_t timeIdx);
	float applySolution(float* sol);

private:
	size_t daysToMaturity = 0;
	size_t ind_counter = 0;
	float Strike = 0.0f;
	float P1 = 0.0f;
	float P2 = 0.0f;
	float B = 0.0;
	size_t* pre_schedule;
};

