#pragma once
#include "Payoff.h"
class BulletOption : public Payoff {
public:
	BulletOption(float Strike, size_t daysToMaturity, float S0, float sMin, float sMax, float P1, float P2, float B, 
		size_t * pre_schedule, size_t spotGridSize, size_t timeGridSize, size_t stateSize);
	void initPayoff(float* payoff);
	void interStep(float* payoff, size_t timeIdx);
	float applySolution(float* sol);
	void getSpotRange(float& sMin, float& sMax);
	size_t getGridSizes(size_t& spotGridSize, size_t& timeGridSize, size_t& stateSize);

private:
	size_t daysToMaturity = 0; //Only working days
	size_t spotGridSize = 0; 
	size_t timeGridSize = 0; 
	size_t stateSize = 0;

	float S0 = 0.0f; //Initial Price
	float Strike = 0.0f; //Strike
	float P1 = 0.0f; //Minimum Times heating B
	float P2 = 0.0f; //Maximum Times heating B
	float B = 0.0f; //The limit 
	float sMin = 0.0f; //Minimum value in the spot grid
	float sMax = 0.0f; //Maximum value in the spot grid
	float dx = 0.0f; //infinitesimal value for the spot grid
	float dt = 0.0f; //infinitesimal -> grid time

	size_t* pre_schedule;
};

