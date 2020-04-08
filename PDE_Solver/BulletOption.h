#pragma once
#include "Payoff.h"
#include <vector>

class BulletOption : public Payoff {
public:
	BulletOption(float strike, size_t daysToMaturity, float barrier, size_t P1, size_t P2, const std::vector<size_t>& scheduleDays);
	
	void setCurrPosition(float currSpot, size_t currState);

	virtual void initPayoff(float* payoff, ParamsLocation paramsLocation = ParamsLocation::DEVICE);
	virtual void interStep(float* payoff, size_t timeIdx);
	
	virtual float applySolution(float* sol);
	
	virtual void getSpotRange(float& sMin, float& sMax);
	virtual void getGridSizes(size_t& spotGridSize, size_t& timeGridSize, size_t& stateSize);

	float currentPrice;

private:
	// current position
	float currSpot;	// Current Spot
	size_t currState; // Current State Variable

	// Payoff related
	size_t daysToMaturity; //Only working days
	float strike; //Strike
	float P1; //Minimum Times heating Barrier
	float P2; //Maximum Times heating Barrier
	float barrier;
	std::vector<bool> isSchedule;
	size_t scheduleCounter;

	// Grid related
	size_t spotGridSize;
	size_t timeGridSize;
	size_t stateSize;

	float Smin; //Minimum value in the spot grid
	float Smax; //Maximum value in the spot grid
	float dx; //infinitesimal value for the spot grid
	float dt; //infinitesimal -> grid time	
};

