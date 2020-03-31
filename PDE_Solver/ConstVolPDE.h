#pragma once

#include "constVolKernel.cuh"

class ConstVolPDE
{
public:
	ConstVolPDE(float volatility, float rate);

	void setMaturityInDays(int N) { this->maturity = N; }

	// set the dimension of the grids
	void setGridDimension(int spotSize, int stateSize) {
		this->spotDim = spotSize;
		this->stateDim = stateSize;
	}

	// nbStdDev defines the size of the spot grid and timestep is given in days
	void setNumericalParameters(int nbStdDev, float timestep) {
		this->nbStdDev = nbStdDev;
		this->timestep = timestep;
	}

	void buildSpaceGrid();
	void buildTimeGrid();


private:
	// model parameters
	float vol;
	float rate;

	// grid dimensions
	int stateDim;
	int spotDim;

	int maturity;

	// numerical parameters
	int nbStdDev;
	float timestep; // in days (it must be a natural number or have of the form 1/r where r is a natural)

	// grids
	float* spaceGrid;
	float* timeGrid;
	
};

