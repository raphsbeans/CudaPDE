#pragma once
#include "TridiagKernel.cuh"

class TridiagSolver {
public:

	enum class Method {
		THOMAS,
		PCR
	};

	enum class ParamsLocation {
		DEVICE,
		HOST
	};

	TridiagSolver(int dim, int size, Method type);

	void TridiagSolver::setParamsLocation(ParamsLocation location);

	void TridiagSolver::solve(float* a, float* b, float* c, float* y);

private:

	int size;
	int dim;

	Method method;
	ParamsLocation paramsLocation;

	void TridiagSolver::solveThomas(float* d_a, float* d_b, float* d_c, float* d_y);
	void TridiagSolver::solvePCR(float* d_a, float* d_b, float* d_c, float* d_y);

};