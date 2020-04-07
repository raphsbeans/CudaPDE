#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../PDE_Solver/TridiagKernel.cuh"

using namespace std;

int NB = 512;
int Dim = 5;

__host__ void read_txt(float* array, string filename) {
	ifstream inFile;
	inFile.open(filename);
	printf("Reading: ");
	cout << filename << '\n';

	if (inFile.is_open()) {
		for (int i = 0; i < NB * Dim; ++i) {
			inFile >> array[i];
		}
	}
}
__host__ void print_array(float* array, string name) {
	cout << name << '\n';
	for (int i = 0; i < NB * Dim; i++) {
		printf("%.3e, ", array[i]);
	}
	cout << '\n';
}

int main() {
	float* a, * b, * c, * x, * y;
	float* y1, * y2, * d_y1, * d_y2;
	float* d_a, * d_b, * d_c;
	int ok_pcr = 1;
	int ok_thomas = 1;

	string str = "python .\\stress_test.py ";
	str = str + to_string(Dim) + " " + to_string(NB);

	const char* command = str.c_str();
	system(command);

	a = (float*)calloc(NB * Dim, sizeof(float));
	b = (float*)calloc(NB * Dim, sizeof(float));
	c = (float*)calloc(NB * Dim, sizeof(float));
	x = (float*)calloc(NB * Dim, sizeof(float));
	y = (float*)calloc(NB * Dim, sizeof(float));
	y1 = (float*)calloc(NB * Dim, sizeof(float));
	y2 = (float*)calloc(NB * Dim, sizeof(float));

	read_txt(a, "a.txt");
	read_txt(b, "b.txt");
	read_txt(c, "c.txt");
	read_txt(x, "x.txt");
	read_txt(y, "y.txt");

	printf("Executing a text with %d different system and %d dimension", NB, Dim);

	cudaMalloc(&d_a, NB * Dim * sizeof(float));
	cudaMalloc(&d_b, NB * Dim * sizeof(float));
	cudaMalloc(&d_c, NB * Dim * sizeof(float));
	cudaMalloc(&d_y1, NB * Dim * sizeof(float));
	cudaMalloc(&d_y2, NB * Dim * sizeof(float));

	cudaMemcpy(d_a, a, NB * Dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, NB * Dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, NB * Dim * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_y1, y, NB * Dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y2, y, NB * Dim * sizeof(float), cudaMemcpyHostToDevice);

	//int minTB = (Dim > 255) + 4 * (Dim > 63 && Dim < 256) + 16 * (Dim > 15 && Dim < 64) + 64 * (Dim > 3 && Dim < 16);

	TridiagSolverImpl::pcr(NB, Dim, d_a, d_b, d_c, d_y1);
	TridiagSolverImpl::thomasGPU(NB, Dim, d_a, d_b, d_c, d_y2);

	//pcr_k << <NB / minTB, minTB* Dim, 5 * minTB * Dim * sizeof(float) >> > (d_a, d_b, d_c, d_y1, Dim);
	//thom_k << <NB / (minTB), minTB >> > (d_a, d_b, d_c, d_y2, Dim);

	cudaMemcpy(y1, d_y1, NB * Dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2, d_y2, NB * Dim * sizeof(float), cudaMemcpyDeviceToHost);

	float limit = 0.001f;
	for (int k = 0; k < NB * Dim; ++k)
	{
		if (abs(x[k] - y1[k]) > limit) {
			printf("\n ERROR in PCR for real value = %.3e, pcr = %.3f, in position k = %d", x[k], y1[k], k);
			ok_pcr = 0;
		}
		if (abs(x[k] - y2[k]) > limit) {
			printf("\n ERROR in Thomas for real value = %.3e, thomas = %.3f, in position k = %d", x[k], y2[k], k);
			ok_thomas = 0;
		}
		//if (!ok_pcr & !ok_thomas)
			//break;
	}

	if (ok_thomas) printf("\n No problems with Thomas");
	else printf("\n WARNING!!! PROBLEM WITH THOMAS");

	if (ok_pcr) printf("\n No problems with PCR");
	else printf("\n WARNING!!! PROBLEM WITH PCR");

}
