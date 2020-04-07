#include "BulletOption.h"
#include  <math.h>

BulletOption::BulletOption (float Strike, size_t daysToMaturity, float S0, float sMin, float sMax, float P1, float P2, float B,
    size_t* pre_schedule, size_t spotGridSize, size_t timeGridSize, size_t stateSize)
    :Strike(Strike), daysToMaturity(daysToMaturity), P1(P1), P2(P2), B(B), pre_schedule(pre_schedule), 
    spotGridSize(spotGridSize), timeGridSize(timeGridSize), stateSize(stateSize), sMin(sMin), sMax(sMax), S0(S0){
    dx = (log(sMax) - log(sMin)) / spotGridSize;
    dt = daysToMaturity / timeGridSize;
}

void BulletOption::initPayoff(float* payoff) {
    double S;
    for (size_t j = 0; j < stateSize; j++) {
        for (size_t i = 0; i < spotGridSize; i++) {
            S = exp(sMin + (double)i * dx);
            payoff[i + j * spotGridSize] = (S - Strike > 0 &&  j > P1 && j < P2) ? (S - Strike) : 0;
        }
    }
}

void BulletOption::interStep(float* payoff, size_t timeIdx) {

}

float BulletOption::applySolution(float* sol) {
    size_t left = (size_t)floor((S0 - sMin) / dx);
    size_t right = left + 1;
    return sol[left] + sol[right] * (S0 - sol[left]) / (sol[right] - sol[left]);

}

size_t BulletOption::getGridSizes(size_t& spotGridSize, size_t& timeGridSize, size_t& stateSize) {
    spotGridSize = this->spotGridSize;
    timeGridSize = this->timeGridSize;
    stateSize = this->stateSize;
    return 1;
}

void BulletOption::getSpotRange(float& sMin, float& sMax) {
    sMin = this->sMin;
    sMax = this->sMax;
}