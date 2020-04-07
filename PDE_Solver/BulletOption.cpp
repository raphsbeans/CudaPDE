#include "BulletOption.h"

BulletOption::BulletOption(float Strike, size_t daysToMaturity, float P1, float P2, float B, size_t* pre_schedule)
    :Strike(Strike), daysToMaturity(daysToMaturity), P1(P1), P2(P2), B(B), pre_schedule(pre_schedule){}

void BulletOption::initPayoff(float* payoff) {

}

void BulletOption::interStep(float* payoff, size_t timeIdx) {

}

float BulletOption::applySolution(float* sol) {

}

