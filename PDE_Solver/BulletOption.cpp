#include "BulletOption.h"
#include "BulletOptionKernel.cuh"
#include  <math.h>

BulletOption::BulletOption( float strike, 
                            size_t daysToMaturity, 
                            float barrier, 
                            size_t P1, 
                            size_t P2, 
                            const std::vector<size_t>& scheduleDays)
    :   strike(strike), P1(P1), P2(P2), barrier(barrier),currSpot(0), currState(0), 
        timeGridSize(daysToMaturity + 1), isSchedule(daysToMaturity + 1, false), scheduleCounter(0),
        Payoff(daysToMaturity), currentPrice(0)
{
    spotGridSize = 128;
    stateSize = scheduleDays.size();

    Smax = 3 * strike;
    Smin = strike / 3.0;

    dx = (logf(Smax) - logf(Smin)) / (spotGridSize - 1);
    dt = getTimeToMaturity() / (spotGridSize - 1);

    for (size_t i = 0; i < scheduleDays.size(); i++) {
        isSchedule[scheduleDays[i]] = true;
    }
}

void BulletOption::setCurrPosition(float currSpot, size_t currState = 0)
{
    this->currSpot = currSpot;
    this->currState = currState;
}

void BulletOption::initPayoff(float* payoff, ParamsLocation paramLocation) 
{
    if (paramLocation == ParamsLocation::HOST) {
        for (size_t j = 0; j < stateSize; j++) {
            for (size_t i = 0; i < spotGridSize; i++) {
                // !! state grid value is equal to state index !!
                double S = Smin * exp((double)i * dx);
                payoff[i + j * spotGridSize] = (S - strike > 0 && j > P1 && j < P2) ? (S - strike) : 0;
            }
        }
    }
    else {
        BulletOptionKernel::initPayoffGPU(spotGridSize, stateSize, payoff, dx, Smin, strike, P1, P2);
        cudaDeviceSynchronize();

        BulletOptionKernel::boundaryConditionGPU(spotGridSize, stateSize, payoff, strike);
        cudaDeviceSynchronize();
    }
}

void BulletOption::interStep(float* payoff, size_t timeIdx) 
{
    // apply boundary condition
    BulletOptionKernel::boundaryConditionGPU(spotGridSize, stateSize, payoff, strike);
    cudaDeviceSynchronize();

    /*/
    if (isSchedule[timeIdx]) {
        BulletOptionKernel::interStepGPU(spotGridSize, stateSize, payoff, scheduleCounter, dx, Smin, P1, P2, barrier);
        cudaDeviceSynchronize();

        scheduleCounter++;
    } 
    */
}

float BulletOption::applySolution(float* sol_d) {

    dumpMatrix(spotGridSize, stateSize, sol_d, "C:\\Users\\erik\\Desktop\\sol.csv");


    float* sol = new float[spotGridSize * stateSize];
    cudaMemcpy(sol, sol_d, spotGridSize * stateSize * sizeof(float), cudaMemcpyDeviceToHost);

    size_t idx_left = (size_t)floor((logf(currSpot) - logf(Smin)) / dx);
    size_t idx_right = idx_left + 1;
    
    float x = logf(currSpot);
    float x1 = logf(Smin) + idx_left * dx;
    float x2 = logf(Smin) + idx_right * dx;
    float y1 = sol[idx_left + currState * spotGridSize];
    float y2 = sol[idx_right + currState * spotGridSize];   
    
    currentPrice = (y1 * (x2 - x) + y2 * (x - x1)) / dx;
    return currentPrice;
}

void BulletOption::getGridSizes(size_t& spotGridSize, size_t& timeGridSize, size_t& stateSize) {
    spotGridSize = this->spotGridSize;
    timeGridSize = this->timeGridSize;
    stateSize = this->stateSize;
}

void BulletOption::getSpotRange(float& sMin, float& sMax) {
    sMin = this->Smin;
    sMax = this->Smax;
}