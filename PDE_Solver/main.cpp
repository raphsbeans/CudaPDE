#include "BulletOption.h"
#include "ConstVolPDESolver.h"

#include <vector>
#include <iostream>

int main() {

	// Build the option
	float spot = 100;
	float strike = 100;
	float barrier = 105;
	size_t daysToMaturity = 20;
	size_t P1 = 3;
	size_t P2 = 8;	
	
	std::vector<size_t> schedule(10);
	for (size_t i = 0; i < schedule.size(); i++) {
		schedule[i] = 2 * i + 1;
	}
	
	BulletOption* bullet = new BulletOption(strike, daysToMaturity, barrier, P1, P2, schedule);
	bullet->setCurrPosition(spot, 0);

	// Build the solver
	float volatility = 0.2;
	float rate = 0;

	ConstVolPDESolver pdeSolver(volatility, rate);
	pdeSolver.setPayoff(bullet);
	pdeSolver.solveGPU();

	std::cout << bullet->currentPrice << std::endl;

	delete bullet;
	return 0;
}