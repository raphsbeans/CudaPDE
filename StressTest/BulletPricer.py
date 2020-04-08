# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 03:30:40 2020

@author: raphaelfeijao
"""
import numpy as np

def ForwardBrownian (n, T):
    '''
    This function will use a forward simulation to create Brownian Motion, of time space T/(2**n)
    '''
    W = np.zeros(n+1) #Initializing W
    Z = np.random.randn(n) #Getting a Normal distribution
    DeltaT = T/(n) #Dividing the time space
    #Using the formula calculated
    for i in np.arange(n):
        W[i+1] = W[i] + Z[i]*np.sqrt(DeltaT)
    return W

def payoff():
    T = np.arange(0,days_to_maturity, dt)
    S = S_0 * np.exp(((mi - sigma**2/2)*T) + sigma*W)
    Indicator = 0
    for i in pre_scheduled:
        ind = int(i/dt)
        if S[ind] < B:            
            Indicator += 1
    
    if Indicator < P1 or Indicator > P2:
        return 0
    if S[-1] <= K:
        return 0
    return np.exp(-r* days_to_maturity) *  (S[-1] - K)


N = 10000
n = 100
days_to_maturity = 1
dt = days_to_maturity/(n+1)
pre_scheduled = np.arange(0,days_to_maturity, dt)
K = 100
B = 120
S_0 = 100
P1 = 10
P2 = 49
mi = 0.1
r = 0.1
sigma = 0.2

Price = 0

for i in range(N):
    W = ForwardBrownian(n, days_to_maturity)
    p = payoff()

    Price += p

print("The price at time 0 is = " + str(Price/N))
