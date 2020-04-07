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
    W = np.zeros(2**n+1) #Initializing W
    Z = np.random.randn(2**n) #Getting a Normal distribution
    DeltaT = T/(2.0**n) #Dividing the time space
    #Using the formula calculated
    for i in np.arange(2**n):
        W[i+1] = W[i] + Z[i]*np.sqrt(DeltaT)
    return W

def payoff():
    T = np.arange(0,days_to_maturity,days_to_maturity/(2**n+1))
    S = S_0 * np.exp(((mi - sigma**2/2)*T) + sigma*W)
    Indicator = 0
    for i in pre_scheduled:
        if S[i*n] < B:
            Indicator += 1
    if Indicator < P1 or Indicator > P2:
        return 0
    if S[-1] <= K:
        return 0
    return np.exp(r* days_to_maturity) *  (S[-1] - K)
    

N = 1000
n = 10
days_to_maturity = 252
pre_scheduled = np.array([1,2,10,20,50])
K = 20
B = 10
S_0 = 5
P1 = 1
P2 = 6
mi = 0.01
r = 0.001
sigma = 0.0001

Price = 0

for i in range(N):
    W = ForwardBrownian(n, days_to_maturity)
    p = payoff()
    
    Price += p

print("The price at time 0 is = " + str(Price/N))