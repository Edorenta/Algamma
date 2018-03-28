
'''
        .=====================================.
       /        Risk Indicator Library:        \
      {                 risklib                 }
       \              by Edorenta              /
        '====================================='
'''

import numpy as np
import numpy.random as rand

'''
      .------------------------.
      | RR I: PTF OPTIMIZATION |
      '------------------------'
'''
def lower_pm(returns, threshold, order):
    #This method returns a lower partial moment of the returns
    #Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    #Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    #Set the minimum of each to 0
    diff = diff.clip(min=0)
    #Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)
 
def higher_pm(returns, threshold, order):
    #This method returns a higher partial moment of the returns
    #Create an array he same length as returns containing the minimum return threshold
    threshold_array = np.empty(len(returns))
    threshold_array.fill(threshold)
    #Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    #Set the minimum of each to 0
    diff = diff.clip(min=0)
    #Return the sum of the different to the power of order
    return np.sum(diff ** order) / len(returns)

#Return SDEV function
def vol(returns):
    #Return volatility of returns
    return np.std(returns)

#Strategy Beta
def beta(returns, market):
    #Create a matrix of [returns, market]
    m = np.matrix([returns, market])
    #Return the covariance of m divided by the standard deviation of the market returns
    return np.cov(m)[0][1] / np.std(market)

def var(returns, alpha):
    #This method calculates the historical simulation var of the returns
    sorted_returns = np.sort(returns)
    #Calculate the index associated with alpha (for var 95%, alpha = 0,05)
    index = int(alpha * len(sorted_returns))
    #VaR should be positive
    return abs(sorted_returns[index])
 
def cvar(returns, alpha):
    #This method calculates the condition VaR of the returns
    sorted_returns = np.sort(returns)
    #Calculate the index associated with alpha (for cvar 95%, alpha = 0,05)
    index = int(alpha * len(sorted_returns))
    #Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    #Return the average VaR
    #CVaR should be positive
    return abs(sum_var / index)

'''   .-----------------------.
      |    RR II: DRAWDOWN    |
      '-----------------------'
'''
def dd(returns, ti): #local max drawdown
    #Returns the draw-down given time pRpiod ti
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - ti
    drawdown = float('+inf')
    #Find the maximum drawdown given ti
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    #Drawdown should be positive
    return abs(drawdown)

def max_dd(returns):
    #Returns the maximum draw-down for any ti in (0, T) where T is the length of the return sRpies
    max_DD = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_DD:
            max_DD = drawdown_i
    #Max draw-down should be positive
    return abs(max_DD)

def average_dd(returns, periods):
    #Returns the average maximum drawdown over n periods
    DDs = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        DDs.append(drawdown_i)
    DDs = sorted(DDs)
    total_dd = abs(DDs[0])
    for i in range(1, periods):
        total_dd += abs(DDs[i])
    return total_dd / periods

def average_dd_squared(returns, periods):
    #Returns the average maximum drawdown squared over n periods
    DDs = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        DDs.append(drawdown_i)
    DDs = sorted(DDs)
    total_dd = abs(DDs[0])
    for i in range(1, periods):
        total_dd += abs(DDs[i])
    return total_dd / periods

'''   .-----------------------.
      |    RR III: ADJUSTED   |
      '-----------------------'
'''
def treynor_ratio(R_p, returns, market, R_f): #R_f = risk free, set 0 by default for simplicity)
    return (R_p - R_f) / beta(returns, market)
 
def sharpe_ratio(R_p, returns, R_f):
    return (R_p - R_f) / vol(returns)
 
def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return np.mean(diff) / vol(diff)
 
def modigliani_ratio(R_p, returns, benchmark, R_f): #R_f = risk free, set 0 by default for simplicity)
    np_rf = np.empty(len(returns))
    np_rf.fill(R_f)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (R_p - R_f) * (vol(rdiff) / vol(bdiff)) + R_f

'''   .-----------------------.
      |    RR IV: MORE ADJ    |
      '-----------------------'
'''
def excess_var(R_p, returns, R_f, alpha):
    return (R_p - R_f) / var(returns, alpha)
 
def conditional_sharpe_ratio(R_p, returns, R_f, alpha):
    return (R_p - R_f) / cvar(returns, alpha)

def omega_ratio(R_p, returns, R_f, target=0):
    return (R_p - R_f) / lower_pm(returns, target, 1)

def sortino_ratio(R_p, returns, R_f, target=0):
    return (R_p - R_f) / math.sqrt(lower_pm(returns, target, 2))

def kappa_three_ratio(R_p, returns, R_f, target=0):
    return (R_p - R_f) / math.pow(lower_pm(returns, target, 3), float(1/3))

def gain_loss_ratio(returns, target=0):
    return higher_pm(returns, target, 1) / lower_pm(returns, target, 1)

def upside_potential_ratio(returns, target=0):
    return higher_pm(returns, target, 1) / math.sqrt(lower_pm(returns, target, 2))

def calmar_ratio(R_p, returns, R_f):
    return (R_p - R_f) / max_dd(returns)

def stRpling_ration(R_p, returns, R_f, periods):
    return (R_p - R_f) / average_dd(returns, periods)

def burke_ratio(R_p, returns, R_f, periods):
    return (R_p - R_f) / math.sqrt(average_dd_squared(returns, periods))
