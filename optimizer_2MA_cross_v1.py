'''
        .=====================================.
       /         Algorithm Optimizer:          \
      {       2 MA Cross Strategy Sample        }
       \             by Edorenta               /
        '====================================='
'''
#TREND FOLLOWING STRATEGY: -in this version of my optimizer we will focus on a simple
#                           yet popular strategy: 2 Moving Average Crossover
#ALWAYS INVESTED:          -for code execution speed constraints we will not to make the
#                           strategy technicaly too complex, hence we will always be invested
#SIMPLE LOGIC:             -if fast MA > slow MA, enter and hold long, else short and hold
#                           to next signal
#SHARPE OPTIMIZATION:      -the solver here draws off the main loop 2 key indicators:
#                           the strategy sharpe ratio and total return for every possible set of setting
#GRAPH PLOT:               -the results are here ploted using pylab and matplot lib under two choosen risks perspectives

#libs to include
import pandas as pd
import numpy as np
import numpy.random as nrand
import talib as ta
from math import sqrt
import matplotlib.pyplot as plt
from pylab import *
import datetime

#import risklib as rsk

#input variables
ma1_start = 15      #first parameter for the fast MA
ma1_stop = 40       #last parameter for the fast MA
ma2_start = 20      #first parameter for the slow MA
ma2_stop = 120      #first parameter for the slow MA
step_1 = 5          #slover increment for fast MA
step_2 = 5          #slover increment for slow MA

import_dir = "data_frame\\historical_data\\"        #root of the historical data .csv
export_dir = "data_frame\\technical_analysis\\"     #root of the export folder (useless atm)
file_name = "EURUSD_M5.csv"                         #instrument historical dt ohlc vol to load, format has to be 
digits = 5                                          #digits of the underlying instrument quote
spread_pts = 20                                     #broker typical spread in base points (pip/10) not implemented yet
transaction_cost = 1.5                              #broker transaction cost not implemented yet
timeframe = 5                                       #the imported file timeframe (OHLC timestamp)

#load historical data as pandas DataFrame
histo = pd.read_csv(import_dir + file_name, names=['datetime','open','high','low','close','volume'], header=0, index_col=0)

#get the date window and workaround for further annualized functions
start_date = histo.index[0]
end_date = histo.index[-1]
dt_window = (pd.to_datetime(end_date, infer_datetime_format=True, format = '%M') - pd.to_datetime(start_date, infer_datetime_format=True, format = '%M')).total_seconds()
dt_window = dt_window/(60*timeframe)
histo_len = len(histo.index)

#time_frame_mutliplier = np.round(dt_window/histo_len,5)
nb_histo_y = dt_window/((365*24*(60/timeframe))) #years on test in the history
data_per_year = round(histo_len/nb_histo_y,0) #ohlc per tradable year (rougly 252 days) => necessary for annualization

#other way:
#>> least_recent_date = df['StartDate'].min()
#>> recent_date = df['StartDate'].max()

#spread conversion
spread = spread_pts/(10^digits) #the bid/ask will only be computed on testing not to waste CPU

#dt = histo['datetime']
open = histo['open']
high = histo['high']
low = histo['low']
close = histo['close']
volume = histo['volume']
diff = close-open

'''   .-----------------------.
      |    RISKLIB EXTRACT    | => optimization criteria lay on these indicators, refer to risklib.py for complete list
      '-----------------------'
'''
#Sharpe Ratio function - Risk free rate excluded for simplicity
def annualised_sharpe(returns, tradable_days):
    vol = np.std(returns)
    if (vol == 0):
        vol = 0.01
    return np.sqrt(tradable_days) * (np.mean(returns) / vol) #annualized (expected return)/vol

#Drawdown
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

#Max DD
def max_dd(returns):
    #Returns the maximum draw-down for any ti in (0, T) where T is the length of the return sRpies
    max_DD = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_DD:
            max_DD = drawdown_i
    #Max draw-down should be positive
    return abs(max_DD)

#Relative changes to absolute
def prices(returns, base):
    #Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return np.array(s)

'''   .-----------------------.
      |    ALGORITHM LOGIC    | => first import indicators (custom or from TA-lib), then store it all in a pandas dataframe
      '-----------------------'
'''
#logic function that loops on every candle in the dataset
def strat_logic(ma1_p, ma2_p):
    #we are here working with moving averages, but TA-Lib recenses dozens of powerful indicators
    #we're getting 2 EMA: ma1 & ma2
    #pandas df columns have to be translated to series (i.e. np) to be readable by TA-Lib
    histo['ma1'] = np.round(ta.EMA(close.values, ma1_p),digits+2)
    histo['ma2'] = np.round(ta.EMA(close.values, ma2_p),digits+2)
    #create column with moving average spread differential
    histo['diff'] = histo['ma1'] - histo['ma2']

    #set desired number of points as threshold for spread difference (divergence) and create column containing strategy directional stance
    #divergence = +/-1/(10^digits)
    #divergence = div_x*10^(-5)
    histo['direction'] = np.where(histo['diff'] >= 0, 1, 0)
    histo['direction'] = np.where(histo['diff'] < 0, -1, histo['direction'])
    histo['direction'].value_counts()

#     .-----------------------.
#     |    HANDLE RETURNS     |
#     '-----------------------'
#
    #create columns containing daily mkt & str log returns for every row (every candle)
    histo['Market Returns'] = np.log(histo['close'] / histo['close'].shift(1))
    histo['Strategy Returns'] = histo['Market Returns'] * histo['direction'].shift(1)

    #create columns containing daily mkt & str candle absolute pts return
    histo['Market Shift'] = histo['close'] - histo['close'].shift(1)
    histo['Strategy Shift'] = histo['Market Shift'] * histo['direction'].shift(1)

    #TO DO: include spread if trade =>>>
        #if histo['direction'].shift(1) != histo['direction']:
        #histo['Strategy Shift'] = histo['Strategy Shift'] - (spread_pt/(10^digits))/2

    #set strategy starting equity to 1 (i.e. 100%) and generate equity curve
    histo['Strategy TR'] = histo['Strategy Returns'].cumsum() + 1
    histo['Benchmark TR'] = histo['Market Returns'].cumsum() + 1
 
    histo['Strategy AR'] = histo['Strategy Shift'].cumsum() + 1
    histo['Benchmark AR'] = histo['Market Shift'].cumsum() + 1

    #series of daily returns for risklib calls
    histo_d_series = [histo['Strategy Returns'], histo['Market Returns']]
    #concatenate to get 1 table
    histo_d = pd.concat(histo_d_series, axis=1, ignore_index=False)
    histo_d.columns=['Strategy DTR','Market DTR']

    #histo_d.reset_index(level=0, inplace=True)
    histo_d['datetime'] = histo_d.index
    histo_d['datetime'] = pd.to_datetime(histo_d['datetime'])
    histo_d = histo_d.set_index(['datetime'])

    #histo_d.index = datetime.datetime.strftime(histo_d.index, "%A")
    histo_d = histo_d.resample('24H').agg({'Strategy DTR': 'sum', 
                                           'Market DTR': 'sum'})
    histo_d = histo_d.dropna() #take the NaN out of the dataset, i.e. Sundays
    traded_days = len(histo_d.index)
    #print(histo, histo_d, traded_days)

    #extected returns translation, median approximation to avoid fat tails corruption:
    Rp = histo['Strategy Returns'].median()
    Rm = histo['Market Returns'].median()
    
    #expected daily returns:
    data_per_day = data_per_year*nb_histo_y/(traded_days) #corr tradable day, better than using 252
    Rpd = data_per_day*Rp
    Rmd = data_per_day*Rm

    #ending absolute return as instrument points
    Rp_pts = histo['Strategy AR'].iloc[-1]
    Rm_pts = histo['Benchmark AR'].iloc[-1]

    #print(histo['Strategy TR'])
    y_tradable = np.round((dt_window/traded_days),0)
    sharpe_strat = annualised_sharpe(histo_d['Strategy DTR'], y_tradable)
    return (histo['Strategy TR'][-1], sharpe_strat)

'''   .-----------------------.
      |  NUMPY OPTIMIZATION   | => optimization made possible through the linspace usage, heuristic solver
      '-----------------------'
'''
#run the previously genericly coded function on np linspaces
nb_pass_1 = np.floor((ma1_stop - ma1_start)/step_1)
nb_pass_2 = np.floor((ma2_stop - ma2_start)/step_2)
nb_pass_1 = max(nb_pass_1, nb_pass_2)
nb_pass_2 = nb_pass_1

ma1 = np.linspace(ma1_start,ma1_stop,nb_pass_1,dtype=int)
ma2 = np.linspace(ma2_start,ma2_stop,nb_pass_2,dtype=int)

#set series with dataframe length for risk/reward indicator storage
results_pnl = np.zeros((len(ma1),len(ma2)))
results_sharpe = np.zeros((len(ma1),len(ma2)))
pass_ = np.zeros((len(ma1),len(ma2)))

'''   .------------------------.
      | LOGIC LOOP IN LINSPACE |
      '------------------------'
'''
k=1 #pass counter
#logic looper through numpy's linspace
for i, fast_ma in enumerate(ma1):
    for j, slow_ma in enumerate(ma2):
        pnl, sharpe = strat_logic(fast_ma,slow_ma)
        pnl = (1-pnl)*100
        results_pnl[i,j] = pnl
        results_sharpe[i,j] = sharpe
        pass_[i,j]=k
        print("Pass %s: [%s|%s] Results: [P&L: %s | Sharpe: %s]" %(k, fast_ma, slow_ma, pnl, sharpe))
        k=k+1

'''   .-----------------------.
      |     PLOT RESULTS      |
      '-----------------------'
'''
#plot parameters
scatter_size = 20+15000*(1/k)
font1= {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 16,
    }
font2= {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 10,
    }
font3= {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': 14,
    }

#create figure frame
plt.figure(1)
figure(1).suptitle("Risk Reward Multiple Criteria Optimization", fontdict=font1, fontsize=16)

#2 dim sharpe graph
subplot(221)
y1 = results_sharpe
x1 = pass_
scatter(x1,y1,alpha=.4,s=scatter_size)
title('Sharpe Ratio Perspective', fontdict=font3)
ylabel('Sharpe Ratio', fontdict=font2)
xlabel('Pass #', fontdict=font2)
margins(0.2) #tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)

#2 dim returns graph
subplot(222)
y2 = results_pnl
x2 = pass_
scatter(x2,y2,alpha=.4,s=scatter_size)
title('Total Return Perspective', fontdict=font3)
ylabel('Return (%)', fontdict=font2)
xlabel('Pass #', fontdict=font2)
margins(0.2)
plt.subplots_adjust(bottom=0.15)

#3 dim sharpe graph
subplot(223)
x3 = ma1
y3 = ma2
z3 = results_sharpe
pcolor(x3,y3,z3)
colorbar()
#title('Sharpe Optimization', fontdict=font)
xlabel('Fast MA Period', fontdict=font2)
ylabel('Slow MA Period', fontdict=font2)
margins(0.2)
plt.subplots_adjust(bottom=0.15)

#3 dim returns graph
subplot(224)
x4 = ma1
y4 = ma2
z4 = results_pnl
pcolor(x4,y4,z4)
colorbar()
#title('TR Optimization', fontdict=font)
xlabel('Fast MA Period', fontdict=font2)
ylabel('Slow MA Period', fontdict=font2)
margins(0.2)
plt.subplots_adjust(bottom=0.15)

plt.show()

'''   .-----------------------.
      |   END OF OPTIMIZER    |
      '-----------------------'
'''
