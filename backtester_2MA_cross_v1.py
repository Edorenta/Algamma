'''
        .=====================================.
       /         Backtesting Algorithm:        \
      {                2MA cross                }
       \              by Edorenta              /
        '====================================='
'''
#libs to include
import pandas as pd
import numpy as np
import numpy.random as nrand
import talib as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from pylab import *
from math import sqrt
import datetime

#import risklib as rsk

#input variables
import_dir = "data_frame\\historical_data\\"
export_dir = "data_frame\\technical_analysis\\"
file_name = "EURUSD_M5.csv" #instrument historical dt ohlc vol to load
ma1_p = 15
ma2_p = 90
digits = 5
spread_pts = 20
timeframe = 5 #in minutes to annualize

#download data into DataFrame and create moving averages columns
histo = pd.read_csv(import_dir + file_name, names=['datetime','open','high','low','close','volume'], header=0, index_col=0)
histo.index = pd.to_datetime(histo.index)

start_date = histo.index[0]
end_date = histo.index[-1]
dt_window = (pd.to_datetime(end_date, infer_datetime_format=True, format = '%M') - pd.to_datetime(start_date, infer_datetime_format=True, format = '%M')).total_seconds()
dt_window = dt_window/(60*timeframe)
histo_len = len(histo.index)
print(histo.head(), histo_len)

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

#Sharpe Ratio function - Risk free rate excluded for simplicity
def annualised_sharpe(returns, N=data_per_year):
    return np.sqrt(N) * (np.mean(returns) / np.std(returns)) #annualized (expected return)/vol

#pandas dataframe to series (i.e. np) to be readable by TA-Lib
histo['ma1_p'] = np.round(ta.SMA(close.values, ma1_p),digits+2)
histo['ma2_p'] = np.round(ta.SMA(close.values, ma2_p),digits+2)
 
#create column with moving average spread differential
histo['ma1_p-ma2_p'] = histo['ma1_p'] - histo['ma2_p']
 
#set desired number of points as threshold for spread difference (divergence) and create column containing strategy directional stance
#divergence = 1/(10^digits)
divergence = 0
histo['direction'] = np.where(histo['ma1_p-ma2_p'] >= divergence, 1, 0)
histo['direction'] = np.where(histo['ma1_p-ma2_p'] < divergence, -1, histo['direction'])
histo['direction'].value_counts()

#print(histo['ma1_p'], histo['ma2_p'])

#create columns containing daily mkt & str candle log returns
histo['Market Returns'] = np.log(histo['close'] / histo['close'].shift(1))
histo['Strategy Returns'] = histo['Market Returns'] * histo['direction'].shift(1)

#print(histo['direction'], histo['Market Returns'], histo['Strategy Returns'])

#create columns containing daily mkt & str candle absolute pts return
histo['Market Shift'] = histo['close'] - histo['close'].shift(1)
histo['Strategy Shift'] = histo['Market Shift'] * histo['direction'].shift(1)

#include spread if trade
#if histo['direction'].shift(1) != histo['direction']:
#histo['Strategy Shift'] = histo['Strategy Shift'] - (spread_pt/(10^digits))/2

#set strategy starting equity to 1 (i.e. 100%) and generate equity curve
histo['Strategy TR'] = histo['Strategy Returns'].cumsum() + 1
histo['Benchmark TR'] = histo['Market Returns'].cumsum() + 1
 
histo['Strategy AR'] = histo['Strategy Shift'].cumsum() + 1
histo['Benchmark AR'] = histo['Market Shift'].cumsum() + 1

#print(histo['Strategy TR'])
histo['Sharpe'] = annualised_sharpe(histo['Strategy TR'])
sharpe = histo['Sharpe']

#plot equity curve charts (absolute points & return)
histo.index.strftime('%m/%d/%Y')

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

plt.figure(1)
histo['Strategy AR'].plot(grid=True,figsize=(8,5), legend=True) #, label='Strategy')
histo['Benchmark AR'].plot(grid=True,figsize=(8,5), legend=True) #, label='Benchmark')
title('MA Crossover Return Analysis', fontdict=font1)
ylabel('Total Return', fontdict=font2)
xlabel('Date', fontdict=font2)

plt.figure(2)
histo['direction'].plot(grid=True,figsize=(8,5))

plt.show()
print(histo)
