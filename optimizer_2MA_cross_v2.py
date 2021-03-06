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
import math.sqrt
import matplotlib.pyplot as plt
import pylab
import datetime

#import risklib as rsk

#input variables
ma1_start = 2       #first parameter for the fast MA
ma1_stop = 10       #last parameter for the fast MA
ma2_start = 2       #first parameter for the slow MA
ma2_stop = 10       #first parameter for the slow MA
step_1 = 1          #slover increment for fast MA
step_2 = 1          #slover increment for slow MA

import_dir = "data_frame\\historical_data\\"        #root of the historical data .csv
export_dir = "data_frame\\technical_analysis\\"     #root of the export folder (useless atm)
file_name = "BTCUSD_M5.csv"                         #instrument historical data, format has to be utc|open|high|low|close or close
digits = 5                                          #digits of the underlying instrument quote
spread_pts = 20                                     #broker typical spread in base points (pip/10) not implemented yet
transaction_cost = 1.5                              #broker transaction cost not implemented yet
timeframe = 5                                       #the imported file timeframe (OHLC timestamp)

#load historical data as pandas DataFrame
histo = pd.read_csv(import_dir + file_name, names=['datetime','open','high','low','close','volume'], header=0, index_col=0)
histo.index = pd.to_datetime(histo.index)

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
    #Returns the draw-down given time period ti
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

#deduct the max passes from the user's input in order to dimension the result matrix
nb_pass_1 = np.floor((ma1_stop - ma1_start)/step_1)
nb_pass_2 = np.floor((ma2_stop - ma2_start)/step_2)
nb_pass_1 = max(nb_pass_1, nb_pass_2)
nb_pass_2 = nb_pass_1

#define NumPy's linspaces as being vectors of every single pass of the optimization
ma1 = np.linspace(ma1_start,ma1_stop,nb_pass_1,dtype=int)
ma2 = np.linspace(ma2_start,ma2_stop,nb_pass_2,dtype=int)

#set series with dataframe length for risk/reward indicator storage
results_pnl = np.zeros((len(ma1),len(ma2)))
results_sharpe = np.zeros((len(ma1),len(ma2)))
pass_ = np.zeros((len(ma1),len(ma2)))
res_pnl = np.zeros(1+nb_pass_1**2)
res_sharpe = np.zeros(1+nb_pass_1**2)
res_ma1 = np.zeros(1+nb_pass_1**2)
res_ma2 = np.zeros(1+nb_pass_1**2)

'''   .------------------------.
      | LOGIC LOOP IN LINSPACE |
      '------------------------'
'''
k=1 #pass counter
#run the previously genericly coded function through numpy's linspace
for i, fast_ma in enumerate(ma1):
    for j, slow_ma in enumerate(ma2):
        #call the strategy which will ouput both the result as P&L and computed Sharpe ratio
        pnl, sharpe = strat_logic(fast_ma,slow_ma)
        #set returns as %
        pnl = (pnl-1)*100
        results_pnl[i,j] = pnl
        results_sharpe[i,j] = sharpe
        pass_[i,j] = k
        #print the current stage of optimization
        print("Pass %s: [%s|%s] Results: [P&L: %s | Sharpe: %s]" %(k, fast_ma, slow_ma, pnl, sharpe))

        #store pass data into nympy vectors:
        res_pnl[k] = pnl
        res_sharpe[k] = sharpe
        res_ma1[k] = fast_ma
        res_ma2[k] = slow_ma

        #next pass:
        k=k+1

#store all the passes test results in a new dataframe
res_series = [res_pnl, res_sharpe, res_ma1, res_ma2]
df_res = pd.DataFrame(data=res_series)   # 1st row as the column names
#transpose to get proper format
df_res = df_res.T
df_res.columns=['return','sharpe','fast ma','slow ma']
df_res.index.names = ['Pass#']

#find the maximum results using pandas' argmax function
#highest Sharpe (risk-adjusted return approach):
max_sharpe_loc = df_res['sharpe'].argmax()
max_sharpe = df_res['sharpe'][max_sharpe_loc]
max_sharpe_MA1 = df_res['fast ma'][max_sharpe_loc]
max_sharpe_MA2 = df_res['slow ma'][max_sharpe_loc]
#highest ending equity (total return approach):
max_tr_loc = df_res['return'].argmax()
max_tr = df_res['return'][max_tr_loc]
max_tr_MA1 = df_res['fast ma'][max_tr_loc]
max_tr_MA2 = df_res['slow ma'][max_tr_loc]

print(df_res)

'''   .-----------------------.
      |     PLOT RESULTS      |
      '-----------------------'
'''

str_result = "Best settings for Sharpe : MA1(%s)/MA2(%s) \nBest settings for Total Return : MA1(%s)/MA2(%s)" %(max_sharpe_MA1,max_sharpe_MA2,max_tr_MA1,max_tr_MA2)
print(df_res, str_result)

#visual parameters
#init dynamic scatter point sizing
scatter_size = 20+15000*(1/k)

#title and axes format
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
#give it a name
figure(1).suptitle("Risk Reward Multiple Criteria Optimization", fontdict=font1, fontsize=16)

#first dimension: Sharpe graphs
pl1 = plt.subplot(221) #2D subplot: pass vs Sharpe
#set axis
y1 = results_sharpe
x1 = pass_
#set dynamic scatter
scatter(x1,y1,alpha=.4,s=scatter_size)
pl1.set_xlim(xmin=0)
pl1.set_xlim(xmax=trunc(k+k/20))
#labels
title('Sharpe Ratio Perspective', fontdict=font3)
ylabel('Sharpe Ratio', fontdict=font2)
xlabel('Pass #', fontdict=font2)
margins(0.2) #tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)

pl3 = plt.subplot(223) #3D subplot: MA1 vs MA2 vs Sharpe heatmap
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

#second dimension: Total Return graphs
pl2 = plt.subplot(222) #2D subplot: pass vs Return
#set axis
y2 = results_pnl
x2 = pass_
#set dynamic scatter
scatter(x2,y2,alpha=.4,s=scatter_size)
pl2.set_xlim(xmin=0)
pl2.set_xlim(xmax=trunc(k+k/20))
#labels
title('Total Return Perspective', fontdict=font3)
ylabel('Return (%)', fontdict=font2)
xlabel('Pass #', fontdict=font2)
margins(0.2) #tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)

#3 dim returns graph
pl4 = plt.subplot(224) #3D subplot: MA1 vs MA2 vs Return heatmap
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

#ask the user whether he'd like to access the return or Sharpe optimization details
decision = int(input('Please press:\n   1] Sharpe Ratio Optimization\n   2] Total Return Optimization\n   3] Exit\n\n'))
#answer management
if ((decision == 1) or (decision == 2)):
    gr_title = 'Crossover Return Analysis'
    if decision == 1: #Sharpe optimization popup
        gr_title = "MA(%s)/MA(%s) %s" %(max_sharpe_MA1,max_sharpe_MA2, gr_title)
        strat_logic(max_sharpe_MA1,max_sharpe_MA2)
    if decision == 2: #Return optimization popup
        gr_title = "MA(%s)/MA(%s) %s" %(max_tr_MA1,max_tr_MA2, gr_title)
        strat_logic(max_tr_MA1,max_tr_MA2)

#plot equity curve charts (absolute points & return)
plt.figure(2)
#get results in %
histo['Strategy TR %']=(histo['Strategy TR']-1)*100
histo['Benchmark TR %']=(histo['Benchmark TR']-1)*100
#set results to fitting grid
histo['Strategy TR %'].plot(grid=True,figsize=(8,5), legend=True) #, label='Strategy')
histo['Benchmark TR %'].plot(grid=True,figsize=(8,5), legend=True) #, label='Benchmark')
#display settings
title(gr_title, fontdict=font1)
ylabel('Total Return (%)', fontdict=font2)
xlabel('Date', fontdict=font2)
#show all the configured graphs
plt.show()

#plt.figure(3)
#histo['direction'].plot(grid=True,figsize=(8,5))



export_path = export_dir + 'BT_results_' + file_name

decision = int(input("Would you like to export the result file as %s?\n   1] YES\n   2] NO\n" %(export_path)))

if (decision == 1):
    #extract the csv
    header = ['open', 'high','low','close', 'ma1', 'ma2', 'direction', 'Strategy TR %', 'Benchmark TR %]']

    histo.to_csv(export_path, columns = header, sep=',', encoding='utf-8', index=False)

'''   .-----------------------.
      |   END OF OPTIMIZER    |
      '-----------------------'
'''
