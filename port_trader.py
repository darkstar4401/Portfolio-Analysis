#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from time import sleep
from binance.client import Client
import pandas as pd
import datetime
import time
import numpy as np
import os
import sys
#from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
#import tensorflow as tf
#import seaborn as sns
#from statsmodels.tsa.stattools import coint

#from crontab import CronTab
import logging
from traceback import format_exc
from schedule import Scheduler


logger = logging.getLogger('schedule')


class SafeScheduler(Scheduler):
    """
    An implementation of Scheduler that catches jobs that fail, logs their
    exception tracebacks as errors, optionally reschedules the jobs for their
    next run time, and keeps going.
    Use this to run jobs that may or may not crash without worrying about
    whether other jobs will run or if they'll crash the entire script.
    """

    def __init__(self, reschedule_on_failure=True):
        """
        If reschedule_on_failure is True, jobs will be rescheduled for their
        next run as if they had completed successfully. If False, they'll run
        on the next run_pending() tick.
        """
        self.reschedule_on_failure = reschedule_on_failure
        super().__init__()

    def _run_job(self, job):
        try:
            super()._run_job(job)
        except Exception:
            logger.error(format_exc())
            job.last_run = datetime.datetime.now()
            job._schedule_next_run()

#add your api keys here
api_key = 'XuozVlFg2WYVemwmHMqCK9j5jSYIV1z3MoDAXY76X2JiCbbgcw7dGxvE0eGVsqhH'
secret_key = 'WyWBCdczLCFvV8fKy16R5dOwhhjK31T7ZrW0cplIzjrvmlSPh5E1KEeAEh5c9I6s'

#Open location of recorded buy/sell walls
bw_location = './data/watcher/buywalls/'
sw_location = './data/watcher/sellwalls/'
save_port = '/githome/me/Documents/BWA/data/bw_portfolios'

#Open binance client
client = Client(api_key, secret_key)

#get current time and format to compare with data
current_milli_time = lambda: int(round(time.time() * 1000))


def get_data(sym,prev_k,k,candle_k='5m'):
    """
    Function downloads hist data from binance API with given timestamps
    """
    #previous kline interval
    prev_k = str(prev_k)
    #current kline interval
    k = str(k)
    #download historical data
    data = client.get_historical_klines(sym,interval= candle_k,start_str =int(prev_k),end_str = int(k))
    data = pd.DataFrame(data,columns=['Open time','Open','High','Low','Close','Volume','Close time','Quote volume','Num trades', 'btc buys', 'coin buys', 'ignore'])
    
    # convert from millesecond time to datetime for potential resampling and readability
    data['Open time'] = data['Open time']/1000
    data['Open time'] = data['Open time'].astype('datetime64[s]')
    data = data.set_index('Open time')
    data = data[['Open','High','Low','Close','Close time','Volume']].astype(float)
    return data

def to_dt(v,is_milli=True):
    """
    Function to convert timestamp to datetime object
    has parameter for millisecond uts timestamp
    """
    if(is_milli):
        v = datetime.datetime.fromtimestamp(int(v)/1000)
    else:
        v = datetime.datetime.fromtimestamp(int(v))
    return v

def get_freq_sig(loc):
    """
    Function loads all signal csv files and concatenates them into a signle dataframe
    """
    # cd cmd
    os.chdir(bw_location)
    #sorting by file creation (could use normal sort as filenames are timestamps)
    files = filter(os.path.isfile,os.listdir(bw_location))
    files =  [int(f) for f in files]
    files = list(files)
    #Create frequency signal dataframe
    fs_df = pd.DataFrame(files)
    #sort again
    fs_df = fs_df[0].sort_values()
    #starttime,endtime
    st,end = to_dt(fs_df.iloc[0]),to_dt(fs_df.iloc[-1])
    bw_freq = []
    
    f_len = int(len(os.listdir(loc)))
    for c,i in enumerate(os.listdir(loc)):
        try:
            curr = pd.read_csv(loc+i)
        except Exception as e:
            print(i,e)
 
        curr.columns = ['Coin','Close','Profit','Loss','Date']  
        bw_freq.append(curr)
        if((c/f_len)%10==0):
            pct = str(int((c/f_len)*100))
            print("{}% of files loaded".format(pct))
    bw_freq = pd.concat(bw_freq)
    bw_freq = bw_freq[['Date','Coin','Close']]
    bw_freq.columns = ['Date','Coin','Close']
    bw_freq = bw_freq.sort_values(['Date'],ascending=False)
    #print(all_w.head(), len(all_w))
    #bw_freq['all_freq'] = bw_freq.groupby('Coin')['Coin'].transform('count')
    #bw_freq = bw_freq.sort_values(['Date'],ascending=False)    
    print(st,end)
    print(to_dt(now), now)
    print(fs_df.head())
    return bw_freq

def clean_fs(fs):
    """
    Function to clean frequency signal dataframe
    """
    fs =fs.drop_duplicates()
    fs = fs.sort_values(['Date'],axis=0)
    fs['Date_m'] = fs['Date']
    fs['Date'] = fs['Date']/1000
    fs['Date'] = fs['Date'].astype('datetime64[s]')

    rolling_f = {}
    f_col = []
    """
    for i in fs.iterrows():
        coin = i[1][1]
        if(coin in rolling_f.keys()):
            rolling_f[coin] += 1
            f_col.append(rolling_f[coin])
        else:
            rolling_f[coin] = 1
            f_col.append(rolling_f[coin])
    print(len(fs),len(f_col))
    fs['rolling_freq'] = f_col
    """    
    return fs



def unix_time_millis(dt):
    """
    Function to convert unix time to millesecond
    """
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0



def interval(intv,st,end):
    """
    Function to resample frequency dataframe to specified frequency
    
    """
    intv_list = []
    out = pd.DataFrame()
    prev = st
    while(prev<end):
        curr = prev+int(86400000*intv)
        n = to_dt(curr).strftime('%Y-%m-%d')
        p = to_dt(prev).strftime('%Y-%m-%d')
        
        print(p,"  |  ",n)
        
        prev = curr
        p = datetime.datetime.strptime(p, '%Y-%m-%d')
        n = datetime.datetime.strptime(n, '%Y-%m-%d')
        intv_list.append(p)
        data = fs_c.loc[p:n]
        #print(data.sort_values('rolling_freq',ascending=False))
        out = pd.concat([out,data],axis=0)
        
    return out.sort_index(),intv_list
    #files.sort(key=lambda x: os.path.getmtime(x))
    
def intv_port(freq_df,intv):
    dates = freq_df.index.get_level_values('Date').drop_duplicates()

    coins = freq_df.index.get_level_values('Coin')
    date_int = pd.DataFrame({"Date":[d.strftime('%Y-%m-%d') for d in dates]})
    date_int = date_int.drop_duplicates()
    date_int = date_int.astype('datetime64[s]')
    intv_dates = list(date_int.Date[::intv])
    port = pd.DataFrame()
    for c,day in enumerate(intv_dates):
        try:
            st,end = day,intv_dates[c+1]
            print(st,end)
            curr = freq_df.loc[st:end].reset_index()
            curr['Date'] = end

        except Exception as e:
            #print(e)
            break

        curr['all_freq'] = curr.groupby('Coin')['Coin'].transform('count')
        curr = curr[['Date','Coin','Close','all_freq']].sort_values("all_freq",ascending=False)
        curr = curr.sort_values(['Date'],ascending=False)
        curr = curr.set_index('Date')
        port_coins = curr.Coin.drop_duplicates()
        c_group = curr.groupby('Coin')
        curr_port = pd.DataFrame()
        for pc in port_coins:
            curr = c_group.get_group(pc).head(1)
            curr_port = pd.concat([curr_port,curr])
            curr_port = curr_port.sort_values("all_freq",ascending=False).head(9)
        port = pd.concat([port,curr_port])

    port = port.reset_index()
    port['Date'] = port['Date'].astype('datetime64[s]')
    #print(port['Date'])
    d2 = [d.strftime('%Y-%m-%d') for d in port.Date]
    port['Date'] = d2
    port = port.set_index(['Date','Coin'])
    return port
                    
def backtest(signals,coin):
    # Set the initial capital
    initial_capital= 10.0
    size = initial_capital#/10
    # Create a DataFrame `positions`
    
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions["size"] = size
    positions[coin] = signals[coin]
    
    # Buy a 100 shares
    #calc share amt
    #print("im good")
    #print(positions.head())
    share = positions["size"]/positions[coin]
    #cname = str(coin)+""
    #print("im good")
    positions["share"] = share.values*signals['signal'].values
    
    # Initialize the portfolio with value owned   
    portfolio = positions.multiply(signals[coin], axis=0)
    # Store the difference in shares owned 
    pos_diff = positions.diff()
    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['bt_returns'] = portfolio['returns']*signals['signal']

    # Print the last lines of `portfolio`
    #print(portfolio.astype(str).iloc[-1])
    return portfolio#.astype(str).iloc[-1]

def make_signal(signals):
    #p over 2
    signals["p>2"] = np.where(signals[coin]>signals["2"],1.0,0.0)
    #p over 1
    signals["p>1"] = np.where(signals[coin]>signals["1"],1.0,0.0)
    #p over-1
    signals["p>-1"] = np.where(signals[coin]>signals["-1"],1.0,0.0)
    #p over -2
    signals["p>-2"] = np.where(signals[coin]>signals["-2"],1.0,0.0)
    # -2 stop loss with -5% risk adj
    signals['stop'] = signals["-2"]-(signals["-2"]*.05)
    # -1 stop for 1 crossver
    signals['stop2'] = signals["-1"]
    #shorts below stoploss
    signals['short'] = np.where(signals[coin]<signals["stop"],-1.0,0.0)
    #shorts before stoploss 2
    signals['short2'] = np.where(signals[coin]<signals["stop2"],-1.0,0.0)
    #long above -1
    signals['long'] = signals["p>-2"]-signals["p>-1"]
    #long 2 above 1
    signals['long2'] = signals["p>1"]-signals["p>2"]
    #exit at between 1 and -1
    signals['exit'] = signals["p>-1"]-signals["p>1"]
    #exit at crossover 2 *1%
    signals['exit2'] = signals["p>2"]-(signals["p>2"]+(signals["p>2"]*.01)) #sell 50% take profit 
    #signals['exit'] = np.where(signals[coin]>=signals["exit"],-1.0,0.0)
    signals['positions'] = signals['short']+signals['long']+signals['exit']
    signals['positions2'] = signals['short2']+signals['long2']+signals['exit2']
    
    return signals

def make_signal2(signals):
    #p over 2
    signals["p>2"] = np.where(signals[coin]>signals["2"],1.0,0.0)
    #p over 1
    signals["p>1"] = np.where(signals[coin]>signals["1"],1.0,0.0)
    signals['ss'] = -1
    signals["short"] = np.where(signals[coin]<signals["-1"],-1.0,0.0)
    #p over-1
    #Long zone 1
    signals["positions"] = signals["p>1"]+signals['short']#-signals["p>2"]#np.where(signals[coin]<signals["-1"],1.0,0.0)
    signals["positions"] = np.where(signals["positions"]==signals["ss"],0.0,1.0)
    #signals[[coin,"positions"]].plot()
    #plt.show()
    #p over -2
    #signals["z-1"] = np.where(signals[coin]>signals["-2"],1.0,0.0)
    return signals

def live_trader(portfolio,coin):
        #print(portfolio.head())
        holdings = 0
        p1 = 0
        entry_t = 0
        exit_t = 0
        pnls = []
        trade_ct = 0
        hold_return = portfolio[coin].iloc[-1]/portfolio[coin].iloc[0]
        #print(portfolio.head())
        for t in range(len(portfolio)):
            #print(portfolio['pos_diff'].iloc[t])
            #buy rep
            if portfolio['pos_diff'].iloc[t] == 1:
                holdings = portfolio['max_holdings'].iloc[t]
                entry_t = t
                #entry_time = portfolio.index.values[t].astype(datetime)
                print("buying... {} shares at: {}".format(holdings,portfolio[coin].iloc[t]))
                p1 = portfolio[coin].iloc[t]

            #sell rep    
            elif portfolio['pos_diff'].iloc[t] == -1 and holdings != 0:
                profit = ((portfolio[coin].iloc[t]/p1)-1)*100
                exit_t = t
                p_seg = portfolio[coin].iloc[entry_t:exit_t]
                #print(p_seg.max())
                maxpp = (p_seg.max()/p1)#-1)*100
                if(maxpp >0):
                    print("max possible profit: ",maxpp)
                #print(profit)
                #exit_time = portfolio.index.values[t].astype(datetime)
                #t_held = datetime.timedelta(entry_time,exit_time)
                print("selling... {} shares at: {} p/l {}".format(holdings,portfolio[coin].iloc[t],profit))
                holdings = 0
                pnls.append(profit)
                trade_ct +=1

            elif t == len(portfolio)-2:
                profit = ((portfolio[coin].iloc[t]/p1)-1)*100
                pnls.append(profit)
                print("\nAlgo pnl: {}% Holding pnl: {}%".format(sum(pnls),hold_return))

                
now = str(current_milli_time())
#get freq. signal df
freq_loc = 'freq_concat.csv'
fs = pd.read_csv(freq_loc)
#clean freq. signal df
fs_c = clean_fs(fs)
#multilevel indexing for interval tracking
fs_c = fs_c.set_index(['Date','Coin'])
str_intv = 14#input("portfolio interval: ")
intv = int(str_intv)
port = intv_port(fs_c,intv)
port = port.sort_values(['Date','all_freq'],ascending=[True,False])
#port.to_csv("/home/me/Documents/BWA/data/{}day_portfolio.csv".format(intv))

#import ffn
p_dates = port.index.get_level_values('Date').drop_duplicates()
returns = []
all_ports = pd.DataFrame()
lastn = p_dates[-2:]
print(lastn)
returns = []
totals = []
coins = []
st_price = []
end_price = []

hist_data = None
for c,d in enumerate(lastn):
    if(c>len(lastn)-1):
        break
    try:
        st = datetime.datetime.strptime(d, '%Y-%m-%d')
        end = datetime.datetime.strptime(lastn[c+1], '%Y-%m-%d')
        print(st,end)
        st,end = int(unix_time_millis(st)),int(unix_time_millis(end))
        prev_st = st-86400000*intv
        curr_port = port.loc[d]
        p_coins = curr_port.index.get_level_values('Coin').values
        curr_port_hist = pd.DataFrame()
        port_hist = pd.DataFrame()
        #for coin in p_coins:
        #    q_hist = get_data(coin,st,end,candle_k='1h')
        #    port_hist[coin] = q_hist.Close
            
        for coin in p_coins:
            #current hist ...lazy way
            #c_hist = get_data(coin,st,end,candle_k='1h')
            #btc_hist = get_data('BTCUSDT',st,end,candle_k='1h')
            #q trader hist with prior data for training
            q_hist = get_data(coin,st,end,candle_k='1h')
            prices = pd.DataFrame()
            st_price.append(q_hist.Close.iloc[0])
            end_price.append(q_hist.Close.iloc[-1])

            high = q_hist.High
            prices[coin] = q_hist['Close']
            
            # Compute the cumulative moving average of the price
            prices['-1'] = [prices[coin][:i].mean() for i in range(len(prices))]
            vol = [prices[coin][:i].std() for i in range(len(prices))]
            ch = [high[:i].mean() for i in range(len(high))]
            prices['2'] = prices['-1'] + vol
            prices['1'] = ch
            prices['-2'] = prices['-1'] -vol
            prices.plot()
            plt.show()
            signals = make_signal2(prices.copy())
            #print(signals.head())
            #print(signals[[coin,'long','short','exit','positions']])
            prices['returns'] = np.log(prices[coin] / prices[coin].shift(1))
            prices['cum_returns'] = prices['returns'].cumsum()
            #signals['positions'] = signals['positions']+signals['positions2']
            prices["signal"] = signals['positions']
            #print(prices["signal"].tail())
            prices["pos_diff"] = prices['signal'].diff()
            prices['bt_returns'] = (prices['returns']*prices["signal"]).cumsum()
            #print(prices[['bt_returns','cum_returns']].iloc[-1])
            portfolio = prices[[coin,'pos_diff','signal']].copy()
            portfolio['max_holdings'] = .1/portfolio[coin]
            live_trader(portfolio,coin)
            #print(prices.head())
            #bt = backtest(prices,coin)
            
            #returns.append(prices['bt_returns'].iloc[-1])
            #totals.append(bt['total'])
            #coins.append(coin)
            
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('time')
            ax1.set_ylabel('price', color=color)
            ax1.plot(signals[coin],color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel("signal", color=color)  # we already handled the x-label with ax1
            ax2.plot(prices["signal"], color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            figname = coin+str(st)
            plt.savefig(figname, bbox_inches="tight")
            plt.show()
            #break
            
            
            
        plt.show()
        break
    except Exception as e:
        print(e)
        pass

#all_ports= all_ports.fillna(0)
#all_ports.to_csv(fname+"all_coins.csv")

