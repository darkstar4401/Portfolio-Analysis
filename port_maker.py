#import run_benchmarks
import random 
#from ddqn_agent import DDQNAgent
import os,shutil
import time
import pandas as pd
from binance.client import Client
"""
Script gets portforlio tickers from user  and a timeframe to get historical data from
"""
api_key = 'XuozVlFg2WYVemwmHMqCK9j5jSYIV1z3MoDAXY76X2JiCbbgcw7dGxvE0eGVsqhH'
secret_key = 'WyWBCdczLCFvV8fKy16R5dOwhhjK31T7ZrW0cplIzjrvmlSPh5E1KEeAEh5c9I6s'
#Open binance client
client = Client(api_key, secret_key)
#Open location of recorded buy/sell walls
bw_location = '/home/me/Documents/BWA/data/watcher/buywalls/'

def get_freq(loc,d,de=False):
	#os.chdir(bw_location)
	#files = filter(os.path.isfile,os.listdir(bw_location))
	#files =  [f for f in files]
	#files.sort(key=lambda x: os.path.getmtime(x))
	bw_concat = []
	now = int(current_milli_time())
	d1= now-(86400000*d)
	d2 = now-(86400000*de)
	#loop through every x days of data

	if(d2>d1):
		print("Got {} days data".format(d))
		de=False
	#print(now)
	f=0
	for i in os.listdir(loc):
		if ".csv" in i:
			f = i.replace(".csv","")
		if(int(f)<d1 or int(f)>d2):
			continue


		try:
			curr = pd.read_csv(loc+i)
		except Exception as e:
			print(i,e)
		curr.columns = ['Coin','Close','Profit','Loss','Date']
		bw_concat.append(curr)
	all_w = pd.concat(bw_concat)
	#print(all_w.head())
	allw = all_w[['Coin','Close','Profit','Loss','Date']]
	all_w.columns = ['Coin','Close','Profit','Loss','Date']
	all_w = all_w.sort_values(['Date'],ascending=False)
	#print(all_w.head(), len(all_w))
	all_w['all_freq'] = all_w.groupby('Coin')['Coin'].transform('count')
	all_w = all_w.sort_values(['Date'],ascending=False)
	#print(all_w.head())
	return all_w

def create_port(top10):
	portfolio = {}
	print(top10.columns)
	#top10.columns = ['Coin','Close','Profit','Loss','Date']
	top10.set_index('Coin', inplace=True)
	for coin in top10.index:
		print(coin)
		startTime = top10.loc[coin]['Date']
	    
	    #endTime=str(float(startTime)+240000)
	    #print(float(endTime)-int(startTime))
	    #histData = client.get_historical_klines(coin,client.KLINE_INTERVAL_1MINUTE,'30 days ago')
	    #histData = pd.DataFrame(histData,columns=['Open time','Open','High','Low','Close','Volume','Close time','Quote volume','Num trades', 'btc buys', 'coin buys', 'ignore'])
	    #portfolio[coin] = histData
	    #print()"""
	return portfolio, top10

def get_n_freq(loc,d):
    port_loc = "/home/me/Documents/BWA/data/bw_portfolios/{}_days/".format(d)
    if os.path.exists(port_loc):
        shutil.rmtree(port_loc)
    os.chdir(loc)

    files = filter(os.path.isfile,os.listdir(loc))
    files =  [f for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    #get start of dataset
    print(files[0],files[-1])
    prev = int(files[0])
    #stepped loop for daily frequnecy
    for i in range(prev,int(files[-1]),int(86400000*d)):
        bw_concat = []
        d1= int(prev)
        d2 = int(i)

    #loop through each file
        for c,j in enumerate(files):

            #cleaning and condition check

            f = j.replace(".csv","")
            if int(f)<d1 or int(f)>d2:
                continue

            #if(".csv" not in i):
            #    i+=".csv"
            #print(i)
            try:

                curr = pd.read_csv(loc+j)
                curr.columns = ['Coin','Close','Profit','Loss','Date']
                bw_concat.append(curr)
            except Exception as e:
                print(i,e)


        try:
            all_w = pd.concat(bw_concat)
            #print(all_w.head())
            allw = all_w[['Coin','Close','Profit','Loss','Date']]
            all_w.columns = ['Coin','Close','Profit','Loss','Date']
            #all_w = all_w.sort_values(['Date'],ascending=False)
            #print(all_w.head(), len(all_w))
            all_w['all_freq'] = all_w.groupby('Coin')['Coin'].transform('count')
            all_w = all_w.drop_duplicates()
            top10 = all_w.sort_values(['all_freq'],ascending=False)
            top10 = top10.Coin.drop_duplicates()
            #create monthly portfolio for testing
            #portfolio,top10 = create_port(all_w.head(10))
            

            if not os.path.exists(port_loc):
                os.mkdir(port_loc)
            top10=top10.head(10)
            print(top10)
            #print(top10,[i for j in range(10)]
            top10.to_csv("{}{}".format(port_loc,i))
        except Exception as e:
            print(e)
            continue
        prev = i
        #print(all_w.head())
    return top10
        
    
    


#d1_bw = get_n_freq(bw_location,1.5)
#d = int(input("Enter days for frequency portfolio: "))
D30 = get_n_freq(bw_location,30)
D20 = get_n_freq(bw_location,20)
D10 = get_n_freq(bw_location,10)
D7 = get_n_freq(bw_location,7)
D5 = get_n_freq(bw_location,5)
D3 = get_n_freq(bw_location,3)
#sellw = get_last_day(monthly)

#print(d1_bw.head())
"""
random.seed(3456)

coin_name = 'bitcoin'
run_benchmarks.run_bollingerband_agent(
    coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)

run_benchmarks.run_random_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)

run_benchmarks.run_alwaysbuy_agent(coin_name=coin_name, num_coins_per_order = num_coins_per_order, recent_k=recent_k)



def portfolio_dqn(syms):
	for sym in syms:
		port_agent = DDQNAgent(coin_name=sym, recent_k = recent_k, num_coins_per_order = num_coins_per_order, 
		                       epsilon_min = epsilon_min,
		                       external_states = ["current_price", "cross_upper_band", "cross_lower_band"],
		                       internal_states = ["is_holding_coin"])
		port_agent.plot_env(states_to_plot=["current_price", "upper_band", "lower_band"])
		port_agent.train(num_episodes=400)
		port_agent.plot_cum_returns()
		port_agent.test(epsilon=0.018)
"""