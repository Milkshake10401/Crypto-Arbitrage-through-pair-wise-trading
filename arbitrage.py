import pandas
from sklearn.cluster import OPTICS
import statsmodels.tsa.stattools as stats
from hurst import compute_Hc

import numpy as np
import pandas as pd
import os
import sys

import SimpleLSTM

def main():
	np.set_printoptions(threshold=sys.maxsize)
	coins = pd.DataFrame()
	minDates = []
	for filename in os.scandir('./data/Historical/Binance'):
		if filename.is_file():

			if str(filename).split('_')[2][0] == 'd':
				frame = pd.read_csv(filename,header=1)
				#print(frame.columns)
				frame = frame[['symbol','date','open']]
				frame['date'] = pd.to_datetime(frame['date'])
				#print(frame)
				if coins.empty:
					coins = frame
					minDates.append(min(frame['date']))
				else:
					minDates.append(min(frame['date']))
					coins = pd.concat([coins,frame],ignore_index=True)
			#coins = pd.merge(coins,frame,on='Date',how='inner')

			#coins.append(frame)
	#print(coins)
	coins = coins[coins['date'] >= max(minDates)]
	coins['date_delta'] = (coins['date'] - coins['date'].min())  / np.timedelta64(1,'D')
	maxD = max(coins['date_delta'])
	#print(maxD)
	for coin in pd.unique(coins['symbol']):
		#print(coin)
		#print(coins[coins['symbol'] == coin]['date_delta'])
		if max(coins[coins['symbol'] == coin]['date_delta']) < maxD:
			coins.drop(coins[coins.symbol == coin].index, inplace = True)

		#print(coin)
	pairs = findPairs(coins)
	print(pairs)

	predictions = SimpleLSTM.run(coins[['open','symbol']],pairs,0.001,1000)

	thresholds = calc_thresholds(predictions)
	print(thresholds)


	'''
	Predicted Change = delta_t+1 = (S_t+1 - S_t)/(S_t) * 100
	S(t): the spread of pair at time t
	Where S_t+1 is the predicted and S_t is the observed value at time t
	if delta_t+1 < alpha_l 
		open short position
	if delta_t+1 > alpha_s
		open long position
		
	Spread Percentage Change
		x_t = (S_t - S_t-1)/S_t-1 * 100
	
	create distribution based on Spread Percentage Change as f(x)
	Select top decile and quantile from f(x) > 0 and f(x) < 0
	
	test decile and quantile performance in validation set and use better performing
	
	
	
	'''


def calc_thresholds(predictions):
	"""

	:param predictions:
	:return:
	"""

	thresholds = {}
	for ts in predictions:
		#print(ts)
		Dp_t = []
		Dn_t = []
		for i in range(1,len(ts[1])):
			D_i = (ts[1][i]-ts[1][i-1])/ts[1][i-1]
			#print(D_i)
			if D_i >= 0:
				Dp_t.append(D_i)
			else:
				Dn_t.append(D_i)

		Dp_t = pd.DataFrame(Dp_t, columns=['D_t'])
		Dn_t = pd.DataFrame(Dn_t, columns=['D_t'])

		#print(ts[0])
		#print('neg:',Dp_t)
		#print('pos:',Dn_t)
		thresholds_pos = Dp_t.quantile([0.1,0.25,0.75,0.9])
		thresholds_neg = Dn_t.quantile([0.1,0.25,0.75,0.9])
		thresholds[ts[0]] = [thresholds_pos, thresholds_neg]

	return thresholds

def findPairs(coins):
	X = coins[['date_delta','open']]

	clusters = OPTICS().fit_predict(X)

	coins['clusterLabel'] = clusters

	coins_noOutliers = coins.copy()
	coins_noOutliers.drop(coins[coins.clusterLabel == -1].index, inplace = True)
	cointegrated = []
	clusteredCoins = set()
	for i in range(coins_noOutliers['clusterLabel'].max()+1):
		names = coins_noOutliers[coins_noOutliers['clusterLabel'] == i]
		names = names['symbol'].unique()
		if len(names) > 1:
			clusteredCoins.add(tuple(names))
		#print(names)

	with open('arbitrageData.csv','w') as f:
		f.write(coins.to_csv())

	for ele in clusteredCoins:
		#print(ele)
		for i in range(len(ele)-1):
			for coin2 in ele[i+1:]:
				co1 = coins[coins['symbol'] == ele[i]]
				co2 = coins[coins['symbol'] == coin2]
				#print(co1)
				#print(co2)
				#print()
				coint = stats.coint(co1['open'], co2['open'])

				if coint[1] < 0.05:
					cointegrated.append((ele[i],coin2))


	co_pairs = set(cointegrated)
	coH_pairs = []
	print(co_pairs)
	for pair in co_pairs:
		#print(coins[coins['symbol'] == pair[0]]['open'].to_numpy())
		#print(coins[coins['symbol'] == pair[1]]['open'].to_numpy())
		spread = np.subtract(coins[coins['symbol'] == pair[0]]['open'].to_numpy(), coins[coins['symbol'] == pair[1]]['open'].to_numpy())
		spread = np.absolute(spread)
		spread[spread==0] = np.finfo(np.float64).eps
		#print(spread)
		H, c, data = compute_Hc(spread, kind='price', simplified=True)
		if H < 0.5:
			coH_pairs.append(pair)

	return coH_pairs





if __name__ == "__main__":
    main()





