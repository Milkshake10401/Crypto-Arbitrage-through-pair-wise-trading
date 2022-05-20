from sklearn.cluster import OPTICS
import statsmodels.tsa.stattools as stats
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
import sys
np.set_printoptions(threshold=sys.maxsize)
coins = pd.DataFrame()
minDates = []
for filename in os.scandir('./data/Historical/Kaggle'):
	if filename.is_file():
		frame = pd.read_csv(filename)
		frame = frame[['Name','Date','Open']]
		frame['Date'] = pd.to_datetime(frame['Date'])		
		if coins.empty:
			coins = frame
			minDates.append(min(frame['Date']))
		else:
			minDates.append(min(frame['Date']))
			coins = pd.concat([coins,frame],ignore_index=True)
		#coins = pd.merge(coins,frame,on='Date',how='inner')
		
		#coins.append(frame)

coins = coins[coins['Date'] >= max(minDates)]
coins['date_delta'] = (coins['Date'] - coins['Date'].min())  / np.timedelta64(1,'D')
'''
with open('arbirageData.txt','w') as f:
	f.write(coins.to_string())
'''
X = coins[['date_delta','Open']]

clusters = OPTICS().fit_predict(X)

coins['clusterLabel'] = clusters

coins_noOutliers = coins.copy()
coins_noOutliers.drop(coins[coins.clusterLabel == -1].index, inplace = True)

cointegrated = []
clusteredCoins = set()
for i in range(coins_noOutliers['clusterLabel'].max()+1):
	names = coins_noOutliers[coins_noOutliers['clusterLabel'] == i]
	names = names['Name'].unique()
	if len(names) > 1:
		clusteredCoins.add(tuple(names))
	#print(names)

with open('arbirageData.txt','w') as f:
	f.write(coins.to_string())

for ele in clusteredCoins:
	print(ele)
	for i in range(len(ele)-1):
		for coin2 in ele[i+1:]:
			co1 = coins[coins['Name'] == ele[i]]
			co2 = coins[coins['Name'] == coin2]

			#print(len(co1))
			#print(len(co2))

			coint = stats.coint(co1['Open'], co2['Open'])
			#print(ele[i])
			#print(coin2)
			#print(coint)
			if coint[1] < 0.05:
				cointegrated.append((ele[i],coin2))
print(set(cointegrated))





