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
for filename in os.scandir('./data/Kaggle'):
	if filename.is_file():
		frame = pd.read_csv(filename)
		frame = frame[['Name','Date','Open']]
		frame['Date'] = pd.to_datetime(frame['Date'])		
		if coins.empty:
			coins = frame
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
coins = coins.sort_values(by = 'clusterLabel')

conis = coins.drop(coins[coins.clusterLabel == -1].index, inplace = True)


with open('arbirageData.txt','w') as f:
	f.write(coins.to_string())



