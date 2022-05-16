from sklearn.cluster import OPTICS, cluster_optics_dbscan
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
		frame['Date'] = pd.to_datetime(frame['Date']
)		if coins.empty:
			coins = frame
		minDates.append(min(frame['Date']))
		coins = pd.concat([coins,frame],ignore_index=True)
		#coins = pd.merge(coins,frame,on='Date',how='inner')
		
		#coins.append(frame)

coins = coins[coins['Date'] >= max(minDates)]
coins['date_delta'] = (coins['Date'] - coins['Date'].min())  / np.timedelta64(1,'D')
with open('arbirageData.txt','w') as f:
	f.write(coins.to_string())

X = coins[['date_delta','Open']]

clust = OPTICS().fit_predict(X)
#print(clusters.core_distances_[clusters.ordering_])


space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]


