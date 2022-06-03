import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

def main():
    dataset = pd.read_csv('arbitrageData.csv')
    #dataset = pd.get_dummies(dataset, columns=["symbol"], prefix="coin")
    data_high = np.asarray(dataset['open']).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(data_high)

    data_normalized = scaler.transform(data_high).reshape(data_high.shape[0])
    nan_array = np.isnan(data_normalized)
    not_nan_array = ~ nan_array
    data = data_normalized[not_nan_array]

    dataset['norm_open'] = data


    time_steps = 14
    x_dtype = torch.FloatTensor
    y_dtype = torch.FloatTensor
    #dataX, dataY = pd.DataFrame(columns=dataset.columns), pd.DataFrame(columns=dataset.columns)
    dataX, dataY = [], []
    for i in range(int(max(dataset['date_delta'])) - time_steps - 1):
        #print(i)
        for coin in dataset['symbol'].unique():
            #print(coin)

            current_coin = dataset[dataset['symbol'] == coin]
            dataX.append(torch.tensor(current_coin.iloc[i:i + time_steps]['norm_open'].values))
            dataY.append(torch.tensor(current_coin.iloc[[i + time_steps]]['norm_open'].values))
            #dataX = pd.concat([dataX,current_coin.iloc[i:i + time_steps]])
            #dataY = pd.concat([dataY,current_coin.iloc[[i + time_steps]]])

    length = len(dataX)

    np.set_printoptions(threshold=sys.maxsize)
    #print(dataX)
    #print(dataY)


    with open('dataX.txt','w') as f:
        for arr in dataX:
            f.write(str(arr)+'\n')

    with open('dataY.txt', 'w') as f:
        for arr in dataY:
            f.write(str(arr)+'\n')




    # list of np arrays?

    #x_data = torch.from_numpy(np.array(dataX)).type(x_dtype)
    # = torch.from_numpy(np.array(dataY)).type(y_dtype)

    #print(x_data)
    #print(y_data)



if __name__ == "__main__":
    main()