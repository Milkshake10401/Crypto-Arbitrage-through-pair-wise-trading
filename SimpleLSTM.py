
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out

def trainTest(df,pair,target,split):
    """
    :param df: pandas dataframe
    :param target: name of the security to be predicted
    :param split: scalar determining size of train dataset
    :return: train pd dataframe, test pd dataframe
    """
    coins = {}
    for symbol in df['symbol'].unique():
        coins[symbol] = df[df['symbol'] == symbol]

    X = pd.DataFrame()

    for symbol in pair:
        X[symbol] = coins[symbol]['open'].values

    #print(X.head())
    Y = pd.DataFrame()
    Y[target] = X[target].values

    mm = MinMaxScaler()
    ss = StandardScaler()

    X = ss.fit_transform(X)
    Y = mm.fit_transform(Y)

    y_train = Y[:int(split*Y.shape[0])]
    y_test = Y[int(split*Y.shape[0]):]

    x_train = X[:int(split*X.shape[0])]
    x_test = X[int(split*X.shape[0]):]

    X_train_tensors = Variable(torch.Tensor(x_train))
    X_test_tensors = Variable(torch.Tensor(x_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    # reshaping to rows, timestamps, features

    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    return X_train_tensors_final,X_test_tensors_final,y_train_tensors,y_test_tensors,mm

def train(lstm1,epochs,criterion,optimizer,x_train,y_train):
    """

    :param epochs:
    :param lr:
    :param criterion:
    :param optimizer:
    :return: trained lstm
    """

    for epoch in range(epochs):
        outputs = lstm1.forward(x_train)  # forward pass
        optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    return lstm1

def test(lstm1,x_test,mm):
    """

    :param x_test:
    :param lstm1:
    :return:
    """

    train_predict = lstm1(x_test)  # forward pass
    return mm.inverse_transform(train_predict.data.numpy())  # numpy conversion and re-scaled

def plot(data_predict,y_test,mm,symbol):
    """

    :param predict:
    :param y_test:
    :return:
    """

    dataY_plot = y_test.data.numpy()
    dataY_plot = mm.inverse_transform(dataY_plot)
    plt.figure(figsize=(10, 6))  # plotting

    plt.xlabel("Days")
    plt.ylabel("Price USD")
    plt.plot(dataY_plot, label='True {name} Price'.format(name=symbol))  # actual plot
    plt.plot(data_predict, label='Predicted {name} Price'.format(name=symbol))  # predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

    return data_predict

def run(df,pairs,learning_rate=0.001,epochs=1000):
    """

    :param df: dataframe containing time series data
    :param pairs: list of pairs found by findPairs() in arbitrage.py
    :return:
    """

    predictions = []
    for pair in pairs:
        for sec in pair:
            x_train, x_test, y_train, y_test, mm = trainTest(df, pair, sec, 0.7)

            #print("Training Shape", x_train.shape, y_train.shape)
            #print("Testing Shape", x_test.shape, y_test.shape)
            print(sec)

            input_size = 2  # number of features
            hidden_size = 2  # number of features in hidden state
            num_layers = 1  # number of stacked lstm layers

            num_classes = 1  # number of output classes

            lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train.shape[1])  # our lstm class

            criterion = torch.nn.MSELoss()  # mean-squared error for regression
            optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

            lstm1 = train(lstm1, epochs, criterion, optimizer, x_train, y_train)

            predict = test(lstm1, x_test,mm)

            predict_unscaled = plot(predict, y_test, mm,sec)
            predictions.append((sec,predict_unscaled))

    return predictions
def main():
    df = pd.read_csv('arbitrageData.csv')


    print(run(df,[('COMP/USDT', 'SOL/USDT'), ('SHIB/USDT', 'TRX/USDT'), ('CELR/USDT', 'DOGE/USDT')],0.001,1000))
    '''
    

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    X_train = X_ss[:200, :]
    X_test = X_ss[200:, :]

    y_train = y_mm[:200, :]
    y_test = y_mm[200:, :]
    '''

if __name__ == "__main__":
    main()