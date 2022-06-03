import sys
import numpy as np
import pandas as pd
from tqdm import trange
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder


def run(dataset_train, dataset_test):
    # Batch size is the number of training examples used to calculate each iteration's gradient

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    # Define the hyperparameters
    learning_rate = 1e-2
    encoder = LSTMEncoder()
    decoder = LSTMDecoder()
    # Initialize the optimizer with above parameters
    optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    loss = train(model=shallow_model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn, indecies=tscv)

    # Test and get the resulting predicted y values
    y_predict = test(model=shallow_model, loader=data_loader_test)

    return loss, y_predict


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data.item()


def train(model, loader, optimizer, loss_fn, epochs=5):
    losses = list()

    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            batch_index += 1

        print("Epoch: ", e + 1)
        print("Batches: ", batch_index)
        if epoch % 100 == 0:
            print("loss: %1.5f" % (losses[-1].item()))

    return losses


def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)

    return y, y_predict


def test(model, loader):
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y, y_predict = test_batch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)

    return y_predict_vector


def plot_loss(losses, show=True):
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)
    ax = pyplot.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    pyplot.plot(x_loss, losses)

    if show:
        pyplot.show()

    pyplot.close()


class CoinsDataset(Dataset):
    def __init__(self, dataset, time_steps=14):
        # One-hot encode dataset
        #dataset = pd.get_dummies(dataset, columns=["symbol"], prefix="coin")

        # Normalize dataset
        data_high = np.asarray(dataset['Open']).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(data_high)

        data_normalized = scaler.transform(data_high).reshape(data_high.shape[0])
        nan_array = np.isnan(data_normalized)
        not_nan_array = ~ nan_array
        open_norm = data_normalized[not_nan_array]
        dataset['open_norm'] = open_norm
        # Split dataset into time periods
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor
        # dataX, dataY = pd.DataFrame(columns=dataset.columns), pd.DataFrame(columns=dataset.columns)
        dataX, dataY = [], []
        for i in range(int(max(dataset['date_delta'])) - time_steps - 1):
            # print(i)
            for coin in dataset['symbol'].unique():
                # print(coin)
                current_coin = dataset[dataset['symbol'] == coin]
                dataX.append(torch.tensor(current_coin.iloc[i:i + time_steps]['norm_open'].values))
                dataY.append(torch.tensor(current_coin.iloc[[i + time_steps]]['norm_open'].values))
                # dataX = pd.concat([dataX, current_coin.iloc[i:i + time_steps]])
                # dataY = pd.concat([dataY, current_coin.iloc[[i + time_steps]]])

        self.length = len(dataX)

        self.x_data = torch.from_numpy(np.array(dataX)).type(x_dtype)
        self.y_data = torch.from_numpy(np.array(dataY)).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)  # lstm

    def forward(self, x):
        out, hidden = self.lstm(x.view(x.shape[0], x.shape[1], self.input_size))

        return out, hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


def main():
    df = pd.read_csv('arbitrageData.csv')

    train = df[df['date_delta'] <= 0.8 * max(df['date_delta'])]
    test = df[df['date_delta'] > 0.8 * max(df['date_delta'])]

    dataset_train = CoinsDataset(train)
    dataset_test = CoinsDataset(test)

    print("Train set size: ", dataset_train.length)
    print("Test set size: ", dataset_test.length)

    losses, y_predict = run(dataset_train=dataset_train, dataset_test=dataset_test)

    print("Final loss:", sum(losses[-100:]) / 100)
    plot_loss(losses)

    fig2 = pyplot.figure()
    fig2.set_size_inches(8, 6)
    pyplot.scatter(x_test, y_test, marker='o', s=0.2)
    pyplot.scatter(x_test, y_predict, marker='o', s=0.3)
    pyplot.text(-9, 0.44, "- Prediction", color="orange", fontsize=8)
    pyplot.text(-9, 0.48, "- Sine (with noise)", color="blue", fontsize=8)
    pyplot.show()


if __name__ == "__main__":
    main()

'''
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length):
        super(LSTMDecoder, self).__init__()

        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)  # lstm

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden_states):
        lstm_out, hidden = self.lstm(x.unsqueeze(0), hidden_states)
        out = self.linear(lstm_out.squeeze(0))

        return out, hidden


class lstm_seq2seq(nn.Module):
     train LSTM encoder-decoder and make predictions 
    def __init__(self, input_size, hidden_size):


        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h


        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = LSTMEncoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = LSTMDecoder(input_size=input_size, hidden_size=hidden_size)

    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size, learning_rate=0.01, dynamic_tf=False):


        train lstm encoder-decoder

        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch


        # initialize array of losses
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                for b in range(n_batches):
                    # select data
                    input_batch = input_tensor[:, b: b + batch_size, :]
                    target_batch = target_tensor[:, b: b + batch_size, :]

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    for t in range(target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = decoder_output

                    # compute the loss
                    loss = loss_fn(outputs, target_batch)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch
                batch_loss /= n_batches
                losses[it] = batch_loss

                    # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

        return losses


class Linear(nn.Module):


    def __init__(self, input_size, time_step):
        # Perform initialization of the pytorch superclass
        super(Linear, self).__init__()

        # Define network layer dimensions
        D_in, H1, H2, D_out = [input_size, input_size / 2, input_size/4,input_size / time_step]  # These numbers correspond to each layer: [input, hidden_1, output]

        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)


class LSTM_Linear(nn.Module):
    def __init__(self, input_size, hidden_size):


        super(LSTM_Linear, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.linear = Linear(input_size=input_size)


'''
