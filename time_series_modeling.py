import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, torch, pickle
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


#Model
class NNTimeSeriesModel(nn.Module):
    def __init__(self, n_lstm_layers, n_input, n_hidden, n_output):
        super(NNTimeSeriesModel, self).__init__()

        self.n_lstm_layers = n_lstm_layers
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.LSTM_layer = nn.LSTM(self.n_input, self.n_hidden, self.n_lstm_layers, batch_first=True)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.n_hidden, int(self.n_hidden*2)),
            nn.ReLU(),
            nn.Linear(int(self.n_hidden*2), self.n_output)
        )

    def forward(self, input):
        #initialize hidden and cell states to zeros (0)
        h0 = torch.zeros(self.n_lstm_layers, input.shape[0], self.n_hidden).requires_grad_()
        c0 = torch.zeros(self.n_lstm_layers, input.shape[0], self.n_hidden).requires_grad_()

        output, (hn, cn) = self.LSTM_layer(input, (h0.detach(), c0.detach()))

        r"""
            Index hidden state of last time step
            out.size() --> batch_size, last_seq_val, n_hidden
            out[:, -1, :] --> get the last sequence / time step
        """
        output = output[:, -1, :]
        output = self.fc_layers(output)

        return output


#data path
data_path = os.path.join('data', 'BTC-USD.csv')

#model path
dir="saved_model"
model_filename='NNTimeSeriesModel.pth.tar'
scaler_filename = 'Scaler.sav'


#read and return data as a dataframe
def read_data(path = data_path):
    data = pd.read_csv(path)
    data = pd.DataFrame(data)
    return data


#preprocess function to scale dataset
def scale_data_and_save_scale_model(raw_data, dir=dir, filename=scaler_filename):
    scaler = MinMaxScaler(feature_range=(0, 1))
    preprocessed_data = scaler.fit_transform(raw_data)

    #save scaler
    scaler_path = os.path.join(dir, filename)
    if not os.path.isfile(scaler_path):
        pickle.dump(scaler, open(scaler_path, 'wb'))
    else:
        pass

    return preprocessed_data


#inverse scaling
def inverse_scaler(data, dir=dir, filename=scaler_filename):
    path = os.path.join(dir, filename)
    scaler = pickle.load(open(path, 'rb'))
    data = scaler.inverse_transform(data)
    return data


#load, ppreprocess and splice data
def load_data(data, n_features, n_labels, test_size=0.3, Sequence_len=20):
    raw_data = data.values.reshape(-1, n_features)
    #scale data to be within range of (0, 1)
    scaled_data = scale_data_and_save_scale_model(raw_data)
    data = []
    for idx in range(len(scaled_data)-Sequence_len):
        data.append(scaled_data[idx:idx+Sequence_len])

    #split sequence data to features and labels
    data = np.array(data)
    features = data[:, :-1, :]
    labels = data[:, -1, :]

    #split data to Train and Test size
    X_train, X_eval, Y_train, Y_eval = train_test_split(features, labels, test_size=test_size, shuffle=False)

    return (X_train, X_eval, Y_train, Y_eval)


#save model and optimizer states
def save_model(model, optimizer, dir=dir, filename=model_filename):
    path = os.path.join(dir, filename)
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, path)
    print('model saved successfully')


#hyper_parameters
n_lstm_layers = 4
n_input = 1
n_output = 1
n_hidden = 64
EPOCHS = 50
batch_size = 100
lr = 1e-3

model = NNTimeSeriesModel(n_lstm_layers, n_input, n_hidden, n_output)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


#batch training process
def training_process(features, labels, batch_size=batch_size):
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()
    model.train()

    batch_loss = []
    for idx in tqdm(range(0, len(features), batch_size)):
        X, Y = features[idx:idx+batch_size], labels[idx:idx+batch_size]
        model.zero_grad()
        pred_Y = model(X)
        loss = criterion(pred_Y, Y)
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    print(f'mean loss: {np.mean(batch_loss)}')


#inference
def test_process(model_path, features, labels):
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()

    df = pd.DataFrame()

    state = torch.load(model_path)
    model.load_state_dict(state['model_state'])
    model.eval()
    
    with torch.no_grad():
        pred_Y = model(features)
        loss = criterion(pred_Y, labels)
        print(loss)
    
    df['actual_labels'] = inverse_scaler(labels.numpy()).reshape(-1)
    df['predicted_labels'] = inverse_scaler(pred_Y.numpy()).reshape(-1)

    print(df)

    plt.style.use('bmh')
    df.plot()
    plt.title('BTC-USD Stock Trend (2014-09-17 to 2020-06-23)')
    plt.ylabel('stock value')
    plt.xlabel('Daily Time Frame')
    plt.show()


#data
data = read_data(data_path)['Adj Close']
X_train, X_eval, Y_train, Y_eval = load_data(data, n_input, n_output)


#training call
if not os.path.isfile(os.path.join(dir, model_filename)):
    for epoch in range(EPOCHS):
        print(f'epoch: {epoch+1}')
        training_process(X_train, Y_train)

    save_model(model, optimizer)
else:
    pass

#inference call
test_process(os.path.join(dir, model_filename), X_eval, Y_eval)