import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch




class LSTM(torch.nn.Module):

    def __init__(self, num_features, max_length,   conv_out=9, dropout=0.5,
                     lstm_out=14):

        super(LSTM, self).__init__()

        self.dropout = 0.5
        self.lstm_out = lstm_out
        self.database_num_features = torch.tensor([num_features])
        self.convolution = torch.nn.Conv1d(self.database_num_features, 9, kernel_size=16, stride=9)
        self.lstm1 = torch.nn.LSTM(9, lstm_out, bidirectional=True, num_layers =2, dropout = 0.5)
        conv_out  = int(((max_length - int(16))/9)+1)
        in_val = lstm_out * 2 * conv_out
        self.linear = torch.nn.Linear(in_val, 1)
        self.softmax = torch.nn.Softmax(dim = 1)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        z_data = data.z
        x = [data.x.numpy() for data in data.to_data_list()]
        batch_size = len(x)
        x = torch.tensor(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.convolution(x))
        x = x.permute(2, 0, 1)

        hidden = (torch.zeros(4, batch_size, self.lstm_out),
                  torch.zeros(4, batch_size, self.lstm_out))

        x, hidden = self.lstm1(x, hidden)

        x = x.permute(1,0,2)
        x = x.contiguous().view(batch_size, -1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x


