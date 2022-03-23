import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch



class LSTMb(torch.nn.Module):

    def __init__(self, num_features,max_length, kernel_size = 15, stride = 10, conv_out = 20, dropout = 0.5, lstm_out = 10, lin1_nodes = 100, lin2_nodes = 41):

        super(LSTMb, self).__init__()
        self.database_num_features = torch.tensor([num_features])
        self.convolution = torch.nn.Conv1d(self.database_num_features, conv_out, kernel_size=kernel_size, stride=stride)
        self.lstm_out = lstm_out
        self.lstm1 = torch.nn.LSTM(conv_out, lstm_out, bidirectional=True, dropout = dropout, num_layers = 2)
        conv_out  = int(((max_length - int(kernel_size))/stride)+1)
        in_val = lstm_out * 2 * conv_out
        self.linear = torch.nn.Linear(100, 1)
        self.lin1_add = torch.nn.Linear(12, lin1_nodes)
        self.lin2_add = torch.nn.Linear(lin1_nodes, lin2_nodes)
        self.linear = torch.nn.Linear(in_val+ lin2_nodes , 1)


    def forward(self, data):

        x, edge_index, batch, z_data = data.x, data.edge_index, data.batch, data.z
        x = [data.x.numpy() for data in data.to_data_list()]
        batch_size = len(x)
        z_data = z_data.reshape(batch_size,-1)
        x = torch.tensor(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.convolution(x))
        x = x.permute(2, 0, 1)

        hidden = (torch.zeros(4, batch_size, self.lstm_out),
                  torch.zeros(4, batch_size, self.lstm_out))

        x, hidden = self.lstm1(x, hidden)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, -1)
        z_data = F.relu(self.lin1_add(z_data))
        z_data = F.relu(self.lin2_add(z_data))
        x = torch.cat([x, z_data], dim=1)
        x = self.linear(x)


        return x


