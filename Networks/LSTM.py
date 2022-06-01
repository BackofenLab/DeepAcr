import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch

class LSTM(torch.nn.Module):
    def __init__(self, nf, ml, ks=16, st=9, co=9, dropout=0.5,
                     lo=14):

        super(LSTM, self).__init__()

        self.dropout = dropout
        self.lo = lo
        self.database_num_features = torch.tensor([nf])
        self.convolution = torch.nn.Conv1d(self.database_num_features, co, kernel_size=ks, stride=st)
        self.lstm1 = torch.nn.LSTM(co, lo, bidirectional=True, num_layers =2, dropout = dropout)
        co  = int(((ml - int(ks))/st)+1)
        in_val = lo * 2 * co
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

        hidden = (torch.zeros(4, batch_size, self.lo),
                  torch.zeros(4, batch_size, self.lo))

        x, hidden = self.lstm1(x, hidden)

        x = x.permute(1,0,2)
        x = x.contiguous().view(batch_size, -1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x

