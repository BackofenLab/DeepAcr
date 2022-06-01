import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch

class LSTMb(torch.nn.Module):

    def __init__(self, nf,ml, ks = 15, st = 10, co = 20, dropout = 0.5, lo = 10, n1 = 100, n2 = 41):

        super(LSTMb, self).__init__()
        self.database_num_features = torch.tensor([nf])
        self.convolution = torch.nn.Conv1d(self.database_num_features, co, kernel_size=ks, stride=st)
        self.lo = lo
        self.lstm1 = torch.nn.LSTM(co, lo, bidirectional=True, dropout = dropout, num_layers = 2)
        co  = int(((ml - int(ks))/st)+1)
        in_val = lo * 2 * co
        self.linear = torch.nn.Linear(100, 1)
        self.lin1_add = torch.nn.Linear(12, n1)
        self.lin2_add = torch.nn.Linear(n1, n2)
        self.linear = torch.nn.Linear(in_val+ n2 , 1)


    def forward(self, data):

        x, edge_index, batch, z_data = data.x, data.edge_index, data.batch, data.z
        x = [data.x.numpy() for data in data.to_data_list()]
        batch_size = len(x)
        z_data = z_data.reshape(batch_size,-1)
        x = torch.tensor(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.convolution(x))
        x = x.permute(2, 0, 1)

        hidden = (torch.zeros(4, batch_size, self.lo),
                  torch.zeros(4, batch_size, self.lo))

        x, hidden = self.lstm1(x, hidden)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, -1)
        z_data = F.relu(self.lin1_add(z_data))
        z_data = F.relu(self.lin2_add(z_data))
        x = torch.cat([x, z_data], dim=1)
        x = self.linear(x)


        return x

