import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch
import random



class GRU(torch.nn.Module):
    def __init__(self, num_features, max_length):


        super(GRU, self).__init__()

        self.gru_out = 20
        self.input_size = 1
        self.bidirectional = True
        self.dropout = 0.009999999776482582
        self.database_num_features = torch.tensor([num_features])
        self.convolution = torch.nn.Conv1d(self.database_num_features, 50, kernel_size=25, stride=20)
        out = int(((max_length - 25)/20)+1)
        self.GRU = torch.nn.GRU(50, 20, bidirectional=True, num_layers = 2, dropout = 0.009999999776482582)
        lin_in = 20*2 * out
        self.linear = torch.nn.Linear(lin_in, 1)
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
        hidden = (torch.zeros(4, batch_size, self.gru_out).numpy())
        x, hidden = self.GRU(x, torch.tensor(hidden))
        #x = torch.squeeze(x,1)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, -1)
        x  = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x


