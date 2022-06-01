import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch
import random


class GRU(torch.nn.Module):
    def __init__(self, fe, ml, co = 50, ks = 25, st = 20, go = 20,do = 0.009999999776482582):


        super(GRU, self).__init__()

        self.go = go
        self.input_size = 1
        self.bidirectional = True
        self.do = do
        self.fe = torch.tensor([fe])
        self.convolution = torch.nn.Conv1d(self.fe, co, kernel_size=ks, stride=st)
        out = int(((ml - int(ks))/st)+1)
        self.GRU = torch.nn.GRU(co, go, bidirectional=True, num_layers = 2, dropout = do)
        lin_in = go*2 * out
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
        hidden = (torch.zeros(4, batch_size, self.go).numpy())
        x, hidden = self.GRU(x, torch.tensor(hidden))
        #x = torch.squeeze(x,1)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, -1)
        x  = F.dropout(x, p=self.do, training=self.training)
        x = self.linear(x)

        return x
