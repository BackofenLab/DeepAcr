import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import torch
import random



class GRUb(torch.nn.Module):
    def __init__(self, nf,ml, co = 50, ks = 17, st = 4,go = 16, hn = 100, hn2 = 83 ,dropout = 0.2104):

        #    def __init__(self, num_features,max_length, conv_out = 50, kernel_size = 17, stride = 4,gru_out = 16, hidden_nodes = 100, hidden_nodes2 = 83 ,dropout = 0.2104):

        super(GRUb, self).__init__()

        self.go = go
        self.bidirectional = True
        self.dropout = dropout
        self.database_nf = torch.tensor([nf])

        self.convolution = torch.nn.Conv1d(self.database_nf, co, kernel_size=ks, stride=st)
        out = int(((ml - int(ks))/st)+1)
        self.dropout = dropout
        self.GRU = torch.nn.GRU(co, go, bidirectional=True, dropout = dropout, num_layers = 2)
        lin_in = go*2 * out
        self.linear = torch.nn.Linear(lin_in + hn2, 1)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.lin1 = torch.nn.Linear(12,  hn)
        self.lin2 = torch.nn.Linear(hn,  hn2)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z_data = data.z
        x = [data.x.numpy() for data in data.to_data_list()]
        x = [data.x.numpy().tolist() for num, data in enumerate(data.to_data_list())]
        batch_size = len(x)
        z_data = z_data.reshape(batch_size,-1)
        z_data = F.relu(self.lin1(z_data))
        z_data  = F.dropout(z_data, p=self.dropout, training=self.training)
        z_data = F.relu(self.lin2(z_data))
        batch_size = len(x)
        x = torch.tensor(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.convolution(x))
        x = x.permute(2, 0, 1)
        hidden = (torch.zeros(4, batch_size, self.go).numpy())
        x, hidden = self.GRU(x, torch.tensor(hidden))
        #raise NotImplementedError
        #x = torch.squeeze(x,1)
        #print(x.size())
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, -1)
        x = torch.cat([x, z_data], dim=1)
        x  = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)

        return x


