from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch_geometric.nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np
from torch_geometric.nn import Set2Set
import pandas as pd
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool as gap
import torchbnn as bnn
from torch_scatter import scatter
import random


class Linear(torch.nn.Module):
    def __init__(self, nf, ml, out1 = 548, out2= 36, out3 = 500, out4 = 115, dropout = 0.05226011067821744):
        super(Linear, self).__init__()


        self.dropout = dropout
        self.linear1 = torch.nn.Linear(ml * nf, out1)
        self.linear2 = torch.nn.Linear(out1, out2)
        self.linear3 = torch.nn.Linear(out2, out3)
        self.linear4 = torch.nn.Linear(out3, out4)
        self.linear5 = torch.nn.Linear(out4, 1)

        self.softmax = torch.nn.Softmax(dim = 1)

        self.flatten = torch.nn.Flatten()

    def forward(self, data):


        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = [data.x.numpy() for data in data.to_data_list()]
        batch_size = len(x)
        x = torch.tensor(x)
        x = self.flatten(x)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear4(x))
        x = self.linear5(x)



        return x


