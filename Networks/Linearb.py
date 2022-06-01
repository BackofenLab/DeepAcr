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







class Linearb(torch.nn.Module):
    def __init__(self, num_features, max_length, out1 = 197, out2 = 870, out3 = 71, out4 = 61, hidden_nodes = 81, dropout = 0.33015396090117194 ):
        super(Linearb, self).__init__()


        self.dropout = dropout

        self.linear1 = torch.nn.Linear(max_length * num_features, out1)
                
        self.linear2 = torch.nn.Linear(out1, out2)
        self.linear3 = torch.nn.Linear(out2, out3)
        self.linear4 = torch.nn.Linear(out3, out4)


        self.linear = torch.nn.Linear(out4, 1)


        self.lin1_add = torch.nn.Linear(12, hidden_nodes)
        self.lin2_add = torch.nn.Linear(hidden_nodes, hidden_nodes)
        self.lin3_add = torch.nn.Linear(hidden_nodes, hidden_nodes)
        self.lin_cat = torch.nn.Linear(
            out4 + hidden_nodes, out4)

        self.flatten = torch.nn.Flatten()

    def forward(self, data):


        x, edge_index, batch = data.x, data.edge_index, data.batch
        z_data = data.z
        x = [data.x.numpy() for data in data.to_data_list()]
        batch_size = len(x)
        x = torch.tensor(x)
        z_data = z_data.reshape(batch_size,-1)


        
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear4(x))

        z_data = F.relu(self.lin1_add(z_data))
        z_data = F.relu(self.lin2_add(z_data))
        z_data = F.dropout(z_data, p=self.dropout, training=self.training)
        z_data = F.relu(self.lin3_add(z_data))

        x = x.contiguous().view(batch_size, -1)
        x = torch.cat([x, z_data], dim=1)
        x = F.relu(self.lin_cat(x))
        x = self.linear(x)




        return x


