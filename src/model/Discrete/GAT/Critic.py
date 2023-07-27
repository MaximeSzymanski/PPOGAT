import torch.nn as nn
from src.model.Discrete.GAT.GATLayer import GAT_layer
import torch
from torch.nn import functional as F
class GATCritic(nn.Module):
    def __init__(self, nfeat, nhid, noutput,dropout, alpha_leaky_relu,n_heads,num_nodes,device,training=True):
        super(GATCritic, self).__init__()
        self.elu = nn.ELU()
        self.device = device
        self.attentions1 = GAT_layer(nfeat, nhid, n_heads=n_heads, dropout=dropout, leaky_relu_negative_slope=alpha_leaky_relu, is_concat=True)
        self.attentions2 = GAT_layer(nhid , nhid, n_heads=n_heads, dropout=dropout, leaky_relu_negative_slope=alpha_leaky_relu, is_concat=True)
        self.attentions3 = GAT_layer(nhid , nhid, n_heads=n_heads, dropout=dropout, leaky_relu_negative_slope=alpha_leaky_relu, is_concat=True)
        self.training = training
        self.output_layer = GAT_layer(nhid , 1, n_heads=1, dropout=dropout, leaky_relu_negative_slope=alpha_leaky_relu, is_concat=False)
        self.fc1 = nn.Linear(noutput *num_nodes*num_nodes , 64)
        self.fc2 = nn.Linear(64, 1)
        # Put layer in the device
        self.to(device)


    def forward(self, x):

        # x is a list of (state, action) pairs
        # get all state, action pairs
        x,adj = x
        x = torch.tensor(x, dtype=torch.float32, requires_grad=False, device=self.device)
        adj = torch.tensor(adj, dtype=torch.float32, requires_grad=False, device=self.device)
        adj = adj.unsqueeze(-1)

        x = self.attentions1(x, adj)
        x = self.elu(x)
        x = self.attentions2(x, adj)
        x = self.elu(x)
        x = self.attentions3(x, adj)
        x = self.elu(x)
        x = self.output_layer(x, adj)
        x = x.view(-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x