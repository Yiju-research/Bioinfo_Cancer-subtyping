import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# AE
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, latent_size)
        self.decoder = nn.Linear(latent_size, input_size)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# SNF
def snf(matrix1, matrix2):
    matrix = (matrix1 + matrix2) / 2
    return matrix

# GCN
class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_size, output_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, adjacency, features):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adjacency, support)
        return output

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, adjacency, features):
        hidden = self.activation(self.gc1(adjacency, features))
        output = self.gc2(adjacency, hidden)
        return output

