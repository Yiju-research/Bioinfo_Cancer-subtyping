# The original code is written in R and this file is a rewrite of the same implementation.

# @Author  : Li Yiju

import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP architecture
class DeepCC(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=INPUT_SIZE, out_features=2000),
            nn.Tanh(),
            nn.Linear(in_features=2000, out_features=500),
            nn.Tanh(),
            nn.Linear(in_features=500, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=30),
            nn.Tanh(),
            nn.Linear(in_features=30, out_features=10),
            nn.Tanh(),
            nn.Linear(in_features=10, out_features=OUTPUT_SIZE),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

#TODO: Set input and output sizes
INPUT_SIZE = <input size>
OUTPUT_SIZE = <output size>

model = DeepCC()

# Xavier
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(weights_init)

# Optimizer
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_adadelta = optim.Adadelta(model.parameters())
