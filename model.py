import torch
from torch import nn


class ModelForSimpleClassification(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super.__init__()
        self.linear1 = nn.Linear(in_dim, in_dim*4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.15)
        self.linear2 = nn.Linear(in_dim*4, out_dim*4)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.15)
        self.linear3 = nn.Linear(out_dim*4, out_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu1(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.dropout2(y)
        y = self.linear3(y)
        y = self.relu3(y)
        return y
