import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from myutils.logger import logger


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
    

def train(
    model: ModelForSimpleClassification,
    num_epochs: float,
    batch_size: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor
):
    model.train()
    dataset = TensorDataset(train_features, train_labels)
    dataLoader = DataLoader(dataset, batch_size)
    trainer = torch.optim.AdamW(model.parameters())
    lossFunction = torch.nn.MSELoss()
    for i in range(num_epochs):
        for x, label in dataLoader:
            loss = lossFunction(model(x), label)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
        l = loss(model(train_features), train_labels)
        logger.info(f'第{i}个epoch，整体损失为{l}')
    model.eval()
