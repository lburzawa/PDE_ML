import torch
import torch.nn as nn

class ModelNet(nn.Module):
    
    def __init__(self):
        super(ModelNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(22, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(2048, 1728)

    def forward(self, x):
        x = (self.relu(self.bn1(self.fc1(x))))
        x = (self.relu(self.bn2(self.fc2(x))))
        #x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

