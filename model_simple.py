import torch
import torch.nn as nn

class ModelSimple(nn.Module):
    
    def __init__(self):
        super(ModelSimple, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(22, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(256, 6)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 216)

    def forward(self, x):

        x = (self.relu((self.fc1(x))))
        x = (self.relu((self.fc2(x))))
        #x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x

