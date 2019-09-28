import torch
import torch.nn as nn

class ModelNet(nn.Module):
    
    def __init__(self):
        super(ModelNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(22, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(6, 128)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(256, 6)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 216)
        self.lstm = nn.LSTMCell(256, 256)

    def forward(self, x):

        results = []

        #x = (self.relu((self.fc1(x))))
        #x = (self.relu(self.bn2(self.fc2(x))))
        #x = self.relu(self.bn3(self.fc3(x)))
        #x = self.fc4(x)

        emb = self.relu(self.bn1(self.fc1(x)))
        x = (torch.ones(x.size(0), 6) * (-0.5)).cuda()
        hx = torch.zeros(x.size(0), 256).cuda()
        cx = torch.zeros(x.size(0), 256).cuda()
        for t in range(36):
            x = self.relu((self.fc2(x)))
            x = torch.cat((emb, x), 1)
            hx, cx = self.lstm(x, (hx, cx))
            x = self.fc3(hx)
            results.append(x)
            x = x.detach()

        return results

