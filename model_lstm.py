import torch
import torch.nn as nn

class ModelLSTM(nn.Module):
    
    def __init__(self):
        super(ModelLSTM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(23, 511)
        self.fc2 = nn.Linear(1, 32)
        self.fc3 = nn.Linear(512, 1)
        self.lstm = nn.LSTMCell(512, 512)

    def forward(self, x):

        results = []

        #x = (self.relu((self.fc1(x))))
        #x = (self.relu(self.bn2(self.fc2(x))))
        #x = self.relu(self.bn3(self.fc3(x)))
        #x = self.fc4(x)

        emb = self.relu((self.fc1(x)))
        x = (torch.ones(x.size(0), 1) * (-0.8)).cuda()
        hx = torch.zeros(x.size(0), 512).cuda()
        cx = torch.zeros(x.size(0), 512).cuda()
        for t in range(36):
            #x = self.relu(self.fc2(x))
            x = torch.cat((emb, x), 1)
            hx, cx = self.lstm(x, (hx, cx))
            x = self.fc3(hx)
            results.append(x)
            x = x.detach()

        return results

