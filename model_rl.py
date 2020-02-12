import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, seq_len):
        super(Net,self).__init__()
        self.seq_len = seq_len
        self.relu = nn.ReLU(inplace=True)
        #self.tanh = nn.Tanh()
        #self.softplus = nn.Softplus()
        self.fc1 = nn.Linear(2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTMCell(128, 128)
        self.fc2 = nn.Linear(128, 32)
        #self.fc_mean = nn.Linear(128, 1)
        #self.fc_std = nn.Linear(128, 1)


    def forward(self, x, hidden):
        output_list = []
        log_probs_list = []
        entropy_list = []
        hx, cx = hidden
        #x = torch.tensor([[min_values[i], max_values[i]]]).cuda()
        x = self.relu((self.fc1(x)))
        (hx, cx) = self.lstm(x, (hx, cx))
        x = self.fc2(hx)
        hidden = [hx, cx]
        return x, hidden


    def init_hidden(self, batch_size):
        hx = torch.zeros(batch_size, 128).cuda()
        cx = torch.zeros(batch_size, 128).cuda()
        return [hx, cx]
