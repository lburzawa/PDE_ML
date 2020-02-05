import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd


def normalize_data(data):
    data = torch.FloatTensor(data)
    data = torch.abs(data)
    data[abs(data) < 1e-8] = 1e-8
    data = torch.log10(data)
    data /= 10.0
    return data


class CSVdata(data.Dataset):

    def __init__(self, data_path):
        self.num_inputs = 23
        self.num_outputs = 36
        data = pd.read_csv(data_path, header=None)
        self.mutation_list = data.iloc[:, -1]
        data = data.iloc[:, :-1].to_numpy()
        self.error_data = torch.from_numpy(data[:, -1])
        data = data[:, :-1]
        data = normalize_data(data)
        self.input_data = data[:, :self.num_inputs].clone()
        self.output_data = data[:, self.num_inputs : self.num_inputs + self.num_outputs].clone()

        self.sstot = (self.output_data - self.output_data.mean()).pow(2).sum().item()
        #print(output_data.mean())
        
    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index], self.error_data[index], self.mutation_list[index]
    
    def __len__(self):
        return self.input_data.size(0)

