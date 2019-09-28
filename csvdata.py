import os
import numpy as np
import torch
import torch.utils.data as data

def read_data(path):
    #data = np.loadtxt(path, dtype=np.float32, delimiter=',')
    #data = torch.from_numpy(data)
    data = torch.load(path)
    return data

class CSVdata(data.Dataset):

    def __init__(self, input_path, output_path):
        self.input_data = read_data(input_path)
        self.output_data = read_data(output_path)
        self.sstot = (self.output_data - self.output_data.mean()).pow(2).sum()
        #print(output_data.mean())
        
    def __getitem__(self, index):
        return self.input_data[index], self.output_data[index]
    
    def __len__(self):
        return self.input_data.size(0)


