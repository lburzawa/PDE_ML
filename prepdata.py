import os
import argparse
import numpy as np
from scipy.io import loadmat
import torch

parser = argparse.ArgumentParser(description='Parsing simulation data')
parser.add_argument('--input_dir', default='', type=str, help='path to input data')
parser.add_argument('--output_dir', default='', type=str, help='path to output data')
args = parser.parse_args()

def normalize_data(data):
    data = np.absolute(data)
    data[abs(data) < 1e-5] = 1e-5
    data = np.log10(data)
    data /= 10.0
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    return data

data_size = 50000
num_proteins = 6
space_size = 36
num_inputs = 22
num_outputs = 6 * space_size
para_grid = loadmat(os.path.join(args.input_dir, 'para_grid.mat'))['para_grid'][:data_size]
para_grid_jBC = loadmat(os.path.join(args.input_dir, 'para_grid_jBC.mat'))['para_grid_jBC'][:data_size]
k = loadmat(os.path.join(args.input_dir, 'para_grid_k.mat'))['k'].transpose()[:data_size]
para_grid_ki = loadmat(os.path.join(args.input_dir, 'para_grid_ki.mat'))['para_grid_ki'][:data_size]
input_data = np.concatenate((para_grid, para_grid_jBC, k, para_grid_ki), 1)

output_vars = loadmat(os.path.join(args.input_dir, 'modeldata_0wto5w.mat'))
output_data = np.zeros((data_size, space_size, num_proteins), dtype=np.float64)
ignore_keys = ['__header__', '__version__', '__globals__', 'MBC_Ahet', 'MBMP_Ahet', 'MBN_Ahet', 'MChd_Ahet', 'MNog_Ahet', 'MSzd_Ahet'] 
add_keys = ['MBC_WT', 'MBMP_WT', 'MBN_WT', 'MChd_WT', 'MNog_WT', 'MSzd_WT']
i = 0
for key, value in output_vars.items():
    if key in add_keys:
        output_data[:, :, add_keys.index(key)] = value.transpose()[:data_size] #.astype(np.float32)
        #i += 1

input_data = normalize_data(input_data)
output_data = normalize_data(output_data)
ind = torch.randperm(data_size)
input_data = input_data[ind]
output_data = output_data[ind]
train_size = int(0.8 * data_size)
train_data_input = input_data[:train_size]
train_data_output = output_data[:train_size]
val_data_input = input_data[train_size:]
val_data_output = output_data[train_size:]
torch.save(train_data_input, os.path.join(args.output_dir, 'train_input.pth'))
torch.save(train_data_output, os.path.join(args.output_dir, 'train_output.pth'))
torch.save(val_data_input, os.path.join(args.output_dir, 'val_input.pth'))
torch.save(val_data_output, os.path.join(args.output_dir, 'val_output.pth'))

