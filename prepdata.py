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

num_samples = 50000
num_proteins = 6
num_types = 8
space_size = 36
num_inputs = 23
data_size = num_samples * num_types
num_outputs = 6 * space_size
para_grid = loadmat(os.path.join(args.input_dir, 'para_grid.mat'))['para_grid'][:num_samples]
para_grid_jBC = loadmat(os.path.join(args.input_dir, 'para_grid_jBC.mat'))['para_grid_jBC'][:num_samples]
k = loadmat(os.path.join(args.input_dir, 'para_grid_k.mat'))['k'].transpose()[:num_samples]
para_grid_ki = loadmat(os.path.join(args.input_dir, 'para_grid_ki.mat'))['para_grid_ki'][:num_samples]
input_data_WT = np.concatenate((para_grid, para_grid_jBC, k, para_grid_ki, 100.0 * np.ones((num_samples, 1), dtype=np.float64)), 1)
input_data = np.zeros((data_size, num_inputs), dtype=np.float64)

output_vars = loadmat(os.path.join(args.input_dir, 'modeldata_0wto5w.mat'))
output_data = np.zeros((data_size, space_size, num_proteins), dtype=np.float64)
ignore_keys = ['__header__', '__version__', '__globals__', 'MBC_Ahet', 'MBMP_Ahet', 'MBN_Ahet', 'MChd_Ahet', 'MNog_Ahet', 'MSzd_Ahet'] 
WT_keys = ['MBC_WT', 'MBMP_WT', 'MBN_WT', 'MChd_WT', 'MNog_WT', 'MSzd_WT']
Chet_keys = ['MBC_Chet', 'MBMP_Chet', 'MBN_Chet', 'MChd_Chet', 'MNog_Chet', 'MSzd_Chet']
CLF_keys = ['MBC_CLF', 'MBMP_CLF', 'MBN_CLF', 'MChd_CLF', 'MNog_CLF', 'MSzd_CLF']
NLF_keys = ['MBC_NLF', 'MBMP_NLF', 'MBN_NLF', 'MChd_NLF', 'MNog_NLF', 'MSzd_NLF']
ALF_keys = ['MBC_ALF', 'MBMP_ALF', 'MBN_ALF', 'MChd_ALF', 'MNog_ALF', 'MSzd_ALF']
TLF_keys = ['MBC_TLF', 'MBMP_TLF', 'MBN_TLF', 'MChd_TLF', 'MNog_TLF', 'MSzd_TLF']
TALF_keys = ['MBC_TALF', 'MBMP_TALF', 'MBN_TALF', 'MChd_TALF', 'MNog_TALF', 'MSzd_TALF']
SLF_keys = ['MBC_SLF', 'MBMP_SLF', 'MBN_SLF', 'MChd_SLF', 'MNog_SLF', 'MSzd_SLF']

i = 0
input_data[i*num_samples : (i+1)*num_samples] = input_data_WT
for key in WT_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, WT_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 1
input_data_Chet = input_data_WT.copy()
input_data_Chet[:, 18] *= 0.5
input_data[i*num_samples : (i+1)*num_samples] = input_data_Chet
for key in Chet_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, Chet_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 2
input_data_CLF = input_data_WT.copy()
input_data_CLF[:, 18] = 0.0
input_data[i*num_samples : (i+1)*num_samples] = input_data_CLF
for key in CLF_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, CLF_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 3
input_data_NLF = input_data_WT.copy()
input_data_NLF[:, 8] = 0.0
input_data[i*num_samples : (i+1)*num_samples] = input_data_NLF
for key in NLF_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, NLF_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 4
input_data_ALF = input_data_WT.copy()
input_data_ALF[:, 15] = 0.0
input_data_ALF[:, 16] = 0.0
input_data[i*num_samples : (i+1)*num_samples] = input_data_ALF
for key in ALF_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, ALF_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 5
input_data_TLF = input_data_WT.copy()
input_data_TLF[:, 13] = 0.0
input_data_TLF[:, 14] = 0.0
input_data[i*num_samples : (i+1)*num_samples] = input_data_TLF
for key in TLF_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, TLF_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 6
input_data_TALF = input_data_WT.copy()
input_data_TALF[:, 13] = 0.0
input_data_TALF[:, 14] = 0.0
input_data_TALF[:, 15] = 0.0
input_data_TALF[:, 16] = 0.0
input_data[i*num_samples : (i+1)*num_samples] = input_data_TALF
for key in TALF_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, TALF_keys.index(key)] = output_vars[key].transpose()[:num_samples]
i = 7
input_data_SLF = input_data_WT.copy()
input_data_SLF[:, 22] = 0.0
input_data[i*num_samples : (i+1)*num_samples] = input_data_SLF
for key in SLF_keys:
    output_data[i*num_samples : (i+1)*num_samples, :, SLF_keys.index(key)] = output_vars[key].transpose()[:num_samples]

input_data = normalize_data(input_data)
output_data = normalize_data(output_data)
ind = torch.randperm(data_size)
input_data = input_data[ind]
output_data = output_data[ind]
train_size = int(0.9 * data_size)
train_data_input = input_data[:train_size]
train_data_output = output_data[:train_size]
val_data_input = input_data[train_size:]
val_data_output = output_data[train_size:]
torch.save(train_data_input, os.path.join(args.output_dir, 'train_input.pth'))
torch.save(train_data_output, os.path.join(args.output_dir, 'train_output.pth'))
torch.save(val_data_input, os.path.join(args.output_dir, 'val_input.pth'))
torch.save(val_data_output, os.path.join(args.output_dir, 'val_output.pth'))

