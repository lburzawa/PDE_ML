import os
import argparse
import numpy as np
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Parsing simulation data')
parser.add_argument('--input_dir', default='', type=str, help='path to input data')
parser.add_argument('--output_dir', default='', type=str, help='path to output data')
args = parser.parse_args()

data_size = 50000
space_size = 36
num_inputs = 22
num_outputs = 48 * space_size
para_grid = loadmat(os.path.join(args.input_dir, 'para_grid.mat'))['para_grid'][:data_size]
para_grid_jBC = loadmat(os.path.join(args.input_dir, 'para_grid_jBC.mat'))['para_grid_jBC'][:data_size]
k = loadmat(os.path.join(args.input_dir, 'para_grid_k.mat'))['k'].transpose()[:data_size]
para_grid_ki = loadmat(os.path.join(args.input_dir, 'para_grid_ki.mat'))['para_grid_ki'][:data_size]
data = np.concatenate((para_grid, para_grid_jBC, k, para_grid_ki), 1) #.astype(np.float32)

output_vars = loadmat(os.path.join(args.input_dir, 'modeldata_0wto5w.mat'))
output_data = np.zeros((data_size, num_outputs), dtype=np.float64)
ignore_keys = ['__header__', '__version__', '__globals__', 'MBC_Ahet', 'MBMP_Ahet', 'MBN_Ahet', 'MChd_Ahet', 'MNog_Ahet', 'MSzd_Ahet'] 
i = 0
for key, value in output_vars.items():
    if key not in ignore_keys:
        output_data[:, i*space_size : (i+1)*space_size] = value.transpose()[:data_size] #.astype(np.float32)
        i += 1

data = np.concatenate((data, output_data), 1)
print(data.shape)
#print((data < 0.0).sum())
data[abs(data) < 1e-5] = 1e-5
data = np.log10(data)
data /= 10.0
data = data.astype(np.float32)
np.random.shuffle(data)

train_size = int(0.8 * data_size)
train_data = data[:train_size]
val_data = data[train_size:]
train_data_input = train_data[:, :num_inputs]
train_data_output = train_data[:, num_inputs:]
val_data_input = val_data[:, :num_inputs]
val_data_output = val_data[:, num_inputs:]
np.savetxt(os.path.join(args.output_dir, 'train_input.csv'), train_data_input, delimiter=',')
np.savetxt(os.path.join(args.output_dir, 'train_output.csv'), train_data_output, delimiter=',')
np.savetxt(os.path.join(args.output_dir, 'val_input.csv'), val_data_input, delimiter=',')
np.savetxt(os.path.join(args.output_dir, 'val_output.csv'), val_data_output, delimiter=',')

