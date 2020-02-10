import argparse
import numpy as np
from scipy.io import loadmat
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser(description='Parsing simulation data')
parser.add_argument('--input_dir', default='', type=str, help='path to input data')
parser.add_argument('--output_dir', default='', type=str, help='path to output data')
args = parser.parse_args()


def read_exp_data(exp_data, var_name):
    exp_data = exp_data[var_name][0]
    ref = (np.sort(exp_data)[-5:]).mean()
    exp_data = exp_data[:16]
    return exp_data, ref


def calculate_error(sim_data, exp_data, ref_sim, ref_exp):
    sim_data = sim_data * (ref_exp / ref_sim)
    sim_data = sim_data[0:32:2]
    error = np.sqrt(np.power(sim_data - exp_data, 2).mean()) / 61.9087
    return error


input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
files = ['1to5w/modeldata_0wto5w.mat', '5wto10w/modeldata_5wto10w.mat', '10wto15w/modeldata_10wto15w.mat', '15wto20w/modeldata_15wto20w.mat',
         '20wto25w/modeldata_20wto25w.mat', '25wto30w/modeldata_25wto30w.mat']
num_samples = 50000
num_mutations = 7  # number of mutations per sample
num_inputs = 23  # how many parameters does the model have
data_size = len(files) * num_samples * num_mutations  # total size of data
num_outputs = 36  # how many points in space for final protein distribution
para_grid = loadmat(input_dir / '1to5w' / 'para_grid.mat')['para_grid'][:len(files) * num_samples]
para_grid_jBC = loadmat(input_dir / '1to5w' / 'para_grid_jBC.mat')['para_grid_jBC'][:len(files) * num_samples]
para_grid_k = loadmat(input_dir / '1to5w' / 'para_grid_k.mat')['k'].transpose()[:len(files) * num_samples]
para_grid_ki = loadmat(input_dir / '1to5w' / 'para_grid_ki.mat')['para_grid_ki'][:len(files) * num_samples]
Vs = 100.0 * np.ones((len(files) * num_samples, 1), dtype=np.float64)  # one of the parameters used in simulations
input_data_WT = np.concatenate((para_grid, para_grid_jBC, Vs, para_grid_ki, para_grid_k), 1)
input_data = np.zeros((data_size, num_inputs), dtype=np.float64)
output_data = np.zeros((data_size, num_outputs), dtype=np.float64)
error_data = np.zeros((data_size, 1), dtype=np.float64)
BMP_keys = ['MBMP_WT', 'MBMP_CLF', 'MBMP_NLF', 'MBMP_ALF', 'MBMP_TLF', 'MBMP_TALF', 'MBMP_SLF']
exp_data = loadmat('pSmad_WT_MT_new.mat')
WT_exp, ref_exp = read_exp_data(exp_data, 'pWT_57')
CLF_exp, _ = read_exp_data(exp_data, 'pCLF_57')
ALF_exp, _ = read_exp_data(exp_data, 'pALF_57')
TLF_exp, _ = read_exp_data(exp_data, 'pTLF_57')
TALF_exp, _ = read_exp_data(exp_data, 'pTALF_57')
SLF_exp, _ = read_exp_data(exp_data, 'pSLF_57')
mutation_list = [None] * data_size
order = np.arange(len(files) * num_samples)
np.random.shuffle(order)

for j in range(len(files)):
    output_vars = loadmat(input_dir / files[j])

    WT_sim = output_vars['MBMP_WT'].transpose()[:num_samples]
    CLF_sim = output_vars['MBMP_CLF'].transpose()[:num_samples]
    NLF_sim = output_vars['MBMP_NLF'].transpose()[:num_samples]
    ALF_sim = output_vars['MBMP_ALF'].transpose()[:num_samples]
    TLF_sim = output_vars['MBMP_TLF'].transpose()[:num_samples]
    TALF_sim = output_vars['MBMP_TALF'].transpose()[:num_samples]
    SLF_sim = output_vars['MBMP_SLF'].transpose()[:num_samples]

    for i in range(num_samples):
        ind = order[j*num_samples+i]
        # WT data
        inputs = input_data_WT[j*num_samples+i].copy()
        outputs = WT_sim[i]
        ref_sim = (np.sort(outputs)[-5:]).mean()
        error = calculate_error(outputs, WT_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 0] = inputs
        output_data[ind * num_mutations + 0] = outputs
        error_data[ind * num_mutations + 0, 0] = error
        mutation_list[ind * num_mutations + 0] = 'WT'
        # CLF data
        inputs = input_data_WT[j*num_samples+i].copy()
        inputs[18] = 0.0
        outputs = CLF_sim[i]
        error = calculate_error(outputs, CLF_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 1] = inputs
        output_data[ind * num_mutations + 1] = outputs
        error_data[ind * num_mutations + 1, 0] = error
        mutation_list[ind * num_mutations + 1] = 'CLF'
        # NLF data
        inputs = input_data_WT[j*num_samples+i].copy()
        inputs[8] = 0.0
        outputs = NLF_sim[i]
        error = calculate_error(outputs, WT_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 2] = inputs
        output_data[ind * num_mutations + 2] = outputs
        error_data[ind * num_mutations + 2, 0] = error
        mutation_list[ind * num_mutations + 2] = 'NLF'
        # ALF data
        inputs = input_data_WT[j*num_samples+i].copy()
        inputs[15] = 0.0
        inputs[16] = 0.0
        outputs = ALF_sim[i]
        error = calculate_error(outputs, ALF_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 3] = inputs
        output_data[ind * num_mutations + 3] = outputs
        error_data[ind * num_mutations + 3, 0] = error
        mutation_list[ind * num_mutations + 3] = 'ALF'
        # TLF data
        inputs = input_data_WT[j*num_samples+i].copy()
        inputs[13] = 0.0
        inputs[14] = 0.0
        outputs = TLF_sim[i]
        error = calculate_error(outputs, TLF_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 4] = inputs
        output_data[ind * num_mutations + 4] = outputs
        error_data[ind * num_mutations + 4, 0] = error
        mutation_list[ind * num_mutations + 4] = 'TLF'
        # TALF data
        inputs = input_data_WT[j*num_samples+i].copy()
        inputs[13] = 0.0
        inputs[14] = 0.0
        inputs[15] = 0.0
        inputs[16] = 0.0
        outputs = TALF_sim[i]
        error = calculate_error(outputs, TALF_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 5] = inputs
        output_data[ind * num_mutations + 5] = outputs
        error_data[ind * num_mutations + 5, 0] = error
        mutation_list[ind * num_mutations + 5] = 'TALF'
        # SLF data
        inputs = input_data_WT[j*num_samples+i].copy()
        inputs[19] = 0.0
        outputs = SLF_sim[i]
        error = calculate_error(outputs, SLF_exp, ref_sim, ref_exp)
        input_data[ind * num_mutations + 6] = inputs
        output_data[ind * num_mutations + 6] = outputs
        error_data[ind * num_mutations + 6, 0] = error
        mutation_list[ind * num_mutations + 6] = 'SLF'


data = np.concatenate((input_data, output_data, error_data), 1)
print(output_data.min())
data = pd.DataFrame(data)
data[num_inputs + num_outputs + 1] = mutation_list
#data = data.sample(frac=1)
train_size = int(0.9 * num_samples * len(files))
train_data = data[:train_size * num_mutations]
val_data = data[train_size * num_mutations:]
train_data.to_csv(output_dir / 'train_data.csv', header=False, index=False)
val_data.to_csv(output_dir / 'val_data.csv', header=False, index=False)
