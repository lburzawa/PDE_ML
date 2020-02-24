import argparse
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd

parser = argparse.ArgumentParser(description='Simulation Data Training')
parser.add_argument('--target_data', default='', type=str, help='path to dataset')
args = parser.parse_args()

mutation_strings = ['WT', 'CLF', 'NLF', 'ALF', 'TLF', 'TALF', 'SLF']
x_values = np.arange(36)
targets = pd.read_csv(args.target_data, header=None)
targets = targets.iloc[:, 23:23+36].to_numpy()
targets = np.log10(targets)
outputs = np.loadtxt('./results.csv', delimiter=',')


outputs = outputs[:, :-2]
outputs = np.log10(outputs)
for i in range(6):
    plt.xlabel('X value')
    plt.ylabel('Concentration (log10 scale)')
    plt.grid()
    target = targets[i*7 + i]
    output = outputs[i*7 + i]
    plt.plot(x_values, target)
    plt.plot(x_values, output)
    plt.title('{:s} mutation'.format(mutation_strings[i]))
    plt.legend(['Simulation', 'NN'])
    plt.savefig('./plot{:d}.png'.format(i))
    plt.clf()


'''
WT = outputs[::7, -1]
CLF = outputs[1::7, -1]
ind = np.logical_and(WT < 1.0, CLF < 1.0)
WT = WT[ind]
CLF = CLF[ind]
plt.xlabel('WT NRMSE')
plt.ylabel('CLF NRMSE')
plt.title('Neural network model')
plt.grid()
axes = plt.gca()
axes.set_ylim(0.0, 1.0)
axes.set_xlim(0.0, 1.0)
plt.plot(WT, CLF, '.')
#plt.plot(x_values, output)
plt.savefig('./plot_pareto_nn.png')
'''