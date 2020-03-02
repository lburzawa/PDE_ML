import argparse
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser(description='Simulation Data Training')
parser.add_argument('--target_data', default='', type=str, help='path to dataset')
args = parser.parse_args()

mutation_strings = ['WT', 'CLF', 'NLF', 'ALF', 'TLF', 'TALF', 'SLF']
x_values = np.arange(36)
targets = pd.read_csv(args.target_data, header=None)
targets = targets.iloc[:, 23:23+36].to_numpy()
targets = np.log10(targets)
outputs = np.loadtxt('./results.csv', delimiter=',')

'''
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

sim_data = outputs[:, -2]
nn_data = outputs[:, -1]
WT_sim = sim_data[0::7]
WT_nn = nn_data[0::7]
CLF_sim = sim_data[1::7]
CLF_nn = nn_data[1::7]
ind = np.float64(WT_sim<1.0) * np.float64(WT_nn<1.0) * np.float64(CLF_sim<1.0) * np.float64(CLF_nn<1.0)
WT_sim = WT_sim * ind
WT_nn = WT_nn * ind
CLF_sim = CLF_sim * ind
CLF_nn = CLF_nn * ind
WT_sim = WT_sim[WT_sim>0.0]
WT_nn = WT_nn[WT_nn>0.0]
CLF_sim = CLF_sim[CLF_sim>0.0]
CLF_nn = CLF_nn[CLF_nn>0.0]

plt.xlabel('Simulation NRMSE')
plt.ylabel('Neural network NRMSE')
plt.title('WT results')
#plt.grid()
axes = plt.gca()
axes.set_ylim(0.0, 1.0)
axes.set_xlim(0.0, 1.0)
x, y = WT_sim.copy(), WT_nn.copy()
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
#plt.plot(WT_sim, WT_nn, '.')
#fig, ax = plt.subplots()
plt.scatter(x, y, c=z) #, s=50) #, edgecolor='')
#plt.colorbar()
plt.savefig('./WT_comp.png')
plt.clf()

axes = plt.gca()
axes.set_ylim(0.0, 1.0)
axes.set_xlim(0.0, 1.0)
x, y = CLF_sim.copy(), CLF_nn.copy()
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x, y, c=z)
plt.xlabel('Simulation NRMSE')
plt.ylabel('Neural network NRMSE')
plt.title('CLF results')
plt.savefig('./CLF_comp.png')
plt.clf()

axes = plt.gca()
axes.set_ylim(0.0, 1.0)
axes.set_xlim(0.0, 1.0)
x, y = WT_sim.copy(), CLF_sim.copy()
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x, y, c=z)
plt.xlabel('WT NRMSE')
plt.ylabel('CLF NRMSE')
plt.title('PDE simulation')
plt.savefig('./plot_pareto_sim_all.png')
plt.clf()

axes = plt.gca()
axes.set_ylim(0.0, 1.0)
axes.set_xlim(0.0, 1.0)
x, y = WT_nn.copy(), CLF_nn.copy()
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
plt.scatter(x, y, c=z)
plt.xlabel('WT NRMSE')
plt.ylabel('CLF NRMSE')
plt.title('Neural network model')
plt.savefig('./plot_pareto_nn_all.png')
plt.clf()

for j in range(2):
    if j==0:
        data = sim_data
    else:
        data = nn_data
    WT = data[0::7]
    CLF = data[1::7]
    NLF = data[2::7]
    ALF = data[3::7]
    TLF = data[4::7]
    TALF = data[5::7]
    SLF = data[6::7]
    ind = np.float64(WT<0.2) * np.float64(CLF<0.2) * np.float64(NLF<0.2) * np.float64(ALF<0.2) * np.float64(TLF<0.2) * np.float64(TALF<0.2) * np.float64(SLF<0.2)
    WT = WT * ind
    CLF = CLF * ind
    WT = WT[WT>0.0]
    CLF = CLF[CLF>0.0]

    area = 0.0
    step = 0.01
    WT_front = []
    CLF_front = []
    for i in np.flip(np.arange(0.0, 0.2, step)):
        ind = np.logical_and(CLF >= i, CLF < i + step)
        WT_vals = WT[ind]
        CLF_vals = CLF[ind]
        if WT_vals.size > 0:
            ind = np.argmin(WT_vals)
            WT_front.append(WT_vals[ind])
            CLF_front.append(CLF_vals[ind])
            area += step * WT_vals[ind]
    for i in np.arange(0.0, 0.2, step):
        ind = np.logical_and(WT >= i, WT < i+step)
        WT_vals = WT[ind]
        CLF_vals = CLF[ind]
        if WT_vals.size > 0:
            ind = np.argmin(CLF_vals)
            WT_front.append(WT_vals[ind])
            CLF_front.append(CLF_vals[ind])
            area += step * CLF_vals[ind]
    WT_front = np.float64(WT_front)
    CLF_front = np.float64(CLF_front)

    plt.xlabel('WT NRMSE')
    plt.ylabel('CLF NRMSE')
    if j==0:
        plt.title('PDE simulation')
    else:
        plt.title('Neural network model')
    plt.grid()
    axes = plt.gca()
    axes.set_ylim(0.0, 0.2)
    axes.set_xlim(0.0, 0.2)
    plt.plot(WT, CLF, '.')
    plt.plot(WT_front, CLF_front)
    if j==0:
        plt.savefig('./plot_pareto_sim.png')
    else:
        plt.savefig('./plot_pareto_nn.png')
    plt.clf()

    print(area)