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

A=np.loadtxt('./save_rl/test.txt', delimiter=',')
plt.plot(A[:, 0], A[:, 2])
plt.xlabel('Number of simulations in 100s')
plt.ylabel('Mean error over 100 simulations')
plt.grid()
plt.savefig('./plot_rl.png')

mutation_strings = ['WT', 'CLF', 'NLF', 'ALF', 'TLF', 'TALF', 'SLF']
x_values = np.arange(36)
targets = pd.read_csv(args.target_data, header=None)
targets = targets.iloc[:, 23:23+36].to_numpy()
targets = np.log10(targets)
outputs = np.loadtxt('./results.csv', delimiter=',')

BMP_outputs = outputs[:, :-2]
BMP_outputs = np.log10(BMP_outputs)
for i in range(7):
    plt.xlabel('X value')
    plt.ylabel('Concentration (log10 scale)')
    plt.grid()
    target = targets[i*7 + i]
    output = BMP_outputs[i*7 + i]
    plt.plot(x_values, target)
    plt.plot(x_values, output)
    plt.title('{:s} mutation'.format(mutation_strings[i]))
    plt.legend(['Simulation', 'NN'])
    plt.savefig('./plot{:d}.png'.format(i))
    plt.clf()

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

# pareto plot
plt.xlabel('WT NRMSE')
plt.ylabel('CLF NRMSE')
plt.title('Comparison of Pareto frontiers')
plt.grid()
for k in range(2):
    if k==0:
        WT, CLF = WT_sim.copy(), CLF_sim.copy()
    if k==1:
        WT, CLF = WT_nn.copy(), CLF_nn.copy()
    WT_ind, CLF_ind = np.argmin(WT), np.argmin(CLF)
    WT_min, WT_max, CLF_min, CLF_max = WT[WT_ind], WT[CLF_ind], CLF[CLF_ind], CLF[WT_ind]
    print(WT_min, WT_max, CLF_min, CLF_max)
    ind = np.float64(WT >= WT_min) * np.float64(WT <= WT_max) * np.float64(CLF >= CLF_min) * np.float64(CLF <= CLF_max)
    WT, CLF = WT*ind, CLF*ind
    WT, CLF = WT[WT>0.0], CLF[CLF>0.0]
    ind = np.ones((WT.shape[0],), dtype=np.int32)
    for i in range(WT.shape[0]-1):
        for j in range(i+1, WT.shape[0]):
            if WT[i] < WT[j] and CLF[i] < CLF[j]:
                ind[j] = 0
            if WT[j] < WT[i] and CLF[j] < CLF[i]:
                ind[i] = 0
    WT, CLF = WT*ind, CLF*ind
    WT, CLF = WT[WT>0.0], CLF[CLF>0.0]
    ind = np.argsort(WT)
    WT, CLF = WT[ind], CLF[ind]
    print(WT)
    plt.plot(WT, CLF, '-o')
plt.legend(['Simulation', 'NN'])
plt.savefig('./plot_pareto_front.png')
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

    plt.figure(1)
    plt.xlabel('WT NRMSE')
    plt.ylabel('CLF NRMSE')
    if j==0:
        plt.title('PDE simulation')
    else:
        plt.title('Neural network model')
    #plt.grid()
    axes = plt.gca()
    axes.set_ylim(0.0, 0.2)
    axes.set_xlim(0.0, 0.2)
    plt.scatter(WT, CLF)
    if j==0:
        plt.savefig('./plot_pareto_sim.png')
    else:
        plt.savefig('./plot_pareto_nn.png')
    plt.clf()

    plt.figure(2)
    plt.xlabel('WT NRMSE')
    plt.ylabel('CLF NRMSE')
    plt.title('Comparison of Pareto frontiers')
    WT_ind, CLF_ind = np.argmin(WT), np.argmin(CLF)
    WT_min, WT_max, CLF_min, CLF_max = WT[WT_ind], WT[CLF_ind], CLF[CLF_ind], CLF[WT_ind]
    print(WT_min, WT_max, CLF_min, CLF_max)
    ind = np.float64(WT >= WT_min) * np.float64(WT <= WT_max) * np.float64(CLF >= CLF_min) * np.float64(CLF <= CLF_max)
    WT, CLF = WT*ind, CLF*ind
    WT, CLF = WT[WT>0.0], CLF[CLF>0.0]
    ind = np.ones((WT.shape[0],), dtype=np.int32)
    for i in range(WT.shape[0]-1):
        for j in range(i+1, WT.shape[0]):
            if WT[i] < WT[j] and CLF[i] < CLF[j]:
                ind[j] = 0
            if WT[j] < WT[i] and CLF[j] < CLF[i]:
                ind[i] = 0
    WT, CLF = WT*ind, CLF*ind
    WT, CLF = WT[WT>0.0], CLF[CLF>0.0]
    ind = np.argsort(WT)
    WT, CLF = WT[ind], CLF[ind]
    print(WT)
    plt.plot(WT, CLF, '-o')

plt.figure(2)
plt.grid()
plt.legend(['simulation', 'nn'])
plt.savefig('./plot_pareto_front2.png')
