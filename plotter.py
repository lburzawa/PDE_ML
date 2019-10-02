import argparse
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser(description='Simulation Data Training')
parser.add_argument('--target_data', default='', type=str, help='path to dataset')
args = parser.parse_args()

x_values = np.arange(36)
targets = torch.load(args.target_data)
outputs = torch.load('./results.pth')


for i in range(6):
    plt.xlabel('X value')
    plt.ylabel('Concentration (log10 scale)')
    plt.grid()
    target = 10.0 * targets[i, :, i].numpy()
    output = 10.0 * outputs[i, :, i].numpy()    
    plt.plot(x_values, target)
    plt.plot(x_values, output)
    plt.legend(['Simulation', 'NN'])
    plt.savefig('./plot{:d}.png'.format(i))
    plt.clf()
    
