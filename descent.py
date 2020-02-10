import os
import numpy as np
from scipy.io import loadmat
from time import time
import random
from random import uniform
from random import randint
import torch
from model_simple import ModelSimple
import pandas as pd
from csvdata import CSVdata
from pathlib import Path
import argparse
from solver import read_exp_data
# random.seed(0)
os.environ['MKL_NUM_THREADS'] = '1'


parser = argparse.ArgumentParser(description='Simulation Data Training')
parser.add_argument('--data', default='', type=str, help='path to dataset')
args = parser.parse_args()


if __name__ == '__main__':

    data_dir = Path(args.data)
    val_dataset = CSVdata(data_dir / 'val_data.csv')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=7, shuffle=False, num_workers=1,
                                             pin_memory=True)

    model = ModelSimple().cuda().eval()
    model_path = './model_best.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    best_score = checkpoint['best_score']
    print(best_score)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))

    exp_data = loadmat('pSmad_WT_MT_new.mat')
    WT_exp, WT_ref_exp = read_exp_data(exp_data, 'pWT_57')
    WT_exp = torch.FloatTensor(WT_exp).cuda()
    ref_exp = torch.FloatTensor([WT_ref_exp]).cuda()

    min_error = torch.FloatTensor([1.0]).cuda()
    lr = 0.01
    for i, (inputs, target, target_error, mutation_type) in enumerate(val_loader):
        inputs = inputs.cuda()
        for j in range(2000):
            inputs.requires_grad = True
            outputs = model(inputs)[0]
            outputs = torch.pow(10.0, 10.0 * outputs)
            ref_sim = (torch.sort(outputs).values[-5:]).mean().detach()
            outputs = outputs[0:32:2]
            outputs = outputs * (ref_exp / ref_sim)
            WT_nrmse = torch.sqrt(torch.pow(outputs - WT_exp, 2).mean()) / 61.9087
            min_error = torch.min(min_error, WT_nrmse)
            model.zero_grad()
            WT_nrmse.backward()
            #print(inputs.grad[0])
            grads = inputs.grad.copy()
            grads[-2:] = 0.0
            inputs = inputs - lr * grads
            inputs = inputs.detach()

            if j % 100 == 0:
                lr *= 0.99
                print(j, WT_nrmse)

        if i==10:
            break

    print(min_error.item())
