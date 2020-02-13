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
from solver import Parameters
from solver import prepare_inputs
from solver import set_discrete_parameters
from solver import run_simulation
from solver import inputs2parameters
from math import log10
random.seed(2)
os.environ['MKL_NUM_THREADS'] = '1'

min_values = (torch.FloatTensor([[-2.0, -2.0, -2.0, log10(0.5),  -5.0, -5.0, -5.0, -5.0, -2.0, -4.0, -4.0, 0.0, 0.0,
                                 -4.0, -4.0, -4.0, -4.0, -3.0, -2.0, -5.0, -5.0, -5.0]]).cuda()) / 10.0
max_values = (torch.FloatTensor([[ 2.0,  2.0,  2.0, log10(50.0), -1.0, -1.0, -3.0, -3.0,  2.0,  0.0,  0.0, 2.0, 2.0,
                                  0.0,  0.0,  0.0,  0.0, -1.0,  1.0, 5.0, 5.0, 5.0]]).cuda()) / 10.0

if __name__ == '__main__':

    model = ModelSimple().cuda().eval()
    model_path = './model_best.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    best_score = checkpoint['best_score']
    print(best_score)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))

    parameters = Parameters()
    WT_exp = torch.FloatTensor(parameters.WT_exp).cuda()
    ref_exp = torch.FloatTensor([parameters.WT_ref_exp]).cuda()
    
    min_error = 1.0 
    lr = 0.01
    penalty_coeff = 0.0
    for i in range(1):
        set_discrete_parameters(parameters)
        inputs = prepare_inputs(parameters).cuda()
        for j in range(5000):
            inputs.requires_grad = True
            outputs = model(inputs)[0]
            outputs = torch.pow(10.0, 10.0 * outputs)
            ref_sim = (torch.sort(outputs).values[-5:]).mean().detach()
            outputs = outputs[0:32:2]
            outputs = outputs * (ref_exp / ref_sim)
            WT_nrmse = torch.sqrt(torch.pow(outputs - WT_exp, 2).mean()) / 61.9087
            penalty = min_values - torch.min(inputs, min_values) + torch.max(inputs, max_values) - max_values
            bad_param_count = (penalty[0] > 0.0).sum().item()
            penalty = penalty.mean()
            loss = WT_nrmse + penalty_coeff * penalty
            if WT_nrmse.item() < min_error: # and penalty.item() == 0.0:
                best_inputs = inputs.detach().clone().squeeze()
                min_error = min(min_error, WT_nrmse.item())
                min_error_penalty = penalty.item()
            model.zero_grad()
            loss.backward()
            #print(inputs.grad[0])
            grads = inputs.grad.clone()
            grads[:, -3:] = 0.0
            inputs = inputs - lr * grads
            inputs = torch.max(inputs, min_values)
            inputs = torch.min(inputs, max_values)
            inputs = inputs.detach()
            
            if j % 100 == 0:
                print(j, WT_nrmse.item(), penalty.item(), loss.item(), bad_param_count)

            if (j+1) % 100 == 0:
                lr *= 0.9
            #if (j+1) == 2000:
            #    lr *= 0.1
                

        #print(min_error, min_error_penalty)


    options = ['WT', 'sim', 'nn']
    print()
    parameters = inputs2parameters(best_inputs)
    print(parameters)
    print(min_error, min_error_penalty, bad_param_count)
    print(run_simulation(parameters, options, model))
