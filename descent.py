import os
import numpy as np
from scipy.io import loadmat
from time import time
import random
from random import uniform
from random import randint
import torch
import torch.nn as nn
from model_simple import ModelSimple
import pandas as pd
from csvdata import CSVdata
from pathlib import Path
import argparse
from solver import Parameters
from solver import parameters2inputs
from solver import set_discrete_parameters
from solver import run_simulation
from solver import inputs2parameters
from math import log10
from math import cos
from math import pi
random.seed(0)
os.environ['MKL_NUM_THREADS'] = '1'

min_values = (torch.FloatTensor([[-2.0, -2.0, -2.0, log10(0.5),  -5.0, -5.0, -5.0, -5.0, -2.0, -4.0, -4.0, 0.0, 0.0,
                                 -4.0, -4.0, -4.0, -4.0, -3.0, -2.0, -5.0, -5.0, -5.0, -5.0]]).cuda()) / 10.0
max_values = (torch.FloatTensor([[ 2.0,  2.0,  2.0, log10(50.0), -1.0, -1.0, -3.0, -3.0,  2.0,  0.0,  0.0, 2.0, 2.0,
                                  0.0,  0.0,  0.0,  0.0, -1.0,  1.0, 5.0, 5.0, 5.0, 5.0]]).cuda()) / 10.0

if __name__ == '__main__':

    model = ModelSimple(23).cuda().eval()
    model_path = './models/mlp.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    best_score = checkpoint['best_score']
    print(best_score)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))

    parameters = Parameters()
    WT_exp = torch.FloatTensor(parameters.WT_exp).cuda()
    ref_exp = torch.FloatTensor([parameters.WT_ref_exp]).cuda()
    CLF_exp = torch.FloatTensor(parameters.CLF_exp).cuda()
    
    min_error = 10.0
    lr = 0.001
    penalty_coeff = 0.0
    #mask.requires_grad = True
    max_steps = 2000
    for i in range(1):
        set_discrete_parameters(parameters)
        parameters.k_nn = 1.0
        all_inputs = parameters2inputs(parameters).cuda()
        inputs = nn.Parameter(all_inputs[:, :-4])
        for j in range(max_steps):
            outputs = model(inputs)[0]
            outputs = torch.pow(10.0, 10.0 * outputs)
            inputs[0, 22] = torch.log10(outputs.max() * 4.1) / 10.0
            inputs = inputs.detach()
            inputs.requires_grad = True
            outputs = model(inputs)[0]
            outputs = torch.pow(10.0, 10.0 * outputs)
            ref_sim = (torch.sort(outputs).values[-5:]).mean().detach()
            outputs = outputs[0:32:2]
            outputs = outputs * (ref_exp / ref_sim)
            '''
            inputs2 = inputs.clone()
            inputs2[0, 18] = -0.8
            inputs2 = inputs2.detach()
            inputs2.requires_grad = True
            #print(inputs.requires_grad)
            outputs2 = model(inputs2)[0]
            outputs2 = torch.pow(10.0, 10.0 * outputs2)
            outputs2 = outputs2[0:32:2]
            outputs2 = outputs2 * (ref_exp / ref_sim)
            '''
            WT_nrmse = torch.sqrt(torch.pow(outputs - WT_exp, 2).mean()) / 61.9087
            #CLF_nrmse = torch.sqrt(torch.pow(outputs2 - CLF_exp, 2).mean()) / 61.9087
            penalty = min_values - torch.min(inputs, min_values) + torch.max(inputs, max_values) - max_values
            bad_param_count = (penalty[0] > 0.0).sum().item()
            penalty = penalty.sum()
            loss = WT_nrmse #+ CLF_nrmse  # + penalty_coeff * penalty
            if loss.item() < min_error: # and penalty.item() == 0.0:
                best_inputs = inputs.detach().clone().squeeze()
                min_error = loss.item() #min(min_error, loss.item())
                min_error_penalty = penalty.item()
                WT_min = WT_nrmse.item()
                #CLF_min = CLF_nrmse.item()
            model.zero_grad()
            loss.backward()
            #print(inputs.grad[0])
            grads = inputs.grad.clone()
            #grads2 = inputs2.grad.clone()
            #grads2[0, 18] = 0.0
            #grads = grads1 + grads2
            grads[:, -4:] = 0.0
            inputs = inputs - lr * grads.detach()
            #print(new_lr)
            #inputs = torch.max(inputs, min_values)
            #inputs = torch.min(inputs, max_values)
            #inputs = inputs.detach()
            #print(loss)
            
            if j % 100 == 0:
                print(j, loss.item(), penalty.item(), bad_param_count)

            if (j+1) % 100 == 0:
                lr *= 0.9
            #if (j+1) == 1000:
            #    lr *= 0.5
            #if (j + 1) == 1500:
            #    lr = 0.001
            #if (j+1) == 3000:
            #    lr = 0.0005
                

        #print(min_error, min_error_penalty)


    options = ['WT', 'sim', 'nn']
    print()
    parameters = inputs2parameters(best_inputs)
    print(parameters)
    print(min_error, WT_min, min_error_penalty, bad_param_count)
    results = run_simulation(parameters, options, model)
    print(results)
    #print(results[1] + results[4])
    #print(results[0] + results[3])