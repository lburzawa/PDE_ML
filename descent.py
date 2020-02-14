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
random.seed(4)
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
    CLF_exp = torch.FloatTensor(parameters.CLF_exp).cuda()
    
    min_error = 10.0
    lr = 0.001
    penalty_coeff = 0.0
    momentum = torch.zeros(1, 22).cuda()
    momentum_coeff = 0.9
    #mask.requires_grad = True
    for i in range(1):
        set_discrete_parameters(parameters)
        inputs = prepare_inputs(parameters).cuda()
        for j in range(1000):
            inputs.requires_grad = True
            outputs = model(inputs)[0]
            outputs = torch.pow(10.0, 10.0 * outputs)
            ref_sim = (torch.sort(outputs).values[-5:]).mean().detach()
            outputs = outputs[0:32:2]
            outputs = outputs * (ref_exp / ref_sim)
            inputs2 = inputs.clone()
            inputs2[0, 18] = -0.8
            inputs2 = inputs2.detach()
            inputs2.requires_grad = True
            #print(inputs.requires_grad)
            outputs2 = model(inputs2)[0]
            outputs2 = torch.pow(10.0, 10.0 * outputs2)
            outputs2 = outputs2[0:32:2]
            outputs2 = outputs2 * (ref_exp / ref_sim)
            WT_nrmse = torch.sqrt(torch.pow(outputs - WT_exp, 2).mean()) / 61.9087
            CLF_nrmse = torch.sqrt(torch.pow(outputs2 - CLF_exp, 2).mean()) / 61.9087
            penalty = min_values - torch.min(inputs, min_values) + torch.max(inputs, max_values) - max_values
            bad_param_count = (penalty[0] > 0.0).sum().item()
            penalty = penalty.mean()
            loss = WT_nrmse + CLF_nrmse #+ penalty_coeff * penalty
            if loss.item() < min_error: # and penalty.item() == 0.0:
                best_inputs = inputs.detach().clone().squeeze()
                min_error = loss.item() #min(min_error, loss.item())
                min_error_penalty = penalty.item()
                WT_min = WT_nrmse.item()
                CLF_min = CLF_nrmse.item()
            model.zero_grad()
            loss.backward()
            #print(inputs.grad[0])
            grads1 = inputs.grad.clone()
            grads2 = inputs2.grad.clone()
            grads2[0, 18] = 0.0
            grads = grads1 + grads2
            grads[:, -3:] = 0.0
            #grads[:, 18] = 0.0
            momentum = momentum_coeff * momentum - lr * grads.detach()
            #print(momentum)
            inputs = inputs + momentum
            #inputs = torch.max(inputs, min_values)
            #inputs = torch.min(inputs, max_values)
            inputs = inputs.detach()
            
            if j % 10 == 0:
                print(j, loss.item(), penalty.item(), bad_param_count)

            if (j+1) % 10 == 0:
                lr *= 0.8
            #if (j+1) == 500:
            #    lr = 0.005
            #if (j + 1) == 1500:
            #    lr = 0.001
            #if (j+1) == 3000:
            #    lr = 0.0005
                

        #print(min_error, min_error_penalty)


    options = ['WT', 'CLF', 'sim', 'nn']
    print()
    parameters = inputs2parameters(best_inputs)
    print(parameters)
    print(min_error, WT_min, CLF_min, min_error_penalty, bad_param_count)
    results = run_simulation(parameters, options, model)
    print(results)
    print(results[1] + results[4])