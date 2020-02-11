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
random.seed(2)
os.environ['MKL_NUM_THREADS'] = '1'


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
            if WT_nrmse.item() < min_error:
                best_inputs = inputs.detach().clone().squeeze()
            min_error = min(min_error, WT_nrmse.item())
            model.zero_grad()
            WT_nrmse.backward()
            #print(inputs.grad[0])
            grads = inputs.grad.clone()
            grads[:, -3:] = 0.0
            inputs = inputs - lr * grads
            inputs = inputs.detach()
            
            if j % 100 == 0:
                if j>0:
                    lr *= 0.9
                print(j, WT_nrmse.item())            

        print(min_error)
            
    print(min_error)

    options = ['WT']
    parameters = inputs2parameters(best_inputs)
    print(parameters)
    print(run_simulation(parameters, model, options))
