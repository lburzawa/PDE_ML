import os
import numpy as np
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from time import time
from ode_fun import set_ode_fun
import random
from random import uniform
from random import randint
import torch
from multiprocessing import Pool
from model_simple import ModelSimple


min_values = np.float64([-4.0, -4.0, -3.0, -4.0, 0.0, -4.0, 0.0, 0.0, -2.0, -4.0, \
              -4.0, -2.0, -5.0, -2.0, -2.0, -5.0, -2.0, -5.0, -5.0])
max_values = np.float64([0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, \
              0.0, 2.0, -1.0, 2.0, 2.0, -3.0, 2.0, -3.0, -1.0])


class Parameters:
    def __init__(self):
        # para_grid, para_grid_jBC, para_grid_k, para_grid_ki = read_parameters(i)
        exp_data = loadmat('pSmad_WT_MT_new.mat')
        self.WT_exp, self.WT_ref_exp = read_exp_data(exp_data, 'pWT_57')
        self.CLF_exp, self.CLF_ref_exp = read_exp_data(exp_data, 'pCLF_57')
        self.ALF_exp, self.ALF_ref_exp = read_exp_data(exp_data, 'pALF_57')
        self.TLF_exp, self.TLF_ref_exp = read_exp_data(exp_data, 'pTLF_57')
        self.num_proteins = 6
        self.n = 36  # number of nodes to evaluate finite difference equations
        self.T = 7900.0  # how many time steps to use in PDE solution
        self.Ltot = 700.0  # length of embryo
        self.Lven_BMP = 400.0  # length of ventral region for BMP
        self.Lven_Tld = 700.0  # length of ventral region for Tolloid
        self.Ldor_Chd = 140.0  # length of dorsal region for Chordin
        self.Ldor_Nog = 78.0  # length of dorsal region for Noggin
        dx = self.Ltot / (self.n - 1)
        self.dx2 = dx * dx
        self.D_BMP = 4.4 / self.dx2  # diffusion rate of BMP
        self.D_Szd = 10.0 / self.dx2  # diffusion rate of Sizzled
        self.dec_BMP = 8.9e-5  # decay rate of BMP
        self.dec_Chd = 9.6e-5  # decay rate of Chordin
        self.nu = 4.0  # cooperative parameter
        self.Vs = 100.0
        self.kit = 510.7204  # para_grid_ki[0]  # inhibitor constant of proteinase Tolloid
        self.kia = 550.5586  # para_grid_ki[1]  # inhibitor constant of proteinase bmp1a
        self.k = 25.8752 ** self.nu  # parameter for hill function
        self.ndor_Chd = round(self.Ldor_Chd * self.n / self.Ltot)
        self.ndor_Nog = round(self.Ldor_Nog * self.n / self.Ltot)
        x_lig = np.arange(0.0, self.Ltot + dx / 2, dx)
        self.yspace = np.exp(-((x_lig + dx) / 20 - 11) / 5) / (0.1 + np.exp(-((x_lig + dx) / 20 - 11) / 5))
        self.init = np.zeros((self.num_proteins * self.n,), dtype=np.float64)
        self.tspan = (0.0, self.T)


'''
def read_parameters(i):
    para_grid = loadmat('para_screened.mat')['para_grid'][i]
    para_grid_jBC = loadmat('para_grid_jBC_screened.mat')['para_grid_jBC'][i]
    para_grid_k = loadmat('para_grid_k_screened.mat')['k'][0, i]
    para_grid_ki = loadmat('para_grid_ki_screened.mat')['para_grid_ki'][i]
    return para_grid, para_grid_jBC, para_grid_k, para_grid_ki
'''


def read_exp_data(exp_data, var_name):
    exp_data = exp_data[var_name][0]
    ref = (np.sort(exp_data)[-5:]).mean()
    exp_data = exp_data[:16]
    return exp_data, ref

def crandint(min_val, max_val):
    return (max_val - min_val) * (randint(0, 15)/15.0) + min_val

def set_continuous_parameters(parameters):
    parameters.D_Nog = (10 ** uniform(-2.0, 2.0)) / parameters.dx2
    parameters.D_BMPChd = (10 ** uniform(-2.0, 2.0)) / parameters.dx2
    parameters.D_BMPNog = (10 ** uniform(-2.0, 2.0)) / parameters.dx2
    parameters.D_Chd = 0.5 * (10 ** uniform(0.0, 2.0)) / parameters.dx2
    parameters.dec_Nog = 10 ** uniform(-5.0, -1.0)
    parameters.dec_Szd = 10 ** uniform(-5.0, -1.0)
    parameters.dec_BMPChd = 10 ** uniform(-5.0, -3.0)
    parameters.dec_BMPNog = 10 ** uniform(-5.0, -3.0)
    parameters.j3 = 10 ** uniform(-2.0, 2.0)
    parameters.k1 = 10 ** uniform(-4.0, 0.0)
    parameters.k_1 = parameters.k1
    parameters.k2 = 10 ** uniform(-4.0, 0.0)
    parameters.k_2 = 0.1 * parameters.k2
    parameters.kmt = 10 ** uniform(0.0, 2.0)
    parameters.kma = 10 ** uniform(0.0, 2.0)
    parameters.lambda_Tld_Chd = 10 ** uniform(-4.0, 0.0)
    parameters.lambda_Tld_BMPChd = 10 ** uniform(-4.0, 0.0)
    parameters.lambda_bmp1a_Chd = 10 ** uniform(-4.0, 0.0)
    parameters.lambda_bmp1a_BMPChd = 10 ** uniform(-4.0, 0.0)
    while True:
        parameters.j1 = 10 ** uniform(-3.0, -1.0)
        parameters.j2 = 10 ** uniform(-2.0, 1.0)
        if parameters.j2 > parameters.j1:
            break


def set_discrete_parameters(parameters):
    parameters.D_Nog = (10 ** crandint(-2, 2)) / parameters.dx2
    parameters.D_BMPChd = (10 ** crandint(-2, 2)) / parameters.dx2
    parameters.D_BMPNog = (10 ** crandint(-2, 2)) / parameters.dx2
    parameters.D_Chd = 0.5 * (10 ** crandint(0, 2)) / parameters.dx2
    parameters.dec_Nog = 10 ** crandint(-5, -1)
    parameters.dec_Szd = 10 ** crandint(-5, -1)
    parameters.dec_BMPChd = 10 ** crandint(-5, -3)
    parameters.dec_BMPNog = 10 ** crandint(-5, -3)
    parameters.j3 = 10 ** crandint(-2, 2)
    parameters.k1 = 10 ** crandint(-4, 0)
    parameters.k_1 = parameters.k1
    parameters.k2 = 10 ** crandint(-4, 0)
    parameters.k_2 = 0.1 * parameters.k2
    parameters.kmt = 10 ** crandint(0, 2)
    parameters.kma = 10 ** crandint(0, 2)
    parameters.lambda_Tld_Chd = 10 ** crandint(-4, 0)
    parameters.lambda_Tld_BMPChd = 10 ** crandint(-4, 0)
    parameters.lambda_bmp1a_Chd = 10 ** crandint(-4, 0)
    parameters.lambda_bmp1a_BMPChd = 10 ** crandint(-4, 0)
    while True:
        parameters.j1 = 10 ** crandint(-3, -1)
        parameters.j2 = 10 ** crandint(-2, 1)
        if parameters.j2 > parameters.j1:
            break

def prepare_inputs(parameters):
    inputs = torch.zeros(23)
    inputs[0] = parameters.D_Nog
    inputs[1] = parameters.D_BMPChd
    inputs[2] = parameters.D_BMPNog
    inputs[3] = parameters.D_Chd
    inputs[4] = parameters.dec_Nog
    inputs[5] = parameters.dec_Szd
    inputs[6] = parameters.dec_BMPChd
    inputs[7] = parameters.dec_BMPNog
    inputs[8] = parameters.j3
    inputs[9] = parameters.k1
    inputs[10] = parameters.k2
    inputs[11] = parameters.kmt
    inputs[12] = parameters.kma
    inputs[13] = parameters.lambda_Tld_Chd
    inputs[14] = parameters.lambda_Tld_BMPChd
    inputs[15] = parameters.lambda_bmp1a_Chd
    inputs[16] = parameters.lambda_bmp1a_BMPChd
    inputs[17] = parameters.j1
    inputs[18] = parameters.j2
    inputs[19] = parameters.k
    inputs[20] = parameters.kit
    inputs[21] = parameters.kia
    inputs[22] = parameters.Vs
    inputs = inputs.unsqueeze(0)
    return inputs


def solve_pde(parameters, ref_exp):
    fun = set_ode_fun(parameters)
    sol = solve_ivp(fun, parameters.tspan, parameters.init, method='BDF', rtol=1e-9)
    BMP = sol.y[:36, -1]
    ref = (np.sort(BMP)[-5:]).mean()
    BMP *= ref_exp / ref
    BMP = BMP[0:32:2]
    return BMP


def run_simulation(parameters):
    # WT simulation
    WT_sim = solve_pde(parameters, parameters.WT_ref_exp)
    WT_nrmse = np.sqrt(np.power(WT_sim - parameters.WT_exp, 2).mean()) / 61.9087
    # CLF simulation
    j2 = parameters.j2
    parameters.j2 = 0.0
    CLF_sim = solve_pde(parameters, parameters.CLF_ref_exp)
    CLF_nrmse = np.sqrt(np.power(CLF_sim - parameters.CLF_exp, 2).mean()) / 61.9087
    parameters.j2 = j2
    # NLF simulation
    j3 = parameters.j3
    parameters.j3 = 0.0
    NLF_sim = solve_pde(parameters, parameters.WT_ref_exp)
    NLF_nrmse = np.sqrt(np.power(NLF_sim - parameters.WT_exp, 2).mean()) / 61.9087
    parameters.j3 = j3
    # ALF simulation
    lambda_bmp1a_Chd = parameters.lambda_bmp1a_Chd
    lambda_bmp1a_BMPChd = parameters.lambda_bmp1a_BMPChd
    parameters.lambda_bmp1a_Chd = 0.0
    parameters.lambda_bmp1a_BMPChd = 0.0
    ALF_sim = solve_pde(parameters, parameters.ALF_ref_exp)
    ALF_nrmse = np.sqrt(np.power(ALF_sim - parameters.ALF_exp, 2).mean()) / 61.9087
    parameters.lambda_bmp1a_Chd = lambda_bmp1a_Chd
    parameters.lambda_bmp1a_BMPChd = lambda_bmp1a_BMPChd
    # TLF simulation
    lambda_Tld_Chd = parameters.lambda_Tld_Chd
    lambda_Tld_BMPChd = parameters.lambda_Tld_BMPChd
    parameters.lambda_Tld_Chd = 0.0
    parameters.lambda_Tld_BMPChd = 0.0
    TLF_sim = solve_pde(parameters, parameters.TLF_ref_exp)
    TLF_nrmse = np.sqrt(np.power(TLF_sim - parameters.TLF_exp, 2).mean()) / 61.9087
    parameters.lambda_Tld_Chd = lambda_Tld_Chd
    parameters.lambda_Tld_BMPChd = lambda_Tld_BMPChd
    return [WT_nrmse, CLF_nrmse, NLF_nrmse, ALF_nrmse, TLF_nrmse]


if __name__ == '__main__':
    random.seed(0)
    os.environ['MKL_NUM_THREADS'] = '1'

    model = ModelSimple()
    model_path = './model_best.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    best_r2 = checkpoint['best_r2']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
    inputs = prepare_inputs(parameters)

    parameters_list = []
    min_val = 1.0
    for i in range(1000):
        parameters = Parameters()
        set_discrete_parameters(parameters)
        parameters_list.append(parameters)
    start_time = time()
    for i in range(1000):
        results = run_simulation(parameters_list[i])
        min_val = min(min_val, sum(results))
        print(i, results, sum(results), min_val)
    print(time()-start_time)
