import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import math
from argparse import ArgumentParser
import os
import random
from random import sample, randint, uniform
from time import time
from tqdm import trange
from model_rl import Net
from solver import Parameters
from solver import run_simulation
from math import exp


parser = ArgumentParser()
_ = parser.add_argument
_('--save_dir', type = str, default = './save_rl', help = 'save directory')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.backends.cudnn.benchmark=True
#torch.backends.cudnn.deteministic=True
random.seed(0)
torch.manual_seed(0)

min_values = np.float64([-4.0, -4.0, -3.0, -4.0, 0.0, -4.0, 0.0, 0.0, -2.0, -4.0, \
              -4.0, -2.0, -5.0, -2.0, -2.0, -5.0, -2.0, -5.0, -5.0])
max_values = np.float64([0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, \
              0.0, 2.0, -1.0, 2.0, 2.0, -3.0, 2.0, -3.0, -1.0])

batch_size = 1
learning_rate = 0.0002
epochs = 100
training_steps_per_epoch = 100
testing_steps_per_epoch = 1
seq_len = 19
reward_scaling = 1.0
entropy_loss_scaling = 0.00001
max_grad = 40.0
load_model = False
model_dir = args.save_dir
model_loadfile = "./save/model.pth"
model_savefile = os.path.join(model_dir, "model.pth")


def set_var_parameters(parameters, outputs):
    parameters.k1 = 10 ** outputs[0].item()
    parameters.k_1 = parameters.k1
    parameters.k2 = 10 ** outputs[1].item()
    parameters.k_2 = 0.1 * parameters.k2
    parameters.j1 = 10 ** outputs[2].item()
    parameters.lambda_Tld_BMPChd = 10 ** outputs[3].item()
    parameters.kmt = 10 ** outputs[4].item()
    parameters.lambda_bmp1a_BMPChd = 10 ** outputs[5].item()
    parameters.kma = 10 ** outputs[6].item()
    parameters.D_Chd = 0.5 * (10 ** outputs[7].item())
    parameters.j2 = 10 ** outputs[8].item()
    parameters.lambda_Tld_Chd = 10 ** outputs[9].item()
    parameters.lambda_bmp1a_Chd = 10 ** outputs[10].item()
    parameters.D_Nog = (10 ** outputs[11].item())
    parameters.dec_Nog = 10 ** outputs[12].item()
    parameters.j3 = 10 ** outputs[13].item()
    parameters.D_BMPChd = (10 ** outputs[14].item())
    parameters.dec_BMPChd = 10 ** outputs[15].item()
    parameters.D_BMPNog = (10 ** outputs[16].item())
    parameters.dec_BMPNog = 10 ** outputs[17].item()
    parameters.dec_Szd = 10 ** outputs[18].item()


if __name__ == '__main__':

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    of = open(os.path.join(model_dir, 'test.txt'), 'w')

    if load_model:
        print("Loading model from: ", model_loadfile)
        model = Net(seq_len)
        my_sd = torch.load(model_loadfile)
        model.load_state_dict(my_sd)
    else:
        model = Net(seq_len)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    parameters = Parameters()

    print("Starting the training!")
    start_time = time()
    forward_time = 0.0
    backward_time = 0.0
    test_time = 0.0
    baseline_sum = 0.0
    baseline_count = 0
    baseline = 0.0
    whole_batch = torch.arange(batch_size)
    solver_options = ['WT', 'CLF', 'NLF', 'ALF', 'TLF', 'TALF', 'SLF', 'sim']

    for epoch in range(epochs):
        print("\nEpoch %d\n-------" % (epoch+1))
        error_total = 0.0
        error_min = 1.0
        loss_policy_total = 0.0
        loss_entropy_total = 0.0
        error_list = []
        print("Training...")
        model.train()
        for learning_step in trange(training_steps_per_epoch, leave=False):
            forward_start_time = time()
            outputs = torch.zeros(batch_size, seq_len).cuda()
            log_probs_list = []
            entropy_list = []
            hidden = model.init_hidden(batch_size)
            for t in range(seq_len):
                inp = torch.tensor([[min_values[t], max_values[t]]] * batch_size).cuda() / 5.0
                policy, hidden = model(inp, hidden)
                probs = F.softmax(policy, 1)
                log_probs = F.log_softmax(policy, 1)
                a = probs.multinomial(num_samples=1).detach().squeeze(1)
                outputs[:, t] = (max_values[t] - min_values[t]) * (a/31.0) + min_values[t]
                log_probs_list.append(log_probs[whole_batch, a])
                entropy_list.append(-(probs * log_probs).sum(1))
            set_var_parameters(parameters, outputs[0])
            errors = run_simulation(parameters, solver_options)  
            error = sum(errors)
            #print(outputs)
            #print(errors)
            error_total += error
            error_min = min(error_min, error)
            log_probs_sum = 0.0
            entropy_sum = 0.0
            for i in range(len(log_probs_list)):
                log_probs_sum += log_probs_list[i]
                entropy_sum += entropy_list[i]
            reward = torch.exp(-torch.FloatTensor([error])).cuda()
            forward_time += (time() - forward_start_time)

            backward_start_time = time()
            adv = reward - baseline
            loss_policy = (-log_probs_sum * adv).mean()
            loss_entropy = (-entropy_sum).mean()
            loss = loss_policy + entropy_loss_scaling * loss_entropy
            loss_policy_total += loss_policy.item()
            loss_entropy_total += loss_entropy.item()
            baseline_sum += reward.sum()
            baseline_count += reward.size(0)
            baseline = baseline_sum / baseline_count
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()
            #baseline = 0.95 * baseline + 0.05 * reward.mean()
            backward_time += (time() - backward_start_time)

        total_steps = batch_size * (epoch + 1) * training_steps_per_epoch
        print(baseline)
        print("Results: mean error: {:f} min: {:f}".format(error_total / training_steps_per_epoch, error_min))
        print('Loss_policy: {:f}, loss_entropy: {:f}'.format(loss_policy_total/(training_steps_per_epoch*seq_len), loss_entropy_total/(training_steps_per_epoch*seq_len)))

        #torch.save(model.state_dict(), model_savefile)
        total_time = time() - start_time
        print("Total training steps: {:d}, Total elapsed time: {:.2f} minutes, Time per step: {:.2f} min".format(total_steps, total_time / 60.0, (total_time / ((epoch+1)*60.0))))
        print("Forward time: {:.2f} min, Backward time: {:.2f} min".format((forward_time / ((epoch+1)*60.0)), (backward_time / ((epoch+1)*60.0))))
        of.write('{:d},{:f},{:f}\n'.format(total_steps, total_time / 60.0, error_total / training_steps_per_epoch))
        of.flush()

    print("======================================")
    print("Training finished.")


