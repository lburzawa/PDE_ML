import argparse
from time import time
import torch
from model_simple import ModelSimple
from model_lstm import ModelLSTM

parser = argparse.ArgumentParser(description='Profiler')
parser.add_argument('--lstm', action='store_true', help='profile LSTM')
parser.add_argument('--cuda', action='store_true', help='run on GPU')

if __name__ == '__main__':

    args = parser.parse_args()

    num_inputs = 23

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.lstm:
        model = ModelLSTM()
    else:
        model = ModelSimple(num_inputs)
    model = model.to(device)

    start_time = time()
    for i in range(1000):
        x = torch.randn(1, num_inputs).to(device)
        y = model(x)

    print("time is {:f}".format((time()-start_time)/1000.0))
