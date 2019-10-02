from time import time
import torch
from model_simple import ModelSimple

device = torch.device("cpu")

net = ModelSimple()#.to(device)
start_time = time()
for i in range(1000):
    x = torch.randn(1, 23)#.to(device)
    y = net(x)

print("time is {:f}".format((time()-start_time)/1000.0))
