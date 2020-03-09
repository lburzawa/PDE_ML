# PDE_ML
Exploring the use of machine learning in PDE simulations.

## Requirements

- Python3 `sudo apt install python3`
- Numpy `pip install numpy`
- Pandas `pip install pandas`
- PyTorch ([pytorch.org](http://pytorch.org))

## Project structure

`csvdata.py` functions for loading and processing data into format suitable for training and/or validation

`descent.py` gradient descent for PDE parameter search done on the neural network model - not working well so far

`model_lstm.py` LSTM model used for PDE acceleration training

`model_rl.py` LSTM model used for PDE parameter search with reinforcement learning

`model_simple.py` MLP model used for PDE acceleration training

`ode_fun.py` set of ODE equations used to solve the PDE system

`pSmad_WT_MT_new.mat` experimental data

`plotter.py` plotting tools

`prepdata.py` data preprocessing script, pulls data from many MAT files and converts it into training and validation CSV files

`profile.py` script for profiling NNs used for PDE acceleration

`run_rl.py` PDE parameter search through reinforcement learning

`solver.py` script to run PDE simulations

`train.py` training script for PDE acceleration models


## Training

To train a model, run `train.py` with the path to the data:

```bash
python train.py --data=/path/to/data
```

## Usage

```
usage: train.py [-h] [--data DATA] [--lstm] [--use_k] [-j WORKERS]
                [--epochs EPOCHS] [--start-epoch START_EPOCH] [-b BATCH_SIZE]
                [--lr LR] [--momentum MOMENTUM] [--wd WEIGHT_DECAY]
                [-p PRINT_FREQ] [--resume RESUME] [-e] [--seed SEED]
                [--gpu GPU]

Simulation Data Training

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to dataset
  --lstm                use lstm
  -j WORKERS, --workers WORKERS
                        number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run
  --start-epoch START_EPOCH
                        manual epoch number (useful on restarts)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum MOMENTUM   momentum
  --wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay (default: 1e-4)
  -p PRINT_FREQ, --print-freq PRINT_FREQ
                        print frequency (default: 10000)
  --resume RESUME       path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.

```
