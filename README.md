# PDE_ML
Exploring the use of machine learning in PDE simulations.

## Requirements

- Python3 `sudo apt install python3`
- Numpy `pip install numpy`
- Pandas `pip install pandas`
- PyTorch ([pytorch.org](http://pytorch.org))

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
