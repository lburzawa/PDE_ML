import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from csvdata import read_data
from csvdata import CSVdata
from model_lstm import ModelLSTM
from model_simple import ModelSimple

parser = argparse.ArgumentParser(description='Simulation Data Training')
parser.add_argument('--data', default='', type=str, help='path to dataset')
parser.add_argument('--lstm', action='store_true', help='use lstm')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

def main():
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    '''    
    if args.seed is not None:
        
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    '''
    best_r2 = 0.0

    print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.lstm:
        model = ModelLSTM()
    else:
        model = ModelSimple()

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_r2 = checkpoint['best_r2']
            #best_r2 = best_r2.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_input_path = os.path.join(args.data, 'train_input.pth')
    train_output_path = os.path.join(args.data, 'train_output.pth')
    val_input_path = os.path.join(args.data, 'val_input.pth')
    val_output_path = os.path.join(args.data, 'val_output.pth')
    train_dataset = CSVdata(train_input_path, train_output_path)
    val_dataset = CSVdata(val_input_path, val_output_path)
    sstot_train = train_dataset.sstot
    sstot_val = val_dataset.sstot
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, -1, sstot_val, best_r2, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, sstot_train, args)

        # evaluate on validation set
        is_best, best_r2 = validate(val_loader, model, criterion, epoch, sstot_val, best_r2, args)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_r2': best_r2,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, sstot, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ssres_vals = AverageMeter('SSres', ':.4f')
    r2_scores = AverageMeter('R2', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, r2_scores],
        prefix="Train: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(inputs)
        if args.lstm:
            target = target.transpose(0,1)
            loss = 0.0
            ssres = 0.0
            for j in range(len(output)):        
                loss += criterion(output[j], target[j])
                ssres += (target[j] - output[j]).pow(2).sum()
        else:
            target = target.view(target.size(0), -1)
            loss = criterion(output, target)
            ssres = (target - output).pow(2).sum()

        # measure accuracy and record loss
        #ssres = (target - output).pow(2).sum()
        losses.update(loss.item(), inputs.size(0))
        ssres_vals.update(ssres.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i+1)

    r2 = 1.0 - (ssres_vals.sum / sstot)
    r2_scores.update(r2)  

    progress.display(i+1)

def validate(val_loader, model, criterion, epoch, sstot, best_r2, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ssres_vals = AverageMeter('SSres', ':.4f')
    r2_scores = AverageMeter('R2', ':.4f')
    best_r2_obj = AverageMeter('Best R2', ':.4f') # dummy object to print out best R2 in the same format
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, r2_scores, best_r2_obj],
        prefix='Test:  [{}]'.format(epoch))
    if epoch==-1:
        results = torch.zeros(len(val_loader.dataset), 36, 6)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(inputs)
            if args.lstm:
                target = target.transpose(0,1)
                loss = 0.0
                ssres = 0.0
                for j in range(len(output)):        
                    loss += criterion(output[j], target[j])
                    ssres += (target[j] - output[j]).pow(2).sum()
                
            else:
                if epoch==-1:
                    results[i*val_loader.batch_size : (i+1)*val_loader.batch_size] = output.view(output.size(0), 36, 6)
                target = target.view(target.size(0), -1)
                loss = criterion(output, target)
                ssres = (target - output).pow(2).sum()

            # measure accuracy and record loss
            #ssres = (target - output).pow(2).sum()
            losses.update(loss.item(), inputs.size(0))
            ssres_vals.update(ssres.item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                progress.display(i+1)
                #print(output[20])
                #print(target[20])

        #print(ssres_vals.avg)
        #print(sstot)
        r2 = 1.0 - (ssres_vals.sum / sstot)
        is_best = r2 > best_r2
        best_r2 = max(r2, best_r2)
        r2_scores.update(r2)
        best_r2_obj.update(best_r2)

        progress.display(i+1)
        if epoch==-1:
            torch.save(results, './results.pth')

        #print(output[12,100:110])
        #print(target[12,100:110])
        #print(10**(10.0*output[12,100:110]))
        #print(10**(10.0*target[12,100:110]))

    return is_best, best_r2


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        #print(self.__dict__)
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
