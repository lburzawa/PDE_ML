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
from csvdata import CSVdata
from model_lstm import ModelLSTM
from model_simple import ModelSimple
from solver import run_simulation
from solver import inputs2parameters
from pathlib import Path
from scipy.io import loadmat


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

    best_score = 1000.0

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
            best_r2 = checkpoint['best_score']
            #best_r2 = best_r2.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = Path(args.data)
    train_dataset = CSVdata(data_dir / 'train_data.csv')
    val_dataset = CSVdata(data_dir / 'val_data.csv')
    sstot_train = train_dataset.sstot
    sstot_val = val_dataset.sstot
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=280, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, -1, sstot_val, best_score, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, sstot_train, args)

        # evaluate on validation set
        is_best, best_score = validate(val_loader, model, criterion, epoch, sstot_val, best_score, args)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': best_score,
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
    for i, (inputs, target, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(inputs)
        if args.lstm:
            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target[:, j].unsqueeze(1))
            output = torch.stack(output)
            output = output.squeeze().transpose(0, 1)
        else:
            #target = target.view(target.size(0), -1)
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


def read_exp_data(exp_data, var_name):
    exp_data = exp_data[var_name][0]
    ref = (np.sort(exp_data)[-5:]).mean()
    exp_data = exp_data[:16]
    return exp_data, ref


def calculate_error(exp_data, nn_data, ref_exp, target_error, mutation_type):
    num_mutations = 7
    sim_error = 0.0
    for i in range(nn_data.size(0)):
        target = target_error[i].item()
        nn_output = nn_data[i].detach().cpu().numpy()
        nn_output = np.power(10.0, (10.0 * nn_output))
        #nn_output = 1000.0 * nn_output
        if i % num_mutations == 0:
            ref_sim = (np.sort(nn_output)[-5:]).mean()
        nn_output = nn_output[0:32:2]
        nn_output *= ref_exp / ref_sim
        error = np.sqrt(np.power(nn_output - exp_data[mutation_type[i]], 2).mean()) / 61.9087
        sim_error += 100.0 * abs(error - target) / target
    sim_error /= nn_data.size(0)
    return sim_error


def validate(val_loader, model, criterion, epoch, sstot, best_score, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ssres_vals = AverageMeter('SSres', ':.4f')
    r2_scores = AverageMeter('R2', ':.4f')
    sim_errors = AverageMeter('Simulation error', ':.4f')
    best_score_obj = AverageMeter('Best score', ':.4f') # dummy object to print out best R2 in the same format
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, r2_scores, sim_errors, best_score_obj],
        prefix='Test:  [{}]'.format(epoch))
    #if epoch==-1:
    #    results = torch.zeros(len(val_loader.dataset), 36, 6)

    exp_data = loadmat('pSmad_WT_MT_new.mat')
    exp_vars = {}
    exp_vars['WT'], ref_exp = read_exp_data(exp_data, 'pWT_57')
    exp_vars['CLF'], _ = read_exp_data(exp_data, 'pCLF_57')
    exp_vars['NLF'] = exp_vars['WT']
    exp_vars['ALF'], _ = read_exp_data(exp_data, 'pALF_57')
    exp_vars['TLF'], _ = read_exp_data(exp_data, 'pTLF_57')
    exp_vars['TALF'], _ = read_exp_data(exp_data, 'pTALF_57')
    exp_vars['SLF'], _ = read_exp_data(exp_data, 'pSLF_57')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (inputs, target, target_error, mutation_type) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(inputs)
            if args.lstm:
                loss = 0.0
                for j in range(len(output)):
                    loss += criterion(output[j], target[:, j].unsqueeze(1))
                output = torch.stack(output)
                output = output.squeeze().transpose(0, 1)
                
            else:
                #if epoch==-1:
                #    results[i*val_loader.batch_size : (i+1)*val_loader.batch_size] = output.view(output.size(0), 36, 6)
                loss = criterion(output, target)
            ssres = (target - output).pow(2).sum()

            # measure accuracy and record loss
            #ssres = (target - output).pow(2).sum()
            losses.update(loss.item(), inputs.size(0))
            ssres_vals.update(ssres.item(), 1)
            sim_error = calculate_error(exp_vars, output, ref_exp, target_error, mutation_type)
            sim_errors.update(sim_error, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                progress.display(i+1)
                #output = output.view(output.size(0), 36, 6)
                #target = target.view(target.size(0), 36, 6)
                print(10 ** (10.0 * inputs[16]))
                #print(10 ** (10.0 * target[16, :, 1]))

                print(10 ** (10.0 * target[7]))
                parameters = inputs2parameters(inputs[7])
                results = run_simulation(parameters, model)
                #print(results)
                #print('---')

        #print(ssres_vals.avg)
        #print(sstot)
        r2 = 1.0 - (ssres_vals.sum / sstot)
        r2_scores.update(r2)
        is_best = sim_errors.avg < best_score
        best_score = min(sim_errors.avg, best_score)
        best_score_obj.update(best_score)

        progress.display(i+1)
        #if epoch==-1:
        #    torch.save(results, './results.pth')

        #print(output[12,100:110])
        #print(target[12,100:110])
        #output = output.view(output.size(0), 36, 6)
        #target = target.view(target.size(0), 36, 6)
        #print(10 ** (10.0*inputs[16]))
        #parameters = inputs2parameters(inputs[16])
        #results = run_simulation(parameters, model)
        #print(10 ** (10.0 * target[16, :, 1]))
        #print(10**(10.0*output[16,:,1]))

    return is_best, best_score


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
