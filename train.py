# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse
import os
import json
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
#from models.model import CifarResNeXt
from models.model_resnext50 import CifarResNeXt
from dataset_v2 import ImageDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments



    Traindatapath='./DATAlist/Trainlist.csv'
    Testdatapath = './DATAlist/Testlist.csv'

    parser.add_argument('--Test_data_path', type=str, default=Testdatapath)
    parser.add_argument('--Train_data_path', type=str, default=Traindatapath)
    parser.add_argument('--cpu_workers', type=int, default=4)
    #parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
    #parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    #parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=32, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default='./', help='Log folder.')
    args = parser.parse_args()

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    args.epochs = args.epochs * 128 // args.batch_size
    args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    #if not os.path.isdir(args.data_path):
    #    os.makedirs(args.data_path)

    train_dataset = ImageDataset(
        data_dir=args.Train_data_path,
        in_memory=False,
    )
    val_dataset = ImageDataset(
        data_dir=args.Test_data_path,
        in_memory=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.cpu_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // 4,
        num_workers=args.cpu_workers,
        shuffle=False,
    )
    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer

    nlabels=6
    #torch.cuda.set_device(0)
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor)
    print(net)
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            #data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward

            for key in batch:
                batch[key] = batch[key].cuda()
            ##256 -> 512 1024 or use all of them
            data = batch["parcels"]
            output = net(data)
            target=batch["GT_Pattern"]/100-1

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8
            print(batch_idx, loss_avg)


        state['train_loss'] = loss_avg



    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        for batch_idx, batch in enumerate(val_dataloader):
            for key in batch:
                batch[key] = batch[key].cuda()
            ##256 -> 512 1024 or use all of them
            data = batch["parcels"]
            output = net(data)
            target = batch["GT_Pattern"] / 100 - 1

            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # test loss average
            loss_avg += float(loss)

        state['test_loss'] = loss_avg / len(val_dataloader)
        state['test_accuracy'] = correct / len(val_dataloader.dataset)


    # Main loop
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        if epoch in args.schedule:
            state['learning_rate'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

        state['epoch'] = epoch
        train()
        test()
        if state['test_accuracy'] > best_accuracy:
            best_accuracy = state['test_accuracy']
            torch.save(net.state_dict(), os.path.join(args.save, 'model.pytorch'))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best accuracy: %f" % best_accuracy)

    log.close()
