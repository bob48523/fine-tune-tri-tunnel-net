#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle

import tri_tunnel_net

def loadData():
    #load dataset
    means = {
       'cifar': [0.49139968, 0.48215827, 0.44653124], 
       'imagenet': [0.485, 0.456, 0.406],
       'oxford102flowers': [0.434, 0.385, 0.296],
    }
    stds = {
       'cifar': [0.24703233, 0.24348505, 0.26158768], 
       'imagenet': [0.229, 0.224, 0.225], 
       'oxford102flowers': [0.285, 0.238, 0.262],
    }
    
    val_transforms = {
       x: transforms.Compose([
          transforms.Scale(72),
          transforms.CenterCrop(64),
          transforms.ToTensor(),
          transforms.Normalize(means[x], stds[x])
       ])
       for x in ['cifar', 'imagenet', 'oxford102flowers']
    }

    train_transforms = {
       x: transforms.Compose([
          transforms.Scale(72),
          transforms.RandomSizedCrop(64),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(means[x], stds[x])
       ])
       for x in ['cifar', 'imagenet', 'oxford102flowers']
    }

    data_dir = {
       'cifar': '/dataset/cifar/',
       'imagenet': '/dataset/TINYIMAGENET_1000_80/',
       'oxford102flowers': '/dataset/102flower/102flowers_20_10/',
    }
    
    batch_size = {
       'cifar': 64,
       'imagenet': 64,
       'oxford102flowers':64
    }
    trainset = {
       'cifar': dset.CIFAR100(root=data_dir['cifar'], train=True, download=True, transform = train_transforms['cifar']),
       'imagenet': dset.ImageFolder(os.path.join(data_dir['imagenet'], 'train'), train_transforms['imagenet']),
       'oxford102flowers': dset.ImageFolder(os.path.join(data_dir['oxford102flowers'], 'train'), train_transforms['oxford102flowers']),
    }

    valset = {
       'cifar': dset.CIFAR100(root=data_dir['cifar'], train=False, download=True, transform = val_transforms['cifar']),
       'imagenet': dset.ImageFolder(os.path.join(data_dir['imagenet'], 'val'), val_transforms['imagenet']),
       'oxford102flowers': dset.ImageFolder(os.path.join(data_dir['oxford102flowers'], 'val'), val_transforms['oxford102flowers']),
    }

    trainloader = {
       x: torch.utils.data.DataLoader(trainset[x], batch_size[x],
                                   shuffle=True, num_workers=4)
       for x in ['cifar', 'imagenet', 'oxford102flowers']
    }

    valloader = {
       x: torch.utils.data.DataLoader(valset[x], batch_size[x],
                                   shuffle=False, num_workers=4)
       for x in ['cifar', 'imagenet', 'oxford102flowers']
    }
    return trainloader, valloader
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochsLocal', type=int, default=60)
    parser.add_argument('--nEpochsGlobal', type=int, default=40)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/tri_tunnul_net.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # if os.path.exists(args.save):
    #     shutil.rmtree(args.save)
    # os.makedirs(args.save, exist_ok=True)
    
    trainLoader, valLoader = loadData()
    net = tri_tunnel_net.TriTunnelNet18()
    # net = torch.load('work/tri_tunnul_net.base/local.pth')
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    trainF = {
        'cifar': open(os.path.join(args.save, 'trainCifar.csv'), 'a'),
        'imagenet': open(os.path.join(args.save, 'trainImagenet.csv'), 'a'),
        'oxford102flowers': open(os.path.join(args.save, 'trainoxford102flowers.csv'), 'a'),
    }

    testF = {
        'cifar': open(os.path.join(args.save, 'testCifar.csv'), 'a'),
        'imagenet': open(os.path.join(args.save, 'testImagenet.csv'), 'a'),
        'oxford102flowers': open(os.path.join(args.save, 'testoxford102flowers.csv'), 'a'),
    }

    '''stage 1: local training'''
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=0.00015)
    for epoch in range(args.nEpochsLocal+80):
        adjust_opt1(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, 'oxford102flowers', mode = 'local')
        test(args, epoch, net, valLoader, optimizer, testF, 'oxford102flowers', mode = 'local')
        torch.save(net, os.path.join(args.save, 'local.pth'))
        # os.system('./plot.py {} &'.format(args.save))

    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.00015)
    for epoch in range(args.nEpochsLocal):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, 'cifar', mode = 'local')
        test(args, epoch, net, valLoader, optimizer, testF, 'cifar', mode = 'local')
        torch.save(net, os.path.join(args.save, 'local.pth'))
        # os.system('./plot.py {} &'.format(args.save))

    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=0.00015)
    for epoch in range(args.nEpochsLocal):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, 'imagenet', mode = 'local')
        test(args, epoch, net, valLoader, optimizer, testF, 'imagenet', mode = 'local')
        torch.save(net, os.path.join(args.save, 'local.pth'))
        # os.system('./plot.py {} &'.format(args.save))

    '''stage 2: global training'''
    for dataClass in ['oxford102flowers','cifar', 'imagenet']:
        if dataClass == 'cifar':
            for param in net.parameters():
                param.requires_grad = False
            params = []
            for m in net.advance2.modules():
                if isinstance(m, nn.Conv2d):
                   m.weight.requires_grad = True
                   params.append({'params': m.parameters()})
                elif isinstance(m, nn.BatchNorm2d):
                   m.weight.requires_grad = True
                   m.bias.requires_grad = True
                   params.append({'params': m.parameters()})
                elif isinstance(m, nn.Linear):
                   m.weight.requires_grad = True
                   m.bias.requires_grad = True
                   params.append({'params': m.parameters()})
            net.fc2.weight.requires_grad = True
            net.fc2.bias.requires_grad = True
            params.append({'params': net.fc2.parameters()})  
            optimizer = optim.SGD(params, lr=1e-1, momentum=0.9, weight_decay=0.00015)
            for epoch in range(args.nEpochsLocal, args.nEpochsLocal+args.nEpochsGlobal):
                adjust_opt(args.opt, optimizer, epoch)
                train(args, epoch, net, trainLoader, optimizer, trainF, dataClass, mode = 'global')
                test(args, epoch, net, valLoader, optimizer, testF, dataClass, mode = 'global')
                torch.save(net, os.path.join(args.save, 'global.pth'))
                # os.system('./plot.py {} &'.format(args.save))

        elif dataClass == 'imagenet':
            for param in net.parameters():
                param.requires_grad = False
            params = []
            for m in net.advance3.modules():
                if isinstance(m, nn.Conv2d):
                   m.weight.requires_grad = True
                   params.append({'params': m.parameters()})
                elif isinstance(m, nn.BatchNorm2d):
                   m.weight.requires_grad = True
                   m.bias.requires_grad = True
                   params.append({'params': m.parameters()})
                elif isinstance(m, nn.Linear):
                   m.weight.requires_grad = True
                   m.bias.requires_grad = True
                   params.append({'params': m.parameters()})
            net.fc3.weight.requires_grad = True
            net.fc3.bias.requires_grad = True
            params.append({'params': net.fc3.parameters()})  
            optimizer = optim.SGD(params, lr=1e-1, momentum=0.9, weight_decay=0.00015)
            for epoch in range(args.nEpochsLocal, args.nEpochsLocal+args.nEpochsGlobal):
                adjust_opt(args.opt, optimizer, epoch)
                train(args, epoch, net, trainLoader, optimizer, trainF, dataClass, mode = 'global')
                test(args, epoch, net, valLoader, optimizer, testF, dataClass, mode = 'global')
                torch.save(net, os.path.join(args.save, 'global.pth'))
                # os.system('./plot.py {} &'.format(args.save))

        elif dataClass == 'oxford102flowers':
            for param in net.parameters():
                param.requires_grad = False
            params = []
            for m in net.advance1.modules():
                if isinstance(m, nn.Conv2d):
                   m.weight.requires_grad = True
                   params.append({'params': m.parameters()})
                elif isinstance(m, nn.BatchNorm2d):
                   m.weight.requires_grad = True
                   m.bias.requires_grad = True
                   params.append({'params': m.parameters()})
                elif isinstance(m, nn.Linear):
                   m.weight.requires_grad = True
                   m.bias.requires_grad = True
                   params.append({'params': m.parameters()})
            net.fc1.weight.requires_grad = True
            net.fc1.bias.requires_grad = True
            params.append({'params': net.fc1.parameters()})  
            optimizer = optim.SGD(params, lr=1e-1, momentum=0.9, weight_decay=0.00015)
            for epoch in range(args.nEpochsLocal+80, args.nEpochsLocal+args.nEpochsGlobal+80):
               adjust_opt1(args.opt, optimizer, epoch)
               train(args, epoch, net, trainLoader, optimizer, trainF, dataClass, mode = 'global')
               test(args, epoch, net, valLoader, optimizer, testF, dataClass, mode = 'global')
               torch.save(net, os.path.join(args.save, 'global.pth'))
               # os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()


def train(args, epoch, net, trainLoader, optimizer, trainF, dataClass, mode):
    net.train() 
    nTrain = len(trainLoader[dataClass].dataset)
    nProcessed = 0
    for batch_idx, (data, target) in enumerate(trainLoader[dataClass]):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data, dataClass, mode)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader[dataClass])
        print('Data: {}, Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(dataClass,
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader[dataClass]), loss.data[0], err))
            
        trainF[dataClass].write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF[dataClass].flush()
        

def test(args, epoch, net, testLoader, optimizer, testF, dataClass, mode):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader[dataClass]:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data, dataClass, mode)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

     
    test_loss /= len(testLoader[dataClass]) # loss function already averages over batch size
    nTotal = len(testLoader[dataClass].dataset)
    err = 100.*incorrect/len(testLoader[dataClass].dataset)
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
            test_loss, incorrect, nTotal, err))

    testF[dataClass].write('{},{},{}\n'.format(epoch, test_loss, err))
    testF[dataClass].flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 40: lr = 1e-1
        elif epoch == 40: lr = 1e-2
        elif epoch == 50: lr = 1e-3
        elif epoch == 60: lr = 1e-1
        elif epoch == 80: lr = 1e-2
        elif epoch == 90: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_opt1(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 120: lr = 1e-1
        elif epoch == 120: lr = 1e-2
        elif epoch == 130: lr = 1e-3
        elif epoch == 140: lr = 1e-1
        elif epoch == 160: lr = 1e-2
        elif epoch == 170: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
