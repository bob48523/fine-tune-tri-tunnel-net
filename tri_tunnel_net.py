
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class TriTunnelNet(nn.Module):
    def __init__(self, flowers_branch, cifar_branch, imagenet_branch):
        super(TriTunnelNet, self).__init__()
        
        self.base1 = flowers_branch 
        self.base2 = cifar_branch
        self.base3 = imagenet_branch
        
        self.in_planes = 1024
        self.fc1 = nn.Linear(self.in_planes, 102)
        self.fc2 = nn.Linear(self.in_planes, 100)
        self.fc3 = nn.Linear(self.in_planes, 1000)

    def forward(self, x, datatype):
        out1 = self.base1(x)
        out2 = self.base2(x)
        out3 = self.base3(x)

        '''(1) add'''
        out = out1+out2+out3         
        '''(2) cat'''
        #out = torch.cat((out1, out2), 1)
        #out = torch.cat((out, out3), 1)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print(out)
        if datatype == 'oxford102flowers':
           out = self.fc1(out)
        elif datatype == 'cifar':
           out = self.fc2(out)
        elif datatype == 'imagenet':
           out = self.fc3(out)
        return out

def TriTunnelNet18(flowers_branch, cifar_branch, imagenet_branch):
    return TriTunnelNet(flowers_branch, cifar_branch, imagenet_branch)
    
