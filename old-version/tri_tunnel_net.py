
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class SE(nn.Module):
    def __init__(self, in_planes):
        super(SE, self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes//16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_planes//16, in_planes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        w = F.avg_pool2d(x, kernel_size=x.size(2))
        w = self.fc1(w.view(w.size(0),-1))
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        w = w.view(w.size(0),w.size(1),1,1)
        w = w.repeat(1,1,x.size(2),x.size(3))
        out = w*x      
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    widen_factor = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*self.widen_factor, kernel_size=3, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes*self.widen_factor)
        self.se = SE(planes*self.widen_factor)

        if stride != 1 or in_planes != planes*self.widen_factor:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.widen_factor, kernel_size=1, stride=stride, bias=False)             
            )
            

    def forward(self, x):
        out = self.bn1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(out)     

        out = self.bn2(out) 
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)               
       
        #out = self.se(out)
              
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    widen_factor = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.widen_factor, kernel_size=1, bias=False)   
        
        self.bn4 = nn.BatchNorm2d(planes*self.widen_factor)
        #self.se = SE(planes*self.widen_factor)

        if stride != 1 or in_planes != planes*self.widen_factor:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.widen_factor, kernel_size=1, stride=stride, bias=False)             
            )
            
            

    def forward(self, x):
        out = self.bn1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        
        out = self.conv1(out) 
        out = self.conv2(F.relu(self.bn2(out)))       
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.bn4(out)  
        out = self.se(out)          
        out += shortcut
        return out


class TriTunnelNet(nn.Module):
    def __init__(self, block, num_blocks, cifar_classes, imagenet_classes, flowers_classes):
        super(TriTunnelNet, self).__init__()
        self.in_planes = 32
        width = [32, 64, 128, 256]
        
        self.base1 = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            self._make_layer(block, width[0], num_blocks[0], stride=2),
            self._make_layer(block, width[1], num_blocks[1], stride=2),
            self._make_layer(block, width[2], num_blocks[2], stride=2),
            nn.BatchNorm2d(self.in_planes),
        )
        self.in_planes = 32
        self.base2 = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            self._make_layer(block, width[0], num_blocks[0], stride=2),
            self._make_layer(block, width[1], num_blocks[1], stride=2),
            self._make_layer(block, width[2], num_blocks[2], stride=2),
            nn.BatchNorm2d(self.in_planes),
        )
        self.in_planes = 32
        self.base3 = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            self._make_layer(block, width[0], num_blocks[0], stride=2),
            self._make_layer(block, width[1], num_blocks[1], stride=2),
            self._make_layer(block, width[2], num_blocks[2], stride=2),
            nn.BatchNorm2d(self.in_planes),
        )
        advance_inplanes = self.in_planes*3
        self.in_planes = advance_inplanes
        self.advance1 = self._make_layer(block, width[3], num_blocks[3], stride=2)
        self.in_planes = advance_inplanes
        self.advance2 = self._make_layer(block, width[3], num_blocks[3], stride=2)
        self.in_planes = advance_inplanes
        self.advance3 = self._make_layer(block, width[3], num_blocks[3], stride=2)
        
        self.fc1 = nn.Linear(self.in_planes, flowers_classes)
        self.fc2 = nn.Linear(self.in_planes, cifar_classes)
        self.fc3 = nn.Linear(self.in_planes, imagenet_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.widen_factor
        return nn.Sequential(*layers)

    def forward(self, x, datatype, mode):
        if mode == 'local':
           if datatype == 'oxford102flowers':
              out = self.base1(x)
              print(out)
              paddingdata = torch.zeros(out.size())
              paddingdata = paddingdata.cuda()
              paddingdata = Variable(paddingdata)
              out = torch.cat((out, paddingdata), 1)
              out = torch.cat((out, paddingdata), 1)
              out = F.avg_pool2d(self.advance1(out), 4)
              out = out.view(out.size(0), -1)
              out = self.fc1(out)
           elif datatype == 'cifar':
              out = self.base2(x)
              paddingdata = torch.zeros(out.size())
              paddingdata = paddingdata.cuda()
              paddingdata = Variable(paddingdata)
              out = torch.cat((out, paddingdata), 1)
              out = torch.cat((out, paddingdata), 1)
              out = F.avg_pool2d(self.advance2(out), 4)
              out = out.view(out.size(0), -1)
              out = self.fc2(out)
           elif datatype == 'imagenet':
              out = self.base3(x)
              paddingdata = torch.zeros(out.size())
              paddingdata = paddingdata.cuda()
              paddingdata = Variable(paddingdata)
              out = torch.cat((out, paddingdata), 1)
              out = torch.cat((out, paddingdata), 1)
              out = F.avg_pool2d(self.advance3(out), 4)
              out = out.view(out.size(0), -1)
              out = self.fc3(out)
        elif mode == 'global':
            out1 = self.base1(x)
            out2 = self.base2(x)
            out3 = self.base3(x)
            #paddingdata = torch.zeros(out.size())
            #paddingdata = paddingdata.cuda()
            #paddingdata = Variable(paddingdata)
            #out = torch.cat((out, paddingdata), 1)
            #out = torch.cat((out, paddingdata), 1)
            out = torch.cat((out1, out2), 1)
            out = torch.cat((out, out3), 1)
            if datatype == 'oxford102flowers':
               out = F.avg_pool2d(self.advance1(out), 4)
               out = out.view(out.size(0), -1)
               out = self.fc1(out)
            elif datatype == 'cifar':
               out = F.avg_pool2d(self.advance2(out), 4)
               out = out.view(out.size(0), -1)
               out = self.fc2(out)
            elif datatype == 'imagenet':
               out = F.avg_pool2d(self.advance3(out), 4)
               out = out.view(out.size(0), -1)
               out = self.fc3(out)
        return out

def TriTunnelNet18():
    return TriTunnelNet(PreActBlock, [2,2,2,2], 100, 1000, 102)
    
