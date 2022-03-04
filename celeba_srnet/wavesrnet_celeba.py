
import time
import math
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pytorch_wavelets import DWTForward, DWTInverse

import torchvision
import torchvision.transforms as transforms

from dataset import *

#=======================================================================================================================================
parser = argparse.ArgumentParser(description='WaveletSRNet for CelebA')
parser.add_argument('--nepoch', default=40, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
parser.add_argument('--lr_schedule', default=10, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.1, type=float, help='Decay rate of learning rate')
parser.add_argument('--trainbatch', default=256, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=1, type=int, help='Test batch size')
parser.add_argument('--mag', default=2, type=int, help='Magnification of image (one of the number between 2,4,8)')

parser.add_argument('--traindata', default='/home/sjhan/datasets/celeba/img_align_celeba/celeba_train/train/', help='Path for train dataset')
parser.add_argument('--trainlist', default='/home/sjhan/datasets/celeba/img_align_celeba/train.list', help='List file for train dataset')
parser.add_argument('--train_num', default=162770, type=int, help='The number of all train image')
parser.add_argument('--testdata', default='/home/sjhan/datasets/celeba/img_align_celeba/celeba_test/test/', help='Path for validation dataset')
parser.add_argument('--testlist', default='/home/sjhan/datasets/celeba/img_align_celeba/test.list', help='List file for validation dataset')
parser.add_argument('--test_num', default=19962, type=int, help='The number of all test image')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='wavesrnet_face_2mag.pth', help='Name for trained model')
parser.add_argument('--test_only', action='store_true', help='Only test the trained model without training step')

parser.add_argument('--workers', default=2, type=int, help='Number of workers')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device')
parser.add_argument('--ngpu', default=2, type=int, help='Number of GPUs')
parser.add_argument('--initial_gpu', type=int, default=0, help='Initial gpu')

args = parser.parse_args()

#=======================================================================================================================================
use_cuda = False
device = torch.device('cpu')
if torch.cuda.is_available() is False and args.cuda:
    print("WARNING: You don't have a CUDA device so this code will be run on CPU.")
elif torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device but you don't use the device this time.")
elif torch.cuda.is_available() and args.cuda:
    use_cuda = True
    device = torch.device('cuda:{}'.format(args.initial_gpu))
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

#=======================================================================================================================================
class block(nn.Module):
    def __init__(self, ich, och):
        super(block, self).__init__()
        
        self.convlayer = nn.Sequential(
            nn.Conv2d(ich, och, 3, stride=1, padding=1), nn.BatchNorm2d(och), nn.ReLU(),
            nn.Conv2d(och, och, 3, stride=1, padding=1))
        if ich != och:
            self.skip = nn.Conv2d(ich, och, 1, stride=1, padding=0)
        else:
            self.skip = None
        self.bn = nn.BatchNorm2d(och)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x1 = self.convlayer(x)
        if self.skip is not None:
            x2 = self.skip(x)
        else:
            x2 = x
        xadd = torch.add(x1, x2)
        out = self.act(self.bn(xadd))
        
        return out

#=======================================================================================================================================
def wavelet_dec(data, mag=args.mag, basis='haar', pad_mode='zero', device=device):
    dwt1 = DWTForward(J=1, mode=pad_mode, wave=basis)
    dwt1 = dwt1.to(device)
    if mag == 1:
        outl = data
        outh = None
        
        return outl, outh
    
    if mag == 2:
        outl, h = dwt1(data)
        outh = [h[0][:,:,0,:,:], h[0][:,:,1,:,:], h[0][:,:,2,:,:]]
        
        return outl, outh
        
    if mag == 4:
        a, b = dwt1(data)
        
        outl, h1 = dwt1(a)
        outh = [h1[0][:,:,0,:,:], h1[0][:,:,1,:,:], h1[0][:,:,2,:,:]]
        l2, h2 = dwt1(b[0][:,:,0,:,:])
        col2 = [l2, h2[0][:,:,0,:,:], h2[0][:,:,1,:,:], h2[0][:,:,2,:,:]]
        outh.extend(col2)
        l3, h3 = dwt1(b[0][:,:,1,:,:])
        col3 = [l3, h3[0][:,:,0,:,:], h3[0][:,:,1,:,:], h3[0][:,:,2,:,:]]
        outh.extend(col3)
        l4, h4 = dwt1(b[0][:,:,2,:,:])
        col4 = [l4, h4[0][:,:,0,:,:], h4[0][:,:,1,:,:], h4[0][:,:,2,:,:]]
        outh.extend(col4)
            
        return outl, outh 
    
    if mag == 8:
        a, b = dwt1(data)
        
        aa, ab = dwt1(a)
        outl, h1 = dwt1(aa)
        outh = [h1[0][:,:,0,:,:], h1[0][:,:,1,:,:], h1[0][:,:,2,:,:]]
        l2, h2 = dwt1(ab[0][:,:,0,:,:])
        col2 = [l2, h2[0][:,:,0,:,:], h2[0][:,:,1,:,:], h2[0][:,:,2,:,:]]
        outh.extend(col2)
        l3, h3 = dwt1(ab[0][:,:,1,:,:])
        col3 = [l3, h3[0][:,:,0,:,:], h3[0][:,:,1,:,:], h3[0][:,:,2,:,:]]
        outh.extend(col3)
        l4, h4 = dwt1(ab[0][:,:,2,:,:])
        col4 = [l4, h4[0][:,:,0,:,:], h4[0][:,:,1,:,:], h4[0][:,:,2,:,:]]
        outh.extend(col4)
        
        aa, ab = dwt1(b[0][:,:,0,:,:])
        l5, h5 = dwt1(aa)
        col5 = [l5, h5[0][:,:,0,:,:], h5[0][:,:,1,:,:], h5[0][:,:,2,:,:]]
        outh.extend(col5)
        l6, h6 = dwt1(ab[0][:,:,0,:,:])
        col6 = [l6, h6[0][:,:,0,:,:], h6[0][:,:,1,:,:], h6[0][:,:,2,:,:]]
        outh.extend(col6)
        l7, h7 = dwt1(ab[0][:,:,1,:,:])
        col7 = [l7, h7[0][:,:,0,:,:], h7[0][:,:,1,:,:], h7[0][:,:,2,:,:]]
        outh.extend(col7)
        l8, h8 = dwt1(ab[0][:,:,2,:,:])
        col8 = [l8, h8[0][:,:,0,:,:], h8[0][:,:,1,:,:], h8[0][:,:,2,:,:]]
        outh.extend(col8)
        
        aa, ab = dwt1(b[0][:,:,1,:,:])
        l9, h9 = dwt1(aa)
        col9 = [l9, h9[0][:,:,0,:,:], h9[0][:,:,1,:,:], h9[0][:,:,2,:,:]]
        outh.extend(col9)
        l10, h10 = dwt1(ab[0][:,:,0,:,:])
        col10 = [l10, h10[0][:,:,0,:,:], h10[0][:,:,1,:,:], h10[0][:,:,2,:,:]]
        outh.extend(col10)
        l11, h11 = dwt1(ab[0][:,:,1,:,:])
        col11 = [l11, h11[0][:,:,0,:,:], h11[0][:,:,1,:,:], h11[0][:,:,2,:,:]]
        outh.extend(col11)
        l12, h12 = dwt1(ab[0][:,:,2,:,:])
        col12 = [l12, h12[0][:,:,0,:,:], h12[0][:,:,1,:,:], h12[0][:,:,2,:,:]]
        outh.extend(col12)
        
        aa, ab = dwt1(b[0][:,:,2,:,:])
        l13, h13 = dwt1(aa)
        col13 = [l13, h13[0][:,:,0,:,:], h13[0][:,:,1,:,:], h13[0][:,:,2,:,:]]
        outh.extend(col13)
        l14, h14 = dwt1(ab[0][:,:,0,:,:])
        col14 = [l14, h14[0][:,:,0,:,:], h14[0][:,:,1,:,:], h14[0][:,:,2,:,:]]
        outh.extend(col14)
        l15, h15 = dwt1(ab[0][:,:,1,:,:])
        col15 = [l15, h15[0][:,:,0,:,:], h15[0][:,:,1,:,:], h15[0][:,:,2,:,:]]
        outh.extend(col15)
        l16, h16 = dwt1(ab[0][:,:,2,:,:])
        col16 = [l16, h16[0][:,:,0,:,:], h16[0][:,:,1,:,:], h16[0][:,:,2,:,:]]
        outh.extend(col16)
        
        return outl, outh
        
    else:
        print('please choose the value of magnification between [1,2,4,8]')
    
def wavelet_rec(l, h, mag=args.mag, basis='haar', pad_mode='zero', device=device):
    idwt = DWTInverse(mode=pad_mode, wave=basis)
    idwt = idwt.to(device)
    if mag == 1:
        out = l
        
    if mag == 2:
        sz = l.shape
        outh = torch.cat((h[0].view(sz[0],sz[1],-1,sz[2],sz[3]), h[1].view(sz[0],sz[1],-1,sz[2],sz[3]), h[2].view(sz[0],sz[1],-1,sz[2],sz[3])),2)
        outh = [outh]
        out = idwt((l, outh))
        
    if mag == 4:
        sz = l.shape
        
        outh0 = torch.cat((h[0].view(sz[0],sz[1],-1,sz[2],sz[3]), h[1].view(sz[0],sz[1],-1,sz[2],sz[3]), h[2].view(sz[0],sz[1],-1,sz[2],sz[3])),2)
        
        a = torch.cat((h[4].view(sz[0],sz[1],-1,sz[2],sz[3]), h[5].view(sz[0],sz[1],-1,sz[2],sz[3]), h[6].view(sz[0],sz[1],-1,sz[2],sz[3])),2)
        a = [a]
        outh1 = idwt((h[3], a))
        
        b = torch.cat((h[8].view(sz[0],sz[1],-1,sz[2],sz[3]), h[9].view(sz[0],sz[1],-1,sz[2],sz[3]), h[10].view(sz[0],sz[1],-1,sz[2],sz[3])),2)
        b = [b]
        outh2 = idwt((h[7], b))
        
        c = torch.cat((h[12].view(sz[0],sz[1],-1,sz[2],sz[3]), h[13].view(sz[0],sz[1],-1,sz[2],sz[3]), h[14].view(sz[0],sz[1],-1,sz[2],sz[3])),2)
        c = [c]
        outh3 = idwt((h[11], c))
        
        sz2 = outh1.shape
        outhh = torch.cat((outh1.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), outh2.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), outh3.view(sz2[0],sz2[1],-1,sz2[2],sz2[3])),2)
        outh = [outhh, outh0]
        out = idwt((l, outh))
        
    if mag == 8:
        sz = l.shape
        
        outh0 = torch.cat((h[0].view(sz[0],sz[1],-1,sz[2],sz[3]), h[1].view(sz[0],sz[1],-1,sz[2],sz[3]), h[2].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        
        a = torch.cat((h[4].view(sz[0],sz[1],-1,sz[2],sz[3]), h[5].view(sz[0],sz[1],-1,sz[2],sz[3]), h[6].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h1 = idwt((h[3], a))
        a = torch.cat((h[8].view(sz[0],sz[1],-1,sz[2],sz[3]), h[9].view(sz[0],sz[1],-1,sz[2],sz[3]), h[10].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h2 = idwt((h[7], a))
        a = torch.cat((h[12].view(sz[0],sz[1],-1,sz[2],sz[3]), h[13].view(sz[0],sz[1],-1,sz[2],sz[3]), h[14].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h3 = idwt((h[11], a))
        sz2 = h1.shape
        outh1 = torch.cat((h1.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h2.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h3.view(sz2[0],sz2[1],-1,sz2[2],sz2[3])), 2)
        
        a = torch.cat((h[16].view(sz[0],sz[1],-1,sz[2],sz[3]), h[17].view(sz[0],sz[1],-1,sz[2],sz[3]), h[18].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h1 = idwt((h[15], a))
        a = torch.cat((h[20].view(sz[0],sz[1],-1,sz[2],sz[3]), h[21].view(sz[0],sz[1],-1,sz[2],sz[3]), h[22].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h2 = idwt((h[19], a))
        a = torch.cat((h[24].view(sz[0],sz[1],-1,sz[2],sz[3]), h[25].view(sz[0],sz[1],-1,sz[2],sz[3]), h[26].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h3 = idwt((h[23], a))
        a = torch.cat((h[28].view(sz[0],sz[1],-1,sz[2],sz[3]), h[29].view(sz[0],sz[1],-1,sz[2],sz[3]), h[30].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h4 = idwt((h[27], a))
        sz2 = h1.shape
        b = torch.cat((h2.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h3.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h4.view(sz2[0],sz2[1],-1,sz2[2],sz2[3])), 2)
        b = [b]
        h_l1 = idwt((h1, b))
        
        a = torch.cat((h[32].view(sz[0],sz[1],-1,sz[2],sz[3]), h[33].view(sz[0],sz[1],-1,sz[2],sz[3]), h[34].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h1 = idwt((h[31], a))
        a = torch.cat((h[36].view(sz[0],sz[1],-1,sz[2],sz[3]), h[37].view(sz[0],sz[1],-1,sz[2],sz[3]), h[38].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h2 = idwt((h[35], a))
        a = torch.cat((h[40].view(sz[0],sz[1],-1,sz[2],sz[3]), h[41].view(sz[0],sz[1],-1,sz[2],sz[3]), h[42].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h3 = idwt((h[39], a))
        a = torch.cat((h[44].view(sz[0],sz[1],-1,sz[2],sz[3]), h[45].view(sz[0],sz[1],-1,sz[2],sz[3]), h[46].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h4 = idwt((h[43], a))
        sz2 = h1.shape
        b = torch.cat((h2.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h3.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h4.view(sz2[0],sz2[1],-1,sz2[2],sz2[3])), 2)
        b = [b]
        h_l2 = idwt((h1, b))
        
        a = torch.cat((h[48].view(sz[0],sz[1],-1,sz[2],sz[3]), h[49].view(sz[0],sz[1],-1,sz[2],sz[3]), h[50].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h1 = idwt((h[47], a))
        a = torch.cat((h[52].view(sz[0],sz[1],-1,sz[2],sz[3]), h[53].view(sz[0],sz[1],-1,sz[2],sz[3]), h[54].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h2 = idwt((h[51], a))
        a = torch.cat((h[56].view(sz[0],sz[1],-1,sz[2],sz[3]), h[57].view(sz[0],sz[1],-1,sz[2],sz[3]), h[58].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h3 = idwt((h[55], a))
        a = torch.cat((h[60].view(sz[0],sz[1],-1,sz[2],sz[3]), h[61].view(sz[0],sz[1],-1,sz[2],sz[3]), h[62].view(sz[0],sz[1],-1,sz[2],sz[3])), 2)
        a = [a]
        h4 = idwt((h[59], a))
        sz2 = h1.shape
        b = torch.cat((h2.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h3.view(sz2[0],sz2[1],-1,sz2[2],sz2[3]), h4.view(sz2[0],sz2[1],-1,sz2[2],sz2[3])), 2)
        b = [b]
        h_l3 = idwt((h1, b))
        
        sz3 = h_l1.shape
        outh2 = torch.cat((h_l1.view(sz3[0],sz3[1],-1,sz3[2],sz3[3]), h_l2.view(sz3[0],sz3[1],-1,sz3[2],sz3[3]), h_l3.view(sz3[0],sz3[1],-1,sz3[2],sz3[3])), 2)
        
        outh = [outh2, outh1, outh0]
        out = idwt((l, outh))
        
    return out

#=======================================================================================================================================
class srnet(nn.Module):
    def __init__(self, mag=args.mag, ch=128):
        super(srnet, self).__init__()
        
        self.mag = int(mag)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, ch, 3, stride=1, padding=1), nn.BatchNorm2d(ch), nn.ReLU())
        
        self.embed = nn.Sequential(
            block(ch, ch),
            block(ch, ch),
            block(ch, 2*ch),
            block(2*ch, 2*ch),
            block(2*ch, 4*ch),
            block(4*ch, 4*ch),
            block(4*ch, 8*ch),
            block(8*ch, 8*ch))
        
        if self.mag >= 1:
            self.wp1 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
        if self.mag >= 2:
            self.wp2 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp3 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp4 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
        if self.mag >= 4:
            self.wp5 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp6 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp7 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp8 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp9 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp10 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp11 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp12 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp13 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp14 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp15 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp16 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
        if self.mag >= 8:
            self.wp17 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp18 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp19 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp20 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp21 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp22 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp23 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp24 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp25 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp26 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp27 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp28 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp29 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp30 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp31 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp32 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp33 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp34 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp35 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp36 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp37 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp38 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp39 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp40 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp41 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp42 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp43 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp44 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp45 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp46 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp47 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp48 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp49 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp50 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp51 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp52 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp53 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp54 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp55 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp56 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp57 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp58 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp59 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp60 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp61 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp62 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp63 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
            self.wp64 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
        
    def forward(self, x):
        out = self.conv1(x)
        embout = self.embed(out)
        wpoutl = self.wp1(embout)
        
        wpouth = []
        if self.mag == 1:
            print("WARNING: You don't enlarge images since mag=1.")
        if self.mag >= 2:
            wpouth.append(self.wp2(embout))
            wpouth.append(self.wp3(embout))
            wpouth.append(self.wp4(embout))
        if self.mag >= 4:
            wpouth.append(self.wp5(embout))
            wpouth.append(self.wp6(embout))
            wpouth.append(self.wp7(embout))
            wpouth.append(self.wp8(embout))
            wpouth.append(self.wp9(embout))
            wpouth.append(self.wp10(embout))
            wpouth.append(self.wp11(embout))
            wpouth.append(self.wp12(embout))
            wpouth.append(self.wp13(embout))
            wpouth.append(self.wp14(embout))
            wpouth.append(self.wp15(embout))
            wpouth.append(self.wp16(embout))
        if self.mag >= 8:
            wpouth.append(self.wp17(embout))
            wpouth.append(self.wp18(embout))
            wpouth.append(self.wp19(embout))
            wpouth.append(self.wp20(embout))
            wpouth.append(self.wp21(embout))
            wpouth.append(self.wp22(embout))
            wpouth.append(self.wp23(embout))
            wpouth.append(self.wp24(embout))
            wpouth.append(self.wp25(embout))
            wpouth.append(self.wp26(embout))
            wpouth.append(self.wp27(embout))
            wpouth.append(self.wp28(embout))
            wpouth.append(self.wp29(embout))
            wpouth.append(self.wp30(embout))
            wpouth.append(self.wp31(embout))
            wpouth.append(self.wp32(embout))
            wpouth.append(self.wp33(embout))
            wpouth.append(self.wp34(embout))
            wpouth.append(self.wp35(embout))
            wpouth.append(self.wp36(embout))
            wpouth.append(self.wp37(embout))
            wpouth.append(self.wp38(embout))
            wpouth.append(self.wp39(embout))
            wpouth.append(self.wp40(embout))
            wpouth.append(self.wp41(embout))
            wpouth.append(self.wp42(embout))
            wpouth.append(self.wp43(embout))
            wpouth.append(self.wp44(embout))
            wpouth.append(self.wp45(embout))
            wpouth.append(self.wp46(embout))
            wpouth.append(self.wp47(embout))
            wpouth.append(self.wp48(embout))
            wpouth.append(self.wp49(embout))
            wpouth.append(self.wp50(embout))
            wpouth.append(self.wp51(embout))
            wpouth.append(self.wp52(embout))
            wpouth.append(self.wp53(embout))
            wpouth.append(self.wp54(embout))
            wpouth.append(self.wp55(embout))
            wpouth.append(self.wp56(embout))
            wpouth.append(self.wp57(embout))
            wpouth.append(self.wp58(embout))
            wpouth.append(self.wp59(embout))
            wpouth.append(self.wp60(embout))
            wpouth.append(self.wp61(embout))
            wpouth.append(self.wp62(embout))
            wpouth.append(self.wp63(embout))
            wpouth.append(self.wp64(embout))
        
        return wpoutl, wpouth      

#=======================================================================================================================================
def loss_MSE(x, y):
    z = torch.mean((x - y)**2)
    return z

def loss_wavelet(x, y, lamb):
    z = 0
    for i in range(len(x)):
        z += lamb * loss_MSE(x[i], y[i])
    return z

def loss_texture(h, hpred, alpha, ep, gam):
    z = 0
    for i in range(len(h)):
        z += gam * F.relu(alpha * loss_MSE(h[i], 0) + ep - loss_MSE(hpred[i], 0))
    return z

def cal_psnr(img, pred, gray_scale=True):
    
    img = img.cpu().numpy().transpose(0, 2, 3, 1)
    img = img[0, ...].squeeze()
        
    pred = torch.clamp(pred,0,1)
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
    pred = pred[0, ...].squeeze()
    
    if gray_scale:
        coeff = np.array([65.738, 129.057, 25.064]).reshape(1, 1, 3) / 255.0
        img = img * coeff
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        pred = pred * coeff
        pred = pred[:, :, 0] + pred[:, :, 1] + pred[:, :, 2]
        
        diff = img - pred
        mse = np.mean(diff ** 2)
        psnr = -10.0 * np.log10(mse)
    else:
        diff = img - pred
        mse = np.mean(diff ** 2)
        psnr = -10.0 * np.log10(mse)

    return psnr

def get_ssim(x, y, data_range=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, grayscale=True, device=device):
    if grayscale:
        coeff = (torch.tensor(np.array([65.738, 129.057, 25.064]).reshape(1, 3, 1, 1) / 255.0, dtype=torch.float)).to(device)
        x = x * coeff
        x = x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]
        x = x.unsqueeze(0)
        y = y * coeff
        y = y[:, 0, :, :] + y[:, 1, :, :] + y[:, 2, :, :]
        y = y.unsqueeze(0)
        
    b, ch, h, w = x.size()
    L = data_range
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    pad = filter_size // 2
    
    gauss =  torch.Tensor([math.exp(-(x - filter_size//2)**2/float(2*filter_sigma**2)) for x in range(filter_size)])
    d1_window = (gauss/gauss.sum()).unsqueeze(1)
    d2_window = d1_window.mm(d1_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(d2_window.expand(3, 1, filter_size, filter_size).contiguous()).to(device)
    
    mu1 = F.conv2d(x, window, padding=pad, groups=ch)
    mu2 = F.conv2d(y, window, padding=pad, groups=ch)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2
    
    sig1_sq = F.conv2d(x*x, window, padding=pad, groups=ch) - mu1_sq
    sig2_sq = F.conv2d(y*y, window, padding=pad, groups=ch) - mu2_sq
    sig12 = F.conv2d(x*y, window, padding=pad, groups=ch) - mu12
    
    l_comp = (2*mu12 + c1)/(mu1_sq + mu2_sq + c1)
    cs_comp = (2*sig12 + c2)/(sig1_sq + sig2_sq + c2)
    ssim = l_comp * cs_comp
    result = ssim.mean()
    
    return result

#=======================================================================================================================================
train_list, _ = loadFromFile(args.trainlist, args.train_num)    
train_set = ImageDatasetFromFile(train_list, args.traindata, 
                                 input_height=134, output_height=128, crop_height=128,
                                 is_random_crop=True, is_mirror=True, is_gray=False, 
                                 upscale=args.mag, is_scale_back=False)    
trainloader = DataLoader(train_set, batch_size=args.trainbatch, shuffle=True, num_workers=args.workers)
    
test_list, _ = loadFromFile(args.testlist, args.test_num)
test_set = ImageDatasetFromFile(test_list, args.testdata, 
                                input_height=128, output_height=128, crop_height=None,
                                is_random_crop=False, is_mirror=False, is_gray=False, 
                                upscale=args.mag, is_scale_back=False)    
testloader = DataLoader(test_set, batch_size=args.testbatch, shuffle=False, num_workers=args.workers)

alpha = 1.2
ep = 0
nu = 0.1
miu = 1
lamb1 = 0.01
lamb = 1
gam = 1

#=======================================================================================================================================
if args.test_only is True :
    testsrnet = srnet(mag=args.mag)
    if use_cuda and args.ngpu > 1:
        testsrnet = nn.DataParallel(testsrnet, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        testsrnet.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        testsrnet = testsrnet.to(device)
    elif (args.ngpu <= 1) or (use_cuda is False):
        testsrnet.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        testsrnet = testsrnet.to(device)
        
    testsrnet.eval()
    with torch.no_grad():
        psnr_sum = 0
        ssim_sum = 0
        time_sum = 0
        for testiter, testdata in enumerate(testloader):
            tinputs, ttargets = testdata[0], testdata[1]
            if use_cuda:
                tinputs = tinputs.to(device)
                ttargets = ttargets.to(device)

            start_time = time.time()
            twpoutl, twpouth = testsrnet(tinputs)
            trec_pred = wavelet_rec(twpoutl, twpouth, mag=args.mag, basis='haar', device=device)
            time_sum += (time.time() - start_time)

            psnr = cal_psnr(ttargets, trec_pred, gray_scale=True)
            psnr_sum += psnr
            ssim = get_ssim(ttargets, trec_pred, data_range=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, grayscale=True, device=device)
            ssim_sum += ssim.item()

        tinfo = "Average time: {:.2f} s, Average PSNR: {:.2f} dB, Average SSIM: {:.4f}".format(time_sum/len(testloader), psnr_sum/len(testloader), ssim_sum/len(testloader))
        print(tinfo)
        
#=======================================================================================================================================
elif args.test_only is False :
    net = srnet(mag=args.mag)
    if use_cuda and args.ngpu > 1:
        net = nn.DataParallel(net, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)

    net.train()
    start_time = time.time()
    for epoch in range(args.nepoch):
        loss_sum = 0
        for iteration, data in enumerate(trainloader):
        #---------------------train--------------------------
            inputs, targets = data[0], data[1].requires_grad_(requires_grad=False) 
            if use_cuda:
                inputs = inputs.to(device)
                targets = targets.to(device)

            inputl, inputh = wavelet_dec(targets, mag=args.mag, basis='haar', device=device)

            optimizer.zero_grad()
            wpoutl, wpouth = net(inputs)

            rec_pred = wavelet_rec(wpoutl, wpouth, mag=args.mag, basis='haar', device=device)

            loss_full = loss_MSE(targets, rec_pred)
            loss_wav = lamb1 * loss_MSE(inputl, wpoutl) + loss_wavelet(inputh, wpouth, lamb)
            loss_tex = loss_texture(inputh, wpouth, alpha, ep, gam)
            loss = loss_wav + miu * loss_tex + nu * loss_full

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if iteration % 100 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "full: {:.5f}, wav: {:.5f}, tex: {:.5f}, ".format(loss_full.item(), loss_wav.item(), loss_tex.item())
                info += "iter_loss: {:.5f} ".format(loss.item())
                print(info)

        print("Epoch : [{}/{}], loss = {:.5f}".format(epoch + 1, args.nepoch, loss_sum/len(trainloader)))

        #---------------------test-------------------------- 
        if epoch % 4 == 0 or epoch == (args.nepoch - 1):
            psnr_sum = 0
            ssim_sum = 0
            net.eval()
            for testiter, testdata in enumerate(testloader):
                tinputs, ttargets = testdata[0], testdata[1]
                if use_cuda:
                    tinputs = tinputs.to(device)
                    ttargets = ttargets.to(device)

                twpoutl, twpouth = net(tinputs)
                trec_pred = wavelet_rec(twpoutl, twpouth, mag=args.mag, basis='haar', device=device)

                psnr = cal_psnr(ttargets, trec_pred.detach(), gray_scale=True)
                psnr_sum += psnr
                ssim = get_ssim(ttargets, trec_pred, data_range=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, grayscale=True, device=device)
                ssim_sum += ssim.item()

            tinfo = "average PSNR: {:.2f} dB, Average SSIM: {:.4f}".format(psnr_sum/len(testloader), ssim_sum/len(testloader))
            print(tinfo)

            net.train() 
        scheduler.step()

    print('train over')

#=======================================================================================================================================
    if args.ngpu <= 1 or use_cuda is False:
        torch.save(net.state_dict(), '/'.join([args.model_save, args.model_name]))
    elif use_cuda and args.ngpu > 1:
        torch.save(net.module.state_dict(), '/'.join([args.model_save, args.model_name]))

#=======================================================================================================================================
