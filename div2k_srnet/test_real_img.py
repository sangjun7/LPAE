import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import *
from utils import *

#=======================================================================================================================================
parser = argparse.ArgumentParser(description='LPAE + SRNet for DIV2K')
parser.add_argument('--mag', default=2, type=int, help='Magnification of image (one of the number between 2,4,8)')
parser.add_argument('--imresize', action='store_true', help='Whether to resize input image or not')
parser.add_argument('--imresize_h', default=128, help='Height size to resize input image')
parser.add_argument('--imresize_w', default=128, help='Width size to resize input image')
parser.add_argument('--imrandcrop', action='store_true', help='Whether to crop randomly input image or not')
parser.add_argument('--imrandcrop_h', default=128, help='Height size to crop randomly input image')
parser.add_argument('--imrandcrop_w', default=128, help='Width size to crop randomly input image')

parser.add_argument('--testdata', default=None, help='Directory path for test images')
parser.add_argument('--trained_ae', default=None, help='Path for a trained Autoencoder model')
parser.add_argument('--trained_srnet', default=None, help='Path for a trained SR model')

parser.add_argument('--save', action='store_true', help='Whether to save result image or not')
parser.add_argument('--save_path', default=None, help='Path to save result image')

parser.add_argument('--workers', default=4, type=int, help='Number of workers')
parser.add_argument('--cuda', action='store_true', help='Use CUDA device')
parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs')
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
            nn.Conv2d(ich, och, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(och, och, 3, stride=1, padding=1))
        if ich != och:
            self.skip = nn.Conv2d(ich, och, 1, stride=1, padding=0)
        else:
            self.skip = None
        self.act = nn.ReLU()
        
    def forward(self, x):
        x1 = self.convlayer(x)
        if self.skip is not None:
            x2 = self.skip(x)
        else:
            x2 = x
        xadd = torch.add(x1, x2)
        out = self.act(xadd)
        
        return out

#=======================================================================================================================================
class wae_dec(nn.Module):
    def __init__(self):
        super(wae_dec, self).__init__()
        
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
        
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
        
        self.layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
            
    def forward(self, x):
        x_c = self.layer1_1(x)      
        x_d = self.layer1_2(x)
            
        return x_c, x_d
    
class wae_rec(nn.Module):
    def __init__(self):
        super(wae_rec, self).__init__()
        
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
        
        self.layer1_2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
        
        self.layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
            
    def forward(self, yc_rec, yd_rec):
        yc_rec_up = self.layer2_1(yc_rec)
        rec_pred = yd_rec + yc_rec_up
            
        return rec_pred

#=======================================================================================================================================
class srnet(nn.Module):
    def __init__(self, mag=args.mag, ch=128):
        super(srnet, self).__init__()
        
        self.mag = int(mag)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3+3, ch, 3, stride=1, padding=1), nn.ReLU())
        
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
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1))
        if self.mag >= 4:
            self.wp3 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.ConvTranspose2d(64, 3, kernel_size=6, stride=4, padding=1))
        if self.mag >= 8:
            self.wp4 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.ConvTranspose2d(64, 64, kernel_size=6, stride=4, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1))
        
        
    def forward(self, x):
        out = self.conv1(x)
        embout = self.embed(out)
        wpoutl = self.wp1(embout)
        
        wpouth = []
        if self.mag == 1:
            print("WARNING: You don't enlarge images since mag=1.")
        if self.mag >= 2:
            wpouth.append(self.wp2(embout))
        if self.mag >= 4:
            wpouth.append(self.wp3(embout))
        if self.mag >= 8:
            wpouth.append(self.wp4(embout))
        
        return wpoutl, wpouth

#=======================================================================================================================================
test_set = RealImgTest(args.testdata, is_resize=args.imresize, resize_h=args.imresize_h, resize_w=args.imresize_w, 
                       is_rcrop=args.imrandcrop, crop_h=args.imrandcrop_h, crop_w=args.imrandcrop_w, grayscale=False)
testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers)

net_rec = wae_rec()
testsrnet = srnet(mag=args.mag)
if args.trained_ae is None or args.trained_srnet is None:
    raise Exception("ERROR: trained_ae or trained_srnet argument is missing.")
if args.ngpu > 1:
    net_rec = nn.DataParallel(net_rec, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net_rec.module.load_state_dict(torch.load(args.trained_ae, map_location='cpu'))
    testsrnet = nn.DataParallel(testsrnet, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    testsrnet.module.load_state_dict(torch.load(args.trained_srnet, map_location='cpu'))
else:
    net_rec.load_state_dict(torch.load(args.trained_ae, map_location='cpu'))
    testsrnet.load_state_dict(torch.load(args.trained_srnet, map_location='cpu'))
net_rec = net_rec.to(device)
testsrnet = testsrnet.to(device)

#=======================================================================================================================================
net_rec.eval()
testsrnet.eval()
with torch.no_grad():
    time_sum = 0
    for testiter, testdata in enumerate(testloader):
        img = testdata
        if use_cuda:
            img = img.to(device)

        tinputs = torch.cat((img, gauss_noise(img, use_cuda=use_cuda, device=device)), 1)

        start_time = time.time()
        twpoutc, twpoutd = testsrnet(tinputs)
        if args.mag == 1:
            trec_pred = twpoutc
        elif args.mag == 2:
            trec_pred = net_rec(yc_rec=twpoutc, yd_rec=twpoutd[0])
        elif args.mag == 4:
            trec1 = net_rec(yc_rec=twpoutc, yd_rec=twpoutd[0])
            trec_pred = net_rec(yc_rec=trec1, yd_rec=twpoutd[1])
        elif args.mag == 8:
            trec1 = net_rec(yc_rec=twpoutc, yd_rec=twpoutd[0])
            trec2 = net_rec(yc_rec=trec1, yd_rec=twpoutd[1])
            trec_pred = net_rec(yc_rec=trec2, yd_rec=twpoutd[2])
        time_sum += (time.time() - start_time)

        print('The size of SR : Height: {}, Width: {}'.format(trec_pred.size(2), trec_pred.size(3)))
        if args.save:
            imgsave(trec_pred, args.save_path, 'result{}'.format(testiter+1))

    print("Time: {:.2f} s".format(time_sum))
