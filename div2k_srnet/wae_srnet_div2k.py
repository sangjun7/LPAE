
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

import torchvision
import torchvision.transforms as transforms

from dataset import *

#=======================================================================================================================================
parser = argparse.ArgumentParser(description='WAE + SRNet for DIV2K')
parser.add_argument('--nepoch', default=300, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--lr_schedule', default=50, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Decay rate of learning rate')
parser.add_argument('--trainbatch', default=8, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=1, type=int, help='Test batch size')
parser.add_argument('--mag', default=2, type=int, help='Magnification of image (one of the number between 2,4,8)')

parser.add_argument('--traindata', default='/home/sjhan/datasets/div2k/train_HR/DIV2K_train_HR/', help='Path for train dataset')
parser.add_argument('--trainlist', default='/home/sjhan/datasets/div2k/train.list', help='List file for train dataset')
parser.add_argument('--train_num', default=800, type=int, help='The number of all train image')
parser.add_argument('--testdata', default='/home/sjhan/datasets/div2k/valid_HR/DIV2K_valid_HR/', help='Path for validation dataset')
parser.add_argument('--testlist', default='/home/sjhan/datasets/div2k/test.list', help='List file for validation dataset')
parser.add_argument('--test_num', default=100, type=int, help='The number of all test image')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='waveletlike_autoencoder_srnet_div2k_2mag.pth', help='Name for trained model')
parser.add_argument('--pretrained_AE', default='./model_save/wae_papersetting_div2k.pth', help='pretrained WAE parameters')
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
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1))
        
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
            
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
        
        self.layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
        
        self.layer3_2 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
            
    def forward(self, x):
        x = self.layer1(x)      
        x_L = self.layer2_1(x)
        x_H = self.layer2_2(x)
            
        return x_L, x_H
    
class wae_rec(nn.Module):
    def __init__(self):
        super(wae_rec, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1))
        
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
            
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
        
        self.layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
        
        self.layer3_2 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 4, stride=2, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1))
            
    def forward(self, yc_rec, yd_rec):
        yc_rec_up = self.layer3_1(yc_rec)
        yd_rec_up = self.layer3_2(yd_rec)
        rec_pred =  yc_rec_up + yd_rec_up
            
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
                nn.Conv2d(64, 3, 3, stride=1, padding=1))
        if self.mag >= 4:
            self.wp3 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1))
        if self.mag >= 8:
            self.wp4 = nn.Sequential(
                block(8*ch, 32),
                block(32, 64),
                nn.ConvTranspose2d(64, 3, kernel_size=6, stride=4, padding=1))
        
        
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
l1_criterion = nn.L1Loss().to(device)

#=======================================================================================================================================
def loss_MSE(x, y):
    z = torch.mean((x - y)**2)
    return z

def loss_wavelet(x, y, lamb):
    z = 0
    for i in range(len(x)):
        z += lamb[i] * l1_criterion(x[i], y[i])
    return z

def gauss_noise(x, mean=0, var=0.1, use_cuda=True, device=device):
    sz = x.shape
    noise = torch.normal(mean, var, size=(sz[0], sz[1], sz[2], sz[3]))
    if use_cuda:
        noise = noise.to(device)
    x = x + noise
    return x

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
                                 input_height=None, output_height=192, crop_height=192,
                                 is_random_crop=True, is_mirror=True, is_flip=True, is_gray=False, 
                                 upscale=args.mag, is_scale_back=False)    
trainloader = DataLoader(train_set, batch_size=args.trainbatch, shuffle=True, num_workers=args.workers)
    
test_list, _ = loadFromFile(args.testlist, args.test_num)
test_set = ImageDatasetFromFile(test_list, args.testdata, 
                                input_height=None, output_height=192, crop_height=192,
                                is_random_crop=True, is_mirror=False, is_flip=False, is_gray=False, 
                                upscale=args.mag, is_scale_back=False)    
testloader = DataLoader(test_set, batch_size=args.testbatch, shuffle=False, num_workers=args.workers)

beta = 1
miu = 10
lamb = [0.8, 1.0, 1.2]

#=======================================================================================================================================
if args.test_only is True :
    net_rec = wae_rec()
    testsrnet = srnet(mag=args.mag)
    
    if use_cuda and args.ngpu > 1:
        net_rec = nn.DataParallel(net_rec, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        net_rec.module.load_state_dict(torch.load(args.pretrained_AE))
        net_rec = net_rec.to(device)
            
        testsrnet = nn.DataParallel(testsrnet, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        testsrnet.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        testsrnet = testsrnet.to(device)
        
    elif (args.ngpu <= 1) or (use_cuda is False):
        net_rec.load_state_dict(torch.load(args.pretrained_AE))
        net_rec = net_rec.to(device)
        
        testsrnet.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        testsrnet = testsrnet.to(device)
        
    net_rec.eval()
    testsrnet.eval()
    with torch.no_grad():
        psnr_sum = 0
        ssim_sum = 0
        time_sum = 0
        for testiter, testdata in enumerate(testloader):
            tlowtarget, ttargets = testdata[0], testdata[1]
            if use_cuda:
                tlowtarget = tlowtarget.to(device)
                ttargets = ttargets.to(device)

            tinputs = torch.cat((tlowtarget, gauss_noise(tlowtarget, use_cuda=use_cuda, device=device)), 1)

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

            psnr = cal_psnr(ttargets, trec_pred, gray_scale=True)
            psnr_sum += psnr
            ssim = get_ssim(ttargets, trec_pred, data_range=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, grayscale=True, device=device)
            ssim_sum += ssim.item()

        tinfo = "Average time: {:.2f} s, Average PSNR: {:.2f} dB, Average SSIM: {:.4f}".format(time_sum/len(testloader), psnr_sum/len(testloader), ssim_sum/len(testloader))
        print(tinfo)
        
#=======================================================================================================================================
elif args.test_only is False :
    net_dec = wae_dec()
    net_rec = wae_rec()
    net = srnet(mag=args.mag)
    
    if use_cuda and args.ngpu > 1:
        net_dec = nn.DataParallel(net_dec, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        net_dec.module.load_state_dict(torch.load(args.pretrained_AE))
        net_dec = net_dec.to(device)
        for param in net_dec.parameters():
            param.requires_grad = False
        net_rec = nn.DataParallel(net_rec, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        net_rec.module.load_state_dict(torch.load(args.pretrained_AE))
        net_rec = net_rec.to(device)
        for param in net_rec.parameters():
            param.requires_grad = False
    elif (args.ngpu <= 1) or (use_cuda is False):
        net_dec.load_state_dict(torch.load(args.pretrained_AE))
        net_dec = net_dec.to(device)
        for param in net_dec.parameters():
            param.requires_grad = False
        net_rec.load_state_dict(torch.load(args.pretrained_AE))
        net_rec = net_rec.to(device)
        for param in net_rec.parameters():
            param.requires_grad = False
    if use_cuda and args.ngpu > 1:
        net = nn.DataParallel(net, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)

    net.train()
    start_time = time.time()
    for epoch in range(args.nepoch):
        loss_sum = 0
        for iteration, data in enumerate(trainloader):
        #---------------------train--------------------------
            lowtarget, targets = data[0], data[1].requires_grad_(requires_grad=False)
            if use_cuda:
                lowtarget = lowtarget.to(device)
                targets = targets.to(device)

            if args.mag == 1:
                inputc = targets
            elif args.mag == 2:
                inputc, inputd = net_dec(targets)
                inputd = [inputd]
            elif args.mag == 4:
                wae1c, wae1d = net_dec(targets)
                inputc, wae2d = net_dec(wae1c)
                inputd = [wae2d, wae1d]
            elif args.mag == 8:
                wae1c, wae1d = net_dec(targets)
                wae2c, wae2d = net_dec(wae1c)
                inputc, wae3d = net_dec(wae2c)
                inputd = [wae3d, wae2d, wae1d]

            inputs = torch.cat((lowtarget, gauss_noise(lowtarget, use_cuda=use_cuda, device=device)), 1)

            optimizer.zero_grad()
            wpoutc, wpoutd = net(inputs)

            if args.mag == 1:
                rec_pred = wpoutc
            elif args.mag == 2:
                rec_pred = net_rec(yc_rec=wpoutc, yd_rec=wpoutd[0])
            elif args.mag == 4:
                rec1 = net_rec(yc_rec=wpoutc, yd_rec=wpoutd[0])
                rec_pred = net_rec(yc_rec=rec1, yd_rec=wpoutd[1])
            elif args.mag == 8:
                rec1 = net_rec(yc_rec=wpoutc, yd_rec=wpoutd[0])
                rec2 = net_rec(yc_rec=rec1, yd_rec=wpoutd[1])
                rec_pred = net_rec(yc_rec=rec2, yd_rec=wpoutd[2])

            loss_full = l1_criterion(targets, rec_pred)
            loss_wav = l1_criterion(inputc, wpoutc) + loss_wavelet(inputd, wpoutd, lamb)
            loss = beta * loss_full + miu * loss_wav

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if iteration % 50 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "full: {:.5f}, wav: {:.5f}, iter_loss: {:.5f} ".format(loss_full.item(), loss_wav.item(), loss.item())
                print(info)

        print("Epoch : [{}/{}], loss = {:.5f}".format(epoch + 1, args.nepoch, loss_sum/len(trainloader)))

        #---------------------test-------------------------- 
        if epoch % 4 == 0 or epoch == (args.nepoch - 1):
            psnr_sum = 0
            ssim_sum = 0
            net.eval()
            for testiter, testdata in enumerate(testloader):
                tlowtarget, ttargets = testdata[0], testdata[1]
                if use_cuda:
                    tlowtarget = tlowtarget.to(device)
                    ttargets = ttargets.to(device)

                tinputs = torch.cat((tlowtarget, gauss_noise(tlowtarget, use_cuda=use_cuda, device=device)), 1)
                twpoutc, twpoutd = net(tinputs)
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
