
import time
import math
import argparse
import numpy as np
from PIL import Image
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

#=======================================================================================================================================
parser = argparse.ArgumentParser(description='LPAE for NaturalScene')
parser.add_argument('--nepoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
parser.add_argument('--lr_schedule', default=10, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.1, type=float, help='Decay rate of learning rate')
parser.add_argument('--trainbatch', default=4, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=1, type=int, help='Test batch size')
parser.add_argument('--traindata', default='/home/sjhan/datasets/naturalscene/seg_train/seg_train', help='Path for train dataset')
parser.add_argument('--testdata', default='/home/sjhan/datasets/naturalscene/seg_test/seg_test', help='Path for test dataset')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='wae_change_structure_naturalscene_rec_l1.pth', help='Name for trained model')
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
class en_wav(nn.Module):
    def __init__(self):
        super(en_wav, self).__init__()
        
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
        x_c_up = self.layer2_1(x_c)
        out =  x_d + x_c_up
            
        return x_c, x_d, out

#=======================================================================================================================================
def loss_MSE(x, y):
    z = torch.mean((x - y)**2)
    return z
    
def loss_l1(x, y):
    z2 = torch.abs((x-y))
    z2 = torch.mean(z2)
    return z2

def init_weights(m):
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def train_reset(batchSize, dataroot, workers=args.workers, resize=True, resizesz=150, rancrop=True, rancropsz=134, horiflip=True):
    trans_list = []
    if resize:
        trans_list.append(transforms.Resize((resizesz, resizesz), interpolation=Image.BICUBIC))
    if rancrop:
        trans_list.append(transforms.RandomCrop((rancropsz, rancropsz)))
    if horiflip:
        trans_list.append(transforms.RandomHorizontalFlip())
    trans_list.append(transforms.ToTensor())
    traintransform = transforms.Compose(trans_list)
    trainset = torchvision.datasets.ImageFolder(root=dataroot,
                                               transform=traintransform)
    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=workers)
    
    return trainloader

def test_reset(batchSize, dataroot, workers=args.workers, resize=True, resizesz=134, rancrop=False, rancropsz=134):
    trans_list=[]
    if resize:
        trans_list.append(transforms.Resize((resizesz, resizesz), interpolation=Image.BICUBIC))
    if rancrop:
        trans_list.append(transforms.RandomCrop((rancropsz, rancropsz)))
    trans_list.append(transforms.ToTensor())
    testtransform = transforms.Compose(trans_list)
    testset = torchvision.datasets.ImageFolder(root=dataroot,
                                               transform=testtransform)
    testloader = DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=workers)
    
    return testloader

#=======================================================================================================================================
testloader = test_reset(args.testbatch, args.testdata, workers=args.workers, resize=True, resizesz=134, rancrop=False, rancropsz=134)
lamb = 1
gamm = 0.8

#=======================================================================================================================================
if args.test_only is True :
    net_wae = en_wav()
    if use_cuda and args.ngpu > 1:
        net_wae = nn.DataParallel(net_wae, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        net_wae.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        net_wae = net_wae.to(device)
    elif (args.ngpu <= 1) or (use_cuda is False):
        net_wae.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        net_wae = net_wae.to(device)
    
    with torch.no_grad():
        testloss_sum = 0
        avg_psnr = 0
        for testiter, testdata in enumerate(testloader):
            tinput, tlabels = testdata
            if use_cuda:
                tinput = tinput.to(device)
            test_input_down = F.interpolate(tinput, scale_factor=(1/2, 1/2), mode='bicubic', align_corners=True)

            test_c, test_d, testout = net_wae(tinput)

            testloss = loss_l1(tinput, testout) + lamb * loss_MSE(test_d, 0) + gamm * loss_MSE(test_input_down, test_c)
            testloss_sum += testloss.item()

            mse = loss_MSE(tinput, testout)
            psnr = -10.0 * math.log10(mse.item())
            avg_psnr += psnr

        tinfo = "testloss: {:.5f}, avg_psnr: {:.4f}".format(testloss_sum/len(testloader), avg_psnr/len(testloader))
        print(tinfo)
    
#=======================================================================================================================================
elif args.test_only is False :
    net = en_wav()
    net.apply(init_weights)
    if use_cuda and args.ngpu > 1:
        net = nn.DataParallel(net, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)

    net.train()
    start_time = time.time()
    for epoch in range(args.nepoch):        
        loss_sum = 0
        trainloader = train_reset(args.trainbatch, args.traindata, workers=args.workers, resize=True, resizesz=150, rancrop=True, rancropsz=134, horiflip=True)
        for iteration, data in enumerate(trainloader):
        #--------------train-------------
            inputs, labels = data
            if use_cuda:
                inputs = inputs.to(device)
            input_down = F.interpolate(inputs, scale_factor=(1/2, 1/2), mode='bicubic', align_corners=True)

            optimizer.zero_grad()
            x_c, x_d, out = net(inputs)

            loss1 = loss_MSE(x_d, 0)
            loss2 = loss_MSE(input_down, x_c)
            loss3 = loss_l1(inputs, out)
            loss = loss3 + lamb * loss1 + gamm * loss2

            loss.backward() 
            optimizer.step()
            loss_sum += loss.item()

            if iteration % 500 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "detail_loss: {:.5f}, app_loss: {:.5f}, rec_l1_loss: {:.5f}, iter_loss: {:.5f} ".format(loss1.item(), loss2.item(), loss3.item(), loss.item())
                print(info)

        print("Epoch : [{}/{}], loss = {:.5f}".format(epoch + 1, args.nepoch, loss_sum/len(trainloader)))

        #--------------test-------------
        if epoch % 2 == 0 or epoch == (args.nepoch - 1):
            net.eval()
            testloss_sum = 0
            for testiter, testdata in enumerate(testloader):
                tinput, tlabels = testdata
                if use_cuda:
                    tinput = tinput.to(device)
                test_input_down = F.interpolate(tinput, scale_factor=(1/2, 1/2), mode='bicubic', align_corners=True)

                test_c, test_d, testout = net(tinput)

                testloss = loss_l1(tinput, testout) + lamb * loss_MSE(test_d, 0) + gamm * loss_MSE(test_input_down, test_c)
                testloss_sum += testloss.item()

            print("testloss: {:.5f} ".format(testloss_sum/len(testloader)))
            
            net.train()
        scheduler.step()

    print('train over')

#=======================================================================================================================================
    if args.ngpu <= 1 or use_cuda is False:
        torch.save(net.state_dict(), '/'.join([args.model_save, args.model_name]))
    elif use_cuda and args.ngpu > 1:
        torch.save(net.module.state_dict(), '/'.join([args.model_save, args.model_name]))

#=======================================================================================================================================
