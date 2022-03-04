
import time
import math
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

#=======================================================================================================================================
parser = argparse.ArgumentParser(description='VGG16 for ImageNet')
parser.add_argument('--nepoch', default=20, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
parser.add_argument('--lr_schedule', default=10, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.1, type=float, help='Decay rate of learning rate')
parser.add_argument('--trainbatch', default=256, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=128, type=int, help='Test batch size')
parser.add_argument('--traindata', default='/home/sjhan/datasets/imagenet/ILSVRC2012_img_train_untar', help='Path for train dataset')
parser.add_argument('--testdata', default='/home/sjhan/datasets/imagenet/ILSVRC2012_img_val_untar', help='Path for validation dataset')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='vgg_of_imagenet.pth', help='Name for trained model')
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
class vggnet16(nn.Module):
    def __init__(self, ch1=64, lastsize=7):
        super(vggnet16, self).__init__()
        
        self.convlayer = nn.Sequential(
            nn.Conv2d(3, ch1, 3, stride=1, padding=1), nn.BatchNorm2d(ch1), nn.ReLU(),
            nn.Conv2d(ch1, ch1, 3, stride=1, padding=1), nn.BatchNorm2d(ch1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(ch1, 2*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(2*ch1), nn.ReLU(),
            nn.Conv2d(2*ch1, 2*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(2*ch1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(2*ch1, 4*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(4*ch1), nn.ReLU(),
            nn.Conv2d(4*ch1, 4*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(4*ch1), nn.ReLU(),
            nn.Conv2d(4*ch1, 4*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(4*ch1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(4*ch1, 8*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch1), nn.ReLU(),
            nn.Conv2d(8*ch1, 8*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch1), nn.ReLU(),
            nn.Conv2d(8*ch1, 8*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(8*ch1, 8*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch1), nn.ReLU(),
            nn.Conv2d(8*ch1, 8*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch1), nn.ReLU(),
            nn.Conv2d(8*ch1, 8*ch1, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch1), nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(8*ch1 * lastsize * lastsize, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 1000))
        
    def forward(self, x):
        fm = self.convlayer(x)
        
        fmview = fm.view(fm.size(0), -1)
        s = self.fc(fmview)
            
        return s
    
#=======================================================================================================================================
def loss_MSE(x, y):
    z = torch.mean((x - y)**2)
    return z
    
def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_reset(batchSize, dataroot, workers=args.workers, resize=True, resizesz=256, rancrop=True, rancropsz=224, horiflip=True):
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

def test_reset(batchSize, dataroot, workers=args.workers, resize=True, resizesz=224, rancrop=False, rancropsz=224):
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
criterion = nn.CrossEntropyLoss().to(device)
testloader = test_reset(args.testbatch, args.testdata, workers=args.workers, resize=True, resizesz=224, rancrop=False, rancropsz=192)

#=======================================================================================================================================
if args.test_only is True :
    vggnet = vggnet16(ch1=64, lastsize=7)
    if use_cuda and args.ngpu > 1:
        vggnet = nn.DataParallel(vggnet, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        vggnet.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        vggnet = vggnet.to(device)
    elif (args.ngpu <= 1) or (use_cuda is False):
        vggnet.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        vggnet = vggnet.to(device)
        
    vggnet.eval()
    with torch.no_grad():
        testcorrect_5 = 0
        testloss_sum = 0
        testtotal = 0
        time_sum = 0
        for testiter, testdata in enumerate(testloader):
            tinput, tlabels = testdata
            testtotal += tlabels.size(0)
            if use_cuda:
                tinput = tinput.to(device)
                tlabels = tlabels.to(device) 

            start_time = time.time()
            ts = vggnet(tinput)
            iter_time = time.time()-start_time
            time_sum += iter_time

            testloss = criterion(ts, tlabels)

            testloss_sum += testloss.item()
            _t, testtop5ind = torch.topk(ts, 5, dim=1)
            tlab = tlabels.view(tlabels.size(0),-1).expand_as(testtop5ind)
            testcorrect_5 += torch.sum(testtop5ind.eq(tlab).float()).item()

        tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
        tinfo += "test_top5_accuracy: {:.4f}, avg_time: {:.2f}".format(100 * testcorrect_5 / testtotal, time_sum/len(testloader))
        print(tinfo)
        
#=======================================================================================================================================
elif args.test_only is False :
    net_vgg = vggnet16(ch1=64, lastsize=7)
    net_vgg.apply(init_weights)
    if use_cuda and args.ngpu > 1:
        net_vgg = nn.DataParallel(net_vgg, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net_vgg = net_vgg.to(device)

    optimizer = optim.SGD(net_vgg.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)

    net_vgg.train()
    start_time = time.time()
    for epoch in range(args.nepoch): 
        correct_5 = 0
        loss_sum = 0
        total=0
        trainloader = train_reset(args.trainbatch, args.traindata, workers=args.workers, resize=True, resizesz=256, rancrop=True, rancropsz=224, horiflip=True)
        for iteration, data in enumerate(trainloader):
        #--------------train-------------
            inputs, labels = data
            total += labels.size(0)
            if use_cuda:
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            s = net_vgg(inputs)

            loss = criterion(s, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            __, top5ind = torch.topk(s, 5, dim=1)
            lab = labels.view(labels.size(0),-1).expand_as(top5ind)
            correct_5 += torch.sum(top5ind.eq(lab).float()).item()

            if iteration % 100 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:4.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "iter_loss: {:.4f} ".format(loss.item())
                print(info)

        print("trainloss: {:.4f}, train_top5_accuracy: {:.4f}\n".format(loss_sum/len(trainloader), 100 * correct_5 / total))

        #--------------test-------------
        if epoch % 2 == 0 or epoch == (args.nepoch - 1):
            net_vgg.eval()

            testcorrect_5 = 0
            testloss_sum = 0
            testtotal = 0
            for testiter, testdata in enumerate(testloader):
                tinput, tlabels = testdata
                testtotal += tlabels.size(0)
                if use_cuda:
                    tinput = tinput.to(device)
                    tlabels = tlabels.to(device)

                ts = net_vgg(tinput)

                testloss = criterion(ts, tlabels)

                testloss_sum += testloss.item()
                _t, testtop5ind = torch.topk(ts, 5, dim=1)
                tlab = tlabels.view(tlabels.size(0),-1).expand_as(testtop5ind)
                testcorrect_5 += torch.sum(testtop5ind.eq(tlab).float()).item()

            tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
            tinfo += "test_top5_accuracy: {:.4f}".format(100 * testcorrect_5 / testtotal)
            print(tinfo)

            net_vgg.train()
        scheduler.step()

    print('train over')

#=======================================================================================================================================
    if args.ngpu <= 1 or use_cuda is False:
        torch.save(net_vgg.state_dict(), '/'.join([args.model_save, args.model_name]))
    elif use_cuda and args.ngpu > 1:
        torch.save(net_vgg.module.state_dict(), '/'.join([args.model_save, args.model_name]))

#=======================================================================================================================================
