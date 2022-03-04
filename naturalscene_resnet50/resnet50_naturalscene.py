
import time
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
parser = argparse.ArgumentParser(description='ResNet50 for NaturalScene')
parser.add_argument('--nepoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
parser.add_argument('--lr_schedule', default=10, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Decay rate of learning rate')
parser.add_argument('--trainbatch', default=256, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=256, type=int, help='Test batch size')
parser.add_argument('--traindata', default='/home/sjhan/datasets/naturalscene/seg_train/seg_train', help='Path for train dataset')
parser.add_argument('--testdata', default='/home/sjhan/datasets/naturalscene/seg_test/seg_test', help='Path for test dataset')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='resnet50_of_naturalscene.pth', help='Name for trained model')
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
class resblock(nn.Module):
    def __init__(self, ich, mch, och, down=True):
        super(resblock, self).__init__()
        
        if down:
            self.convlayer = nn.Sequential(
                nn.Conv2d(ich, mch, 1, stride=2, padding=0), nn.BatchNorm2d(mch), nn.ReLU(),
                nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.BatchNorm2d(mch), nn.ReLU(),
                nn.Conv2d(mch, och, 1, stride=1, padding=0))
            self.skip = nn.Conv2d(ich, och, 1, stride=2, padding=0)
        else:
            self.convlayer = nn.Sequential(
                nn.Conv2d(ich, mch, 1, stride=1, padding=0), nn.BatchNorm2d(mch), nn.ReLU(),
                nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.BatchNorm2d(mch), nn.ReLU(),
                nn.Conv2d(mch, och, 1, stride=1, padding=0))
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
        x_add = torch.add(x1, x2)
        out = self.act(self.bn(x_add))
        
        return out
    
#=======================================================================================================================================
class resnet50(nn.Module):
    def __init__(self, ch=64, lastsize=4):
        super(resnet50, self).__init__()
        
        self.conv1 = nn.Conv2d(3, ch, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.block1 = nn.Sequential(
            resblock(ch, ch, 4*ch, down=False),
            resblock(4*ch, ch, 4*ch, down=False),
            resblock(4*ch, ch, 4*ch, down=False))
        self.block2 = nn.Sequential(
            resblock(4*ch, 2*ch, 8*ch, down=True),
            resblock(8*ch, 2*ch, 8*ch, down=False),
            resblock(8*ch, 2*ch, 8*ch, down=False),
            resblock(8*ch, 2*ch, 8*ch, down=False))
        self.block3 = nn.Sequential(
            resblock(8*ch, 4*ch, 16*ch, down=True),
            resblock(16*ch, 4*ch, 16*ch, down=False),
            resblock(16*ch, 4*ch, 16*ch, down=False),
            resblock(16*ch, 4*ch, 16*ch, down=False),
            resblock(16*ch, 4*ch, 16*ch, down=False),
            resblock(16*ch, 4*ch, 16*ch, down=False))
        self.block4 = nn.Sequential(
            resblock(16*ch, 8*ch, 32*ch, down=True),
            resblock(32*ch, 8*ch, 32*ch, down=False),
            resblock(32*ch, 8*ch, 32*ch, down=False))
        
        self.ap = nn.AvgPool2d((lastsize,lastsize))
        self.fc = nn.Linear(32*ch, 6)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool1(conv1)
        block1 = self.block1(conv1_pool)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        ap = self.ap(block4)
        ap_flat = ap.view(ap.size(0), -1)
        s = self.fc(ap_flat)
        
        return s

#=======================================================================================================================================
def loss_MSE(x, y):
    z = torch.mean((x - y)**2)
    return z
    
def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train_reset(batchSize, dataroot, workers=args.workers, resize=True, resizesz=150, rancrop=True, rancropsz=128, horiflip=True):
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

def test_reset(batchSize, dataroot, workers=args.workers, resize=True, resizesz=128, rancrop=False, rancropsz=128):
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
testloader = test_reset(args.testbatch, args.testdata, workers=args.workers, resize=True, resizesz=128, rancrop=False, rancropsz=128)

#=======================================================================================================================================
if args.test_only is True :
    resnet = resnet50(ch=64, lastsize=4)
    if use_cuda and args.ngpu > 1:
        resnet = nn.DataParallel(resnet, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        resnet.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        resnet = resnet.to(device)
    elif (args.ngpu <= 1) or (use_cuda is False):
        resnet.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        resnet = resnet.to(device)
    
    resnet.eval()
    with torch.no_grad():
        testcorrect = 0
        testloss_sum = 0
        testtotal = 0
        for testiter, testdata in enumerate(testloader):
            tinput, tlabels = testdata
            testtotal += tlabels.size(0)
            if use_cuda:
                tinput = tinput.to(device)
                tlabels = tlabels.to(device) 

            ts = resnet(tinput)

            testloss = criterion(ts, tlabels)

            testloss_sum += testloss.item()
            _t, testpred = torch.max(ts, dim=1)
            testcorrect += torch.sum(testpred==tlabels).item()

        tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
        tinfo += "test_accuracy: {:.4f}".format(100 * testcorrect / testtotal)
        print(tinfo)
        
#=======================================================================================================================================
elif args.test_only is False : 
    net = resnet50(ch=64, lastsize=4)
    net.apply(init_weights)
    if use_cuda and args.ngpu > 1:
        net = nn.DataParallel(net, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)

    net.train()
    start_time = time.time()
    for epoch in range(args.nepoch): 
        correct = 0
        loss_sum = 0
        total=0
        trainloader = train_reset(args.trainbatch, args.traindata, workers=args.workers, resize=True, resizesz=150, rancrop=True, rancropsz=128, horiflip=True)
        for iteration, data in enumerate(trainloader):
        #--------------train-------------
            inputs, labels = data
            total += labels.size(0)
            if use_cuda:
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            s = net(inputs)

            loss = criterion(s, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _,pred = torch.max(s, dim=1)
            correct += torch.sum(pred==labels).item()

            if iteration % 50 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:4.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "iter_loss: {:.4f} ".format(loss.item())
                print(info)

        print("trainloss: {:.4f}, train_accuracy: {:.4f}\n".format(loss_sum/len(trainloader), 100 * correct / total))

        #--------------test-------------
        if epoch % 4 == 0 or epoch == (args.nepoch - 1):
            net.eval()

            testcorrect = 0
            testloss_sum = 0
            testtotal = 0
            for testiter, testdata in enumerate(testloader):
                tinput, tlabels = testdata
                testtotal += tlabels.size(0)
                if use_cuda:
                    tinput = tinput.to(device)
                    tlabels = tlabels.to(device)

                ts = net(tinput)

                testloss = criterion(ts, tlabels)

                testloss_sum += testloss.item()
                _t, testpred = torch.max(ts, dim=1)
                testcorrect += torch.sum(testpred==tlabels).item()

            tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
            tinfo += "test_accuracy: {:.4f}".format(100 * testcorrect / testtotal)
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
