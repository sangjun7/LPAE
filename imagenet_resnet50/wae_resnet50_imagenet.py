
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
parser = argparse.ArgumentParser(description='WAE + ResNet50 for ImageNet')
parser.add_argument('--nepoch', default=20, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')
parser.add_argument('--lr_schedule', default=10, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Decay rate of learning rate')
parser.add_argument('--trainbatch', default=256, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=128, type=int, help='Test batch size')
parser.add_argument('--traindata', default='/home/sjhan/datasets/imagenet/ILSVRC2012_img_train_untar', help='Path for train dataset')
parser.add_argument('--testdata', default='/home/sjhan/datasets/imagenet/ILSVRC2012_img_val_untar', help='Path for validation dataset')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='resnet50_of_papersetting_imagenet_combine.pth', help='Name for trained model')
parser.add_argument('--pretrained_AE', default='./model_save/wae_papersetting_imagenet.pth', help='pretrained WAE parameters')
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
    def __init__(self, ch1=64, ch2=16, lastsize=4):
        super(resnet50, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1))
        
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
            
        self.layer2_2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, stride=2, padding=1))
        
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(3, ch1, 7, stride=2, padding=3), nn.MaxPool2d(3, stride=2, padding=1),
            resblock(ch1, ch1, 4*ch1, down=False),
            resblock(4*ch1, ch1, 4*ch1, down=False),
            resblock(4*ch1, ch1, 4*ch1, down=False),
            resblock(4*ch1, 2*ch1, 8*ch1, down=True),
            resblock(8*ch1, 2*ch1, 8*ch1, down=False),
            resblock(8*ch1, 2*ch1, 8*ch1, down=False),
            resblock(8*ch1, 2*ch1, 8*ch1, down=False),
            resblock(8*ch1, 4*ch1, 16*ch1, down=True),
            resblock(16*ch1, 4*ch1, 16*ch1, down=False),
            resblock(16*ch1, 4*ch1, 16*ch1, down=False),
            resblock(16*ch1, 4*ch1, 16*ch1, down=False),
            resblock(16*ch1, 4*ch1, 16*ch1, down=False),
            resblock(16*ch1, 4*ch1, 16*ch1, down=False),
            resblock(16*ch1, 8*ch1, 32*ch1, down=True),
            resblock(32*ch1, 8*ch1, 32*ch1, down=False),
            resblock(32*ch1, 8*ch1, 32*ch1, down=False))
            
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(3, ch2, 7, stride=2, padding=3), nn.MaxPool2d(3, stride=2, padding=1),
            resblock(ch2, ch2, 4*ch2, down=False),
            resblock(4*ch2, ch2, 4*ch2, down=False),
            resblock(4*ch2, ch2, 4*ch2, down=False),
            resblock(4*ch2, 2*ch2, 8*ch2, down=True),
            resblock(8*ch2, 2*ch2, 8*ch2, down=False),
            resblock(8*ch2, 2*ch2, 8*ch2, down=False),
            resblock(8*ch2, 2*ch2, 8*ch2, down=False),
            resblock(8*ch2, 4*ch2, 16*ch2, down=True),
            resblock(16*ch2, 4*ch2, 16*ch2, down=False),
            resblock(16*ch2, 4*ch2, 16*ch2, down=False),
            resblock(16*ch2, 4*ch2, 16*ch2, down=False),
            resblock(16*ch2, 4*ch2, 16*ch2, down=False),
            resblock(16*ch2, 4*ch2, 16*ch2, down=False),
            resblock(16*ch2, 8*ch2, 32*ch2, down=True),
            resblock(32*ch2, 8*ch2, 32*ch2, down=False),
            resblock(32*ch2, 8*ch2, 32*ch2, down=False))
        
        self.ap = nn.AvgPool2d((lastsize,lastsize))
        
        self.fc1 = nn.Linear(32*ch1, 1000)
        self.fc2 = nn.Linear(32*ch1 + 32*ch2, 1000)
        
    def forward(self, x):
        
        x = self.layer1(x)      
        c = self.layer2_1(x)
        d = self.layer2_2(x)
        
        fL = self.convlayer1(c)
        fH = self.convlayer2(d)
        fLavg = self.ap(fL)
        fHavg = self.ap(fH)
        
        fLview = fLavg.view(fLavg.size(0), -1)
        fHview = fHavg.view(fHavg.size(0), -1)
        C = torch.cat((fLview, fHview), dim=1)
        
        sL = self.fc1(fLview)
        sC = self.fc2(C)
        s = (sL + sC)/2
            
        return sL, sC, s

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
    res = resnet50(ch1=64, ch2=16, lastsize=4)
    if use_cuda and args.ngpu > 1:
        res = nn.DataParallel(res, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        res.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        res = res.to(device)
    elif (args.ngpu <= 1) or (use_cuda is False):
        res.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        res = res.to(device)
    
    res.eval()
    with torch.no_grad():
        testcorrect_5 = 0
        testloss_sum = 0
        testtotal = 0
        for testiter, testdata in enumerate(testloader):
            tinput, tlabels = testdata
            testtotal += tlabels.size(0)
            if use_cuda:
                tinput = tinput.to(device)
                tlabels = tlabels.to(device)  

            tsL, tsC, ts = res(tinput)

            testloss1 = criterion(tsL, tlabels)
            testloss2 = criterion(tsC, tlabels)
            testloss = testloss1 + testloss2

            testloss_sum += testloss.item()
            _t, testtop5ind = torch.topk(ts, 5, dim=1)
            tlab = tlabels.view(tlabels.size(0),-1).expand_as(testtop5ind)
            testcorrect_5 += torch.sum(testtop5ind.eq(tlab).float()).item()

        tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
        tinfo += "test_top5_accuracy: {:.4f}".format(100 * testcorrect_5 / testtotal)
        print(tinfo)
    
#=======================================================================================================================================
elif args.test_only is False : 
    net_res = resnet50(ch1=64, ch2=16, lastsize=4)
    net_res.apply(init_weights)

    pretrained_wae = torch.load(args.pretrained_AE)
    net_res_dict = net_res.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_wae.items() if k in net_res_dict}
    net_res_dict.update(pretrained_dict)
    net_res.load_state_dict(net_res_dict)

    for k, v in net_res.named_parameters():
        if k in pretrained_wae:
            v.requires_grad = False
#         print(k, v.requires_grad, v.device)

    if use_cuda and args.ngpu > 1:
        net_res = nn.DataParallel(net_res, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net_res = net_res.to(device)

    optimizer = optim.Adam(net_res.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)

    net_res.train()
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
            sL, sC, s = net_res(inputs)

            loss1 = criterion(sL, labels)
            loss2 = criterion(sC, labels)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            __, top5ind = torch.topk(s, 5, dim=1)
            lab = labels.view(labels.size(0),-1).expand_as(top5ind)
            correct_5 += torch.sum(top5ind.eq(lab).float()).item()

            if iteration % 100 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:4.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "loss_L: {:.4f}, loss_C: {:.4f}, tot_loss: {:.4f} ".format(loss1.item(), loss2.item(), loss.item())
                print(info)

        print("trainloss: {:.4f}, train_top5_accuracy: {:.4f}\n".format(loss_sum/len(trainloader), 100 * correct_5 / total))

        #--------------test-------------
        if epoch % 4 == 0 or epoch == (args.nepoch - 1):
            net_res.eval()

            testcorrect_5 = 0
            testloss_sum = 0
            testtotal = 0
            for testiter, testdata in enumerate(testloader):
                tinput, tlabels = testdata
                testtotal += tlabels.size(0)
                if use_cuda:
                    tinput = tinput.to(device)
                    tlabels = tlabels.to(device)

                tsL, tsC, ts = net_res(tinput)

                testloss1 = criterion(tsL, tlabels)
                testloss2 = criterion(tsC, tlabels)
                testloss = testloss1 + testloss2

                testloss_sum += testloss.item()
                _t, testtop5ind = torch.topk(ts, 5, dim=1)
                tlab = tlabels.view(tlabels.size(0),-1).expand_as(testtop5ind)
                testcorrect_5 += torch.sum(testtop5ind.eq(tlab).float()).item()

            tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
            tinfo += "test_top5_accuracy: {:.4f}".format(100 * testcorrect_5 / testtotal)
            print(tinfo)

            net_res.train()
        scheduler.step()

    print('train over')

#=======================================================================================================================================
    if args.ngpu <= 1 or use_cuda is False:
        torch.save(net_res.state_dict(), '/'.join([args.model_save, args.model_name]))
    elif use_cuda and args.ngpu > 1:
        torch.save(net_res.module.state_dict(), '/'.join([args.model_save, args.model_name]))

#=======================================================================================================================================
