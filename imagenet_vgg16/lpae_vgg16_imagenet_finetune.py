
import time
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
parser = argparse.ArgumentParser(description='LPAE + VGG16 finetune for ImageNet')
parser.add_argument('--nepoch', default=10, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate')
parser.add_argument('--trainbatch', default=256, type=int, help='Train batch size')
parser.add_argument('--testbatch', default=128, type=int, help='Test batch size')
parser.add_argument('--traindata', default='/home/sjhan/datasets/imagenet/ILSVRC2012_img_train_untar', help='Path for train dataset')
parser.add_argument('--testdata', default='/home/sjhan/datasets/imagenet/ILSVRC2012_img_val_untar', help='Path for validation dataset')

parser.add_argument('--model_save', default='./model_save', help='Path to save trained model')
parser.add_argument('--model_name', default='vgg_of_change_structure_rec_l1_imagenet_combine_finetune.pth', help='Name for trained model')
parser.add_argument('--pretrained_AE', default='./model_save/wae_change_structure_imagenet_rec_l1.pth', help='pretrained LPAE parameters')
parser.add_argument('--pretrained_vgg', default='./model_save/vgg_of_change_structure_rec_l1_imagenet_combine.pth', help='pretrained VGG parameters')
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
    def __init__(self, ch1=64, ch2=16, lastsize=4):
        super(vggnet16, self).__init__()
        
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
        
        self.convlayer1 = nn.Sequential(
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
            nn.MaxPool2d(2, stride=2, padding=1))
            
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(3, ch2, 3, stride=1, padding=1), nn.BatchNorm2d(ch2), nn.ReLU(),
            nn.Conv2d(ch2, ch2, 3, stride=1, padding=1), nn.BatchNorm2d(ch2), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(ch2, 2*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(2*ch2), nn.ReLU(),
            nn.Conv2d(2*ch2, 2*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(2*ch2), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(2*ch2, 4*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(4*ch2), nn.ReLU(),
            nn.Conv2d(4*ch2, 4*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(4*ch2), nn.ReLU(),
            nn.Conv2d(4*ch2, 4*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(4*ch2), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(4*ch2, 8*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch2), nn.ReLU(),
            nn.Conv2d(8*ch2, 8*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch2), nn.ReLU(),
            nn.Conv2d(8*ch2, 8*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch2), nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(8*ch2, 8*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch2), nn.ReLU(),
            nn.Conv2d(8*ch2, 8*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch2), nn.ReLU(),
            nn.Conv2d(8*ch2, 8*ch2, 3, stride=1, padding=1), nn.BatchNorm2d(8*ch2), nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        
        self.fc1 = nn.Sequential(
            nn.Linear(8*ch1 * lastsize * lastsize, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 1000))
        
        lastsizeH = 2*lastsize - 1
        self.fc2 = nn.Sequential(
            nn.Linear(8*ch1 * lastsize * lastsize + 8*ch2 * lastsizeH * lastsizeH, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 1000))
        
    def forward(self, x):
        
        c = self.layer1_1(x)      
        d = self.layer1_2(x)
        c_up = self.layer2_1(c)
        out =  d + c_up
        
        fL = self.convlayer1(c)
        fH = self.convlayer2(d)
        
        fLview = fL.view(fL.size(0), -1)
        fHview = fH.view(fH.size(0), -1)
        C = torch.cat((fLview, fHview), dim=1)
        
        sL = self.fc1(fLview)
        sC = self.fc2(C)
        s = (sL + sC)/2
            
        return sL, sC, s, c, d, out

#=======================================================================================================================================
def loss_MSE(x, y):
    z = torch.mean((x - y)**2)
    return z

def loss_l1(x, y):
    z2 = torch.abs((x-y))
    z2 = torch.mean(z2)
    return z2
    
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
gam = 0.001

#=======================================================================================================================================
if args.test_only is True :
    vggnet = vggnet16(ch1=64, ch2=16, lastsize=4)
    if use_cuda and args.ngpu > 1:
        vggnet = nn.DataParallel(vggnet, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
        vggnet.module.load_state_dict(torch.load('/'.join([args.model_save, args.model_name])))
        vggnet = vggnet.to(device)
    elif args.ngpu <= 1 or use_cuda is False:
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
            test_input_down = F.interpolate(tinput, (112, 112), mode='bicubic', align_corners=True)

            start_time = time.time()
            tsL, tsC, ts, test_c, test_d, testout = vggnet(tinput)
            iter_time = time.time()-start_time
            time_sum += iter_time

            testloss_wae = loss_l1(tinput, testout) + loss_MSE(test_d, 0) + 0.8 * loss_MSE(test_input_down, test_c)

            testloss1 = criterion(tsL, tlabels)
            testloss2 = criterion(tsC, tlabels)
            testloss_vgg = testloss1 + testloss2

            testloss = testloss_vgg + gam * testloss_wae

            testloss_sum += testloss.item()
            _t, testtop5ind = torch.topk(ts, 5, dim=1)
            tlab = tlabels.view(tlabels.size(0),-1).expand_as(testtop5ind)
            testcorrect_5 += torch.sum(testtop5ind.eq(tlab).float()).item()

        tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
        tinfo += "test_top5_accuracy: {:.4f}, avg_time: {:.2f}".format(100 * testcorrect_5 / testtotal, time_sum/len(testloader))
        print(tinfo)
        
#=======================================================================================================================================
elif args.test_only is False :
    net_vgg = vggnet16(ch1=64, ch2=16, lastsize=4)

    pretrained_wae = torch.load(args.pretrained_AE)
    pretrained_vgg = torch.load(args.pretrained_vgg)
    net_vgg_dict = net_vgg.state_dict()

    pretrained_wae_dict = {k: v for k, v in pretrained_wae.items() if k in net_vgg_dict}
    pretrained_vgg_dict = {k: v for k, v in pretrained_vgg.items() if k in net_vgg_dict}
    net_vgg_dict.update(pretrained_wae_dict)
    net_vgg_dict.update(pretrained_vgg_dict)
    net_vgg.load_state_dict(net_vgg_dict)
    
    if use_cuda and args.ngpu > 1:
        net_vgg = nn.DataParallel(net_vgg, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
    net_vgg = net_vgg.to(device)

    optimizer = optim.SGD(net_vgg.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

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
            input_down = F.interpolate(inputs, (112, 112), mode='bicubic', align_corners=True)

            optimizer.zero_grad()
            sL, sC, s, input_c, input_d, inputout = net_vgg(inputs)

            loss_wae = loss_l1(inputs, inputout) + loss_MSE(input_d, 0) + 0.8 * loss_MSE(input_down, input_c)

            loss1 = criterion(sL, labels)
            loss2 = criterion(sC, labels)
            loss_vgg = loss1 + loss2

            loss = loss_vgg + gam * loss_wae

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            __, top5ind = torch.topk(s, 5, dim=1)
            lab = labels.view(labels.size(0),-1).expand_as(top5ind)
            correct_5 += torch.sum(top5ind.eq(lab).float()).item()

            if iteration % 100 == 0:
                info = "===> Epoch[{}]({}/{}): time: {:4.4f} ".format(epoch, iteration, len(trainloader), time.time()-start_time)
                info += "loss_vgg: {:.4f}, loss_wae: {:.4f}, tot_loss: {:.4f} ".format(loss_vgg.item(), loss_wae.item(), loss.item())
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
                test_input_down = F.interpolate(tinput, (112, 112), mode='bicubic', align_corners=True)

                tsL, tsC, ts, test_c, test_d, testout = net_vgg(tinput)

                testloss_wae = loss_l1(tinput, testout) + loss_MSE(test_d, 0) + 0.8 * loss_MSE(test_input_down, test_c)

                testloss1 = criterion(tsL, tlabels)
                testloss2 = criterion(tsC, tlabels)
                testloss_vgg = testloss1 + testloss2

                testloss = testloss_vgg + gam * testloss_wae

                testloss_sum += testloss.item()
                _t, testtop5ind = torch.topk(ts, 5, dim=1)
                tlab = tlabels.view(tlabels.size(0),-1).expand_as(testtop5ind)
                testcorrect_5 += torch.sum(testtop5ind.eq(tlab).float()).item()

            tinfo = "testloss: {:.4f} ".format(testloss_sum/len(testloader))
            tinfo += "test_top5_accuracy: {:.4f}".format(100 * testcorrect_5 / testtotal)
            print(tinfo)

            net_vgg.train()

    print('train over')

#=======================================================================================================================================
    if args.ngpu <= 1 or use_cuda is False:
        torch.save(net_vgg.state_dict(), '/'.join([args.model_save, args.model_name]))
    elif use_cuda and args.ngpu > 1:
        torch.save(net_vgg.module.state_dict(), '/'.join([args.model_save, args.model_name]))

#=======================================================================================================================================
