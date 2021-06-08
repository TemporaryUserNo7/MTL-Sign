# -*- coding: utf-8 -*-
import copy
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.optim import Adam,SGD
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
#import gen_qr_keys
import time
import os
n=200.0
# Assign device.
device=torch.device("cuda:0")
class QRDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
             line.rstrip()
             words= line.split()
             imgs.append((words[0], int(words[1])))
        self.imgs = imgs 
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
	    fn, label = self.imgs[index]
	    img = Image.open(fn).convert('L') 
	    if self.transform is not None:
		    img = self.transform(img) 
	    return img, label
    def __len__(self):
	    return len(self.imgs)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        x=x.to(device)
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes,test_loader,qr_host_loader,qr_steal_loader,do=None):
        super(ResNet, self).__init__()
        self.test_loader=test_loader
        self.qr_host_loader=qr_host_loader
        self.qr_steal_loader=qr_steal_loader
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        if do==None:
            self.mac=nn.Sequential(
                nn.Linear(1408,256),
                nn.ReLU(),
                nn.Linear(256,32),
                nn.ReLU(),
                nn.Linear(32,2),
            )
        else:
            self.mac=nn.Sequential(
                nn.Dropout(p=do),
                nn.Linear(1408,256),
                nn.ReLU(),
                nn.Linear(256,32),
                nn.ReLU(),
                nn.Linear(32,2),
            )
        if do==None:
            self.backdoor=nn.Sequential(
                nn.Linear(1408,256),
                nn.ReLU(),
                nn.Linear(256,32),
                nn.ReLU(),
                nn.Linear(32,2),
            )
        else:
            self.backdoor=nn.Sequential(
                nn.Linear(1408,256),
                nn.ReLU(),
                nn.Linear(256,32),
                nn.ReLU(),
                nn.Linear(32,2),
            )
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x=x.to(device)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    def verify(self,x):
        x=x.to(device)
        out=self.conv1(x)
        out=self.layer1(out)
        out2=self.layer2(out)
        out3=self.layer3(out2)
        out2=F.avg_pool2d(out2,4)
        out2=out2.view(out2.size(0),-1)
        out3=F.avg_pool2d(out3,4)
        out3=out3.view(out3.size(0),-1)
        out_mac=torch.cat((out2,out3),dim=1)
        #print(out_mac.shape)
        out_mac=self.mac(out_mac)
        return out_mac  
    def mnist_acc(self):
        error_count=0
        for step,(b_x,b_y) in enumerate(self.test_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ans=self(b_x)
            for a in range(len(ans)):
                if torch.argmax(ans[a])!=b_y[a]:
                    error_count=error_count+1
        return error_count/10000.0*100
    def verify_acc(self):
        error_count=0
        for step,(b_x,b_y) in enumerate(self.qr_host_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ans=self.verify(b_x)
            for i in range(len(b_y)):
                if torch.argmax(ans[i])!=b_y[i]:
                    error_count=error_count+1
        return error_count/n*100.0  
        
def run(do,epoch1,epoch2,epoch3,epoch4,l,DA_flag):
    # Initialize primary task's dataset(training).
    transform_train=transforms.Compose([
        #transforms.RandomCrop(28,padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    transform_test=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    train_data=torchvision.datasets.MNIST(
        root="./data/MNIST",
        train=True,
        transform=transform_train,
        download=False    
    )
    train_loader=Data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True    
    )
    test_data=torchvision.datasets.MNIST(
        root="./data/MNIST",
        train=False,
        transform=transform_test,
        download=False    
    )   
    test_loader=Data.DataLoader(
        dataset=test_data,
        batch_size=128,
        shuffle=True    
    )
    
    # Initialize WM task's dataset(host and steal).
    qrdataset=QRDataset("./data/newqr/200_28/index.txt",torchvision.transforms.ToTensor())
    qr_host,qr_steal=torch.utils.data.random_split(qrdataset,[int(n),int(n)],generator=torch.Generator().manual_seed(4396))
    qr_host_loader=Data.DataLoader(
        dataset=qr_host,
        batch_size=16,
        shuffle=True    
    )
    qr_steal_loader=Data.DataLoader(
        dataset=qr_steal,
        batch_size=16,
        shuffle=True    
    )
    # Initialize the model.
    mm=ResNet(ResidualBlock,10,test_loader,qr_host_loader,qr_steal_loader,do).to(device)
    
    # Initialize the loss function.
    loss_function=nn.CrossEntropyLoss()

    # Initialize the optimizer.          
    optimizer=Adam(mm.parameters(),lr=0.0003)
    
    # Train the primary task.
    for epoch in range(epoch1):
        print("Primary task, epoch = %i in %i"% (epoch,epoch1))
        time_start=time.process_time()
        for step,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            op=mm(b_x)
            loss=loss_function(op,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        temp1=mm.mnist_acc()
        print("Primary error rate = %f" % temp1)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")

    # Prepare the regularizer in embedding.
    conv1_p=[]
    for param in mm.conv1.parameters():
        temp=copy.deepcopy(param)
        conv1_p.append(temp)
    layer1_p=[]
    for param in mm.layer1.parameters():
        temp=copy.deepcopy(param)
        layer1_p.append(temp)
    layer2_p=[]
    for param in mm.layer2.parameters():
        temp=copy.deepcopy(param)
        layer2_p.append(temp)
    layer3_p=[]
    for param in mm.layer3.parameters():
        temp=copy.deepcopy(param)
        layer3_p.append(temp)
    optimizer=Adam(mm.parameters(),lr=0.0003)
    
    # Train the WM task (Embedding).
    for epoch in range(epoch2):
        print("Watermark embedding, epoch = %i in %i"% (epoch,epoch2))
        time_start=time.process_time()
        if (epoch>(epoch2*0.5) and (epoch%5)==0 and DA_flag):
            # When DA option is activated.
            # Copy and train the shadow model for one step.
            # Simulating tuning.
            mm_=copy.deepcopy(mm)
            optimizer_=Adam(mm_.parameters(),lr=0.0003)
            for t in range(5):
                step,(ptb_x,ptb_y)=next(enumerate(train_loader))
                ptb_x=ptb_x.to(device)
                ptb_y=ptb_y.to(device)
                op=mm_(ptb_x)
                loss=loss_function(op,ptb_y)
                optimizer_.zero_grad()
                loss.backward()
                optimizer_.step()
            # Propogate the shadow model's feature extractor as DA for the WM task.
            for i in range(5):
                #print("Shadow...")
                for step,(b_x,b_y) in enumerate(qr_host_loader):
                    b_x=b_x.to(device)
                    b_y=b_y.to(device)
                    op=mm_.conv1(b_x)
                    op=mm_.layer1(op)
                    op2=mm_.layer2(op)
                    op3=mm_.layer3(op2)
                    op2=F.avg_pool2d(op2,4)
                    op2=op2.view(op2.size(0),-1)
                    op3=F.avg_pool2d(op3,4)
                    op3=op3.view(op3.size(0),-1)
                    op_mac=torch.cat((op2,op3),dim=1)
                    op_mac=op_mac.view(op.size(0),-1)
                    op_final=mm.mac(op_mac)
                    loss=loss_function(op_final,b_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        else:  
            for step,(b_x,b_y) in enumerate(qr_host_loader):
                b_x=b_x.to(device)
                b_y=b_y.to(device)
                op=mm.verify(b_x)
                loss=loss_function(op,b_y)
                temp=0
                for param in mm.conv1.parameters():
                    loss=loss+l*torch.sum((param-conv1_p[temp])**2)
                    temp=temp+1
                temp=0
                for param in mm.layer1.parameters():
                    loss=loss+l*torch.sum((param-layer1_p[temp])**2)
                    temp=temp+1
                temp=0
                for param in mm.layer2.parameters():
                    loss=loss+l*torch.sum((param-layer2_p[temp])**2)
                    temp=temp+1
                temp=0
                for param in mm.layer3.parameters():
                    loss=loss+l*torch.sum((param-layer3_p[temp])**2)
                    temp=temp+1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        temp1=mm.mnist_acc()
        temp2=mm.verify_acc()
        print("Primary error rate = %f" % temp1)
        print("WM host error rate = %f" % temp2)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")

    # Prepare FT optimizer.
    optimizer=Adam(mm.parameters(),lr=0.0003)

    # Primary task, FT.
    for epoch in range(epoch3):
        print("Fine tuned epoch = %i in %i"% (epoch,epoch3))
        time_start=time.process_time()
        for step,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            op=mm(b_x)
            loss=loss_function(op,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        temp1=mm.mnist_acc()
        temp2=mm.verify_acc()
        print("Primary error rate = %f" % temp1)
        print("WM host error rate = %f" % temp2)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")

DA_flag=True
l=0.1
mm=run(None,80,400,0,l,DA_flag)


