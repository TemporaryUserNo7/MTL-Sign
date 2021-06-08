# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:16:52 2020

@author: Administrator
"""
import copy
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.optim import Adam,SGD
import torch.utils.data as Data
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import gen_qr_keys
import time
import os
# Initialize random seed.
np.random.seed(516)
torch.manual_seed(516)
# Assign security parameter.
n=200.0
# Assign device.
device=torch.device("cuda:3")
DA_flag=True

# Define QR dataset.
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

# Define the target neural network.
# Parameters: Dropout, loaders for test.
class MTCNN(nn.Module):
    def __init__(self,test_data_x,test_data_y,qr_host_loader,qr_steal_loader,do=None):
        super(MTCNN,self).__init__()
        self.test_data_x=test_data_x
        self.test_data_y=test_data_y
        self.qr_host_loader=qr_host_loader
        self.qr_steal_loader=qr_steal_loader
        self.conv1=nn.Sequential(
            nn.Conv2d(1,16,3,1,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,2,1,1),
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            nn.Linear(64*8*8,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,10)
        )
        if do==None:
            self.mac=nn.Sequential(
                nn.Linear(64*8*8,512),
                nn.ReLU(),
                #nn.Dropout(p=0.2),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Linear(128,32),
                nn.ReLU(),
                nn.Linear(32,2)   
            )
        else:
            self.mac=nn.Sequential(
                nn.Linear(64*8*8,512),
                nn.ReLU(),
                nn.Dropout(p=do),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Linear(128,32),
                nn.ReLU(),
                nn.Linear(32,2)   
            )
        if do==None:
            self.hacker=nn.Sequential(
                nn.Linear(64*8*8,512),
                nn.ReLU(),
                #nn.Dropout(p=0.2),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Linear(128,32),
                nn.ReLU(),
                nn.Linear(32,2)   
            )
        else:
           self.hacker=nn.Sequential(
                nn.Linear(64*8*8,512),
                nn.ReLU(),
                nn.Dropout(do),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Linear(128,32),
                nn.ReLU(),
                nn.Linear(32,2)   
            ) 
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
    def verify(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.mac(x)
        return x
    def hack(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.hacker(x)
        return x
    def mnist_acc(self):
        error_count=0
        ans=self(self.test_data_x)
        for i in range(10000):
            if torch.argmax(ans[i])!=self.test_data_y[i]:
                error_count=error_count+1
        return error_count/10000.0*100.0
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
    def valid_acc(self):
        error_count=0
        for step,(b_x,b_y) in enumerate(self.qr_steal_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ans=self.verify(b_x)
            for i in range(len(b_y)):
                if torch.argmax(ans[i])!=b_y[i]:
                    error_count=error_count+1
        return error_count/n*100.0
    def hack_verify_acc(self):
        error_count=0
        for step,(b_x,b_y) in enumerate(self.qr_host_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ans=self.hack(b_x)
            for i in range(len(b_y)):
                if torch.argmax(ans[i])!=b_y[i]:
                    error_count=error_count+1
        return error_count/n*100.0  
    def hack_valid_acc(self):
        error_count=0
        for step,(b_x,b_y) in enumerate(self.qr_steal_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            ans=self.hack(b_x)
            for i in range(len(b_y)):
                if torch.argmax(ans[i])!=b_y[i]:
                    error_count=error_count+1
        return error_count/n*100.0
    def NP(self,threshold=0.01):
        for param in self.parameters():
            param.data.apply_(lambda x:x if abs(x)>=threshold else 0)
        return 0

def run(do,epoch1,epoch2,epoch3,epoch4):
    # Initialize primary task's dataset(training).
    train_data=torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )
    train_loader=Data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True    
    )
    # Initialize primary task's dataset(training).
    test_data=torchvision.datasets.FashionMNIST(
        root="./data/FashionMNIST",
        train=False,
        download=False
    )
    test_data_x=test_data.data.type(torch.FloatTensor)/255.0    
    test_data_x=torch.unsqueeze(test_data_x,dim=1)        
    test_data_y=test_data.targets 
    test_data_x=test_data_x.to(device)
    test_data_y=test_data_y.to(device)    
    # Initialize WM task's dataset(host and steal).
    qrdataset=gen_qr_keys.QRDataset("./data/newqr/200_28/index.txt",torchvision.transforms.ToTensor())
    qr_host,qr_steal=torch.utils.data.random_split(qrdataset,[int(n),int(n)])
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
    mm=MTCNN(test_data_x,test_data_y,qr_host_loader,qr_steal_loader,do).to(device)
    
    # Initialize the history records.
    row1_primary=[]
    row1_host_host=[]
    row1_host_steal=[]
    row1_steal_host=[]
    row1_steal_steal=[]
    row2_primary=[]
    row2_host_host=[]
    row2_host_steal=[]
    row2_steal_host=[]
    row2_steal_steal=[]
    row3_primary=[]
    row3_host_host=[]
    row3_host_steal=[]
    row3_steal_host=[]
    row3_steal_steal=[]
    row4_primary=[]
    row4_host_host=[]
    row4_host_steal=[]
    row4_steal_host=[]
    row4_steal_steal=[]
    
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
        temp2=mm.verify_acc()
        temp3=mm.valid_acc()
        temp4=mm.hack_verify_acc()
        temp5=mm.hack_valid_acc()
        print("Primary error rate = %f" % temp1)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")
        row1_primary.append(temp1)
        row1_host_host.append(temp2)
        row1_host_steal.append(temp3)
        row1_steal_host.append(temp4)
        row1_steal_steal.append(temp5)

    # Prepare the regularizer in embedding.
    conv1_p=[]
    for param in mm.conv1.parameters():
        temp=copy.deepcopy(param)
        conv1_p.append(temp)
    conv2_p=[]
    for param in mm.conv2.parameters():
        temp=copy.deepcopy(param)
        conv2_p.append(temp)
    conv3_p=[]
    for param in mm.conv3.parameters():
        temp=copy.deepcopy(param)
        conv3_p.append(temp)   
      
    l=0.3
    # Train the WM task (Embedding).
    for epoch in range(epoch2):
        print("Watermark embedding, epoch = %i in %i"% (epoch,epoch2))
        time_start=time.process_time()
        if (epoch>(epoch2*0.25) and (epoch%10)==0 and DA_flag):
            step,(ptb_x,ptb_y)=next(enumerate(train_loader))
            ptb_x=ptb_x.to(device)
            ptb_y=ptb_y.to(device)
            mm_=copy.deepcopy(mm)
            optimizer_=Adam(mm_.parameters(),lr=0.0003)
            op=mm_(ptb_x)
            loss=loss_function(op,ptb_y)
            optimizer_.zero_grad()
            loss.backward()
            optimizer_.step()
            for i in range(10):
                for step,(b_x,b_y) in enumerate(qr_host_loader):
                    b_x=b_x.to(device)
                    b_y=b_y.to(device)
                    op=mm_.conv1(b_x)
                    op=mm_.conv2(op)
                    op=mm_.conv3(op)
                    op=op.view(op.size(0),-1)
                    op=mm.mac(op)
                    loss=loss_function(op,b_y)
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
                for param in mm.conv2.parameters():
                    loss=loss+l*torch.sum((param-conv2_p[temp])**2)
                    temp=temp+1
                temp=0
                for param in mm.conv3.parameters():
                    loss=loss+l*torch.sum((param-conv3_p[temp])**2)
                    temp=temp+1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        temp1=mm.mnist_acc()
        temp2=mm.verify_acc()
        temp3=mm.valid_acc()
        temp4=mm.hack_verify_acc()
        temp5=mm.hack_valid_acc()
        print("Primary error rate = %f" % temp1)
        print("WM host error rate = %f" % temp2)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")
        row2_primary.append(temp1)
        row2_host_host.append(temp2)
        row2_host_steal.append(temp3)
        row2_steal_host.append(temp4)
        row2_steal_steal.append(temp5)

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
        temp3=mm.valid_acc()
        temp4=mm.hack_verify_acc()
        temp5=mm.hack_valid_acc()
        print("Primary error rate = %f" % temp1)
        print("WM host error rate = %f" % temp2)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")
        row3_primary.append(temp1)
        row3_host_host.append(temp2)
        row3_host_steal.append(temp3)
        row3_steal_host.append(temp4)
        row3_steal_steal.append(temp5)

    # Prepare the regularizer in thief's embedding.
    conv1_p=[]
    for param in mm.conv1.parameters():
        temp=copy.deepcopy(param)
        conv1_p.append(temp)
    conv2_p=[]
    for param in mm.conv2.parameters():
        temp=copy.deepcopy(param)
        conv2_p.append(temp)
    conv3_p=[]
    for param in mm.conv3.parameters():
        temp=copy.deepcopy(param)
        conv3_p.append(temp)
        
    # Prepare the thief's WM optimizer.
    optimizer=Adam(mm.parameters(),lr=0.0005)
    for epoch in range(epoch4):
        print("Hacker tune Epoch = %i in %i"% (epoch,epoch4))
        time_start=time.process_time()
        for step,(b_x,b_y) in enumerate(qr_steal_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            op=mm.hack(b_x)
            loss=loss_function(op,b_y)
            temp=0
            for param in mm.conv1.parameters():
                loss=loss+torch.sum((param-conv1_p[temp])**2)
                temp=temp+1
            temp=0
            for param in mm.conv2.parameters():
                loss=loss+torch.sum((param-conv2_p[temp])**2)
                temp=temp+1
            temp=0
            for param in mm.conv3.parameters():
                loss=loss+torch.sum((param-conv3_p[temp])**2)
                temp=temp+1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        temp1=mm.mnist_acc()
        temp2=mm.verify_acc()
        temp3=mm.valid_acc()
        temp4=mm.hack_verify_acc()
        temp5=mm.hack_valid_acc()
        print("Primary error rate = %f" % temp1)
        print("WM host error rate = %f" % temp2)
        print("WM steal error rate = %f" % temp5)
        print("Time elapsed = ", time.process_time()-time_start)
        print("---------------------------------------------")
        row4_primary.append(temp1)
        row4_host_host.append(temp2)
        row4_host_steal.append(temp3)
        row4_steal_host.append(temp4)
        row4_steal_steal.append(temp5)
        
    history=[]
    history.append(row1_primary)
    history.append(row1_host_host)
    history.append(row1_host_steal)
    history.append(row1_steal_host)
    history.append(row1_steal_steal)
    history.append(row2_primary)
    history.append(row2_host_host)
    history.append(row2_host_steal)
    history.append(row2_steal_host)
    history.append(row2_steal_steal)
    history.append(row3_primary)
    history.append(row3_host_host)
    history.append(row3_host_steal)
    history.append(row3_steal_host)
    history.append(row3_steal_steal)
    history.append(row4_primary)
    history.append(row4_host_host)
    history.append(row4_host_steal)
    history.append(row4_steal_host)
    history.append(row4_steal_steal)
    # Dump the history
    return mm,history

def paint_12(history,filepath):
    plt.figure(figsize=(50,50),dpi=80)
    plt.subplot(4,3,1)
    plt.plot(history[0])
    plt.ylim([0,50])
    plt.title("FashionMNIST performance.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,2)
    plt.plot(history[1],color="blue",label="Host WM.")
    plt.plot(history[2],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Host performance.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,3)
    plt.plot(history[3],color="blue",label="Host WM.")
    plt.plot(history[4],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Adversary performance.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")

    plt.subplot(4,3,4)
    plt.plot(history[5])
    plt.ylim([0,20])
    plt.title("Fashion MNIST performance. WM embedding.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,5)
    plt.plot(history[6],color="blue",label="Host WM.")
    plt.plot(history[7],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Host performance. WM embedding.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,6)
    plt.plot(history[8],color="blue",label="Host WM.")
    plt.plot(history[9],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Adversary performance. WM embedding.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")

    plt.subplot(4,3,7)
    plt.plot(history[10])
    plt.ylim([0,20])
    plt.title("Fashion MNIST performance. Fine tuned.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,8)
    plt.plot(history[11],color="blue",label="Host WM.")
    plt.plot(history[12],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Host performance. Fine tuned.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,9)
    plt.plot(history[13],color="blue",label="Host WM.")
    plt.plot(history[14],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Adversary performance. Fine tuned.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    
    plt.subplot(4,3,10)
    plt.plot(history[15])
    plt.ylim([0,20])
    plt.title("Fashion MNIST performance. Hacker tuned.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,11)
    plt.plot(history[16],color="blue",label="Host WM.")
    plt.plot(history[17],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Host performance. Hacker tuned.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.subplot(4,3,12)
    plt.plot(history[18],color="blue",label="Host WM.")
    plt.plot(history[19],color="red",label="Adversary WM.")
    plt.legend()
    plt.ylim([0,100])
    plt.title("Adversary performance. Hacker tuned.")
    plt.xlabel("No. of iterations.")
    plt.ylabel("Error rate (%)")
    plt.savefig(filepath)
    return 0

def paint_detail(history,prefix):
    fs=18
    if not os.path.exists(prefix):
        os.mkdir(prefix)
        
    mem=history[0]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem,linewidth=2,color="deeppink")
    plt.ylim([0,20])
    plt.ylabel("Primary task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST primary task training.",fontsize=fs)
    plt.savefig(prefix+"primary_1.png")
    
    mem=history[1]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem,linewidth=2,color="blue")
    plt.ylim([0,100])
    plt.ylabel("WM task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST primary task training.",fontsize=fs)
    plt.savefig(prefix+"primary_2.png")
    
    mem=history[5]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem,linewidth=2,color="deeppink")
    plt.ylim([0,40])
    plt.ylabel("Primary task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST WM task training.",fontsize=fs)
    plt.savefig(prefix+"embedding_1.png")
    
    mem1=history[6]
    mem2=history[7]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem1,linewidth=2,color="blue",label="Host's WM.")
    plt.plot(mem2,linewidth=2,color="red",label="Adversary's WM.")
    plt.legend(fontsize=fs)
    plt.ylim([0,100])
    plt.ylabel("WM task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem1),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST WM task training",fontsize=fs)
    plt.savefig(prefix+"embedding_2.png")
    
    mem=history[10]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem,linewidth=2,color="deeppink")
    plt.ylim([0,20])
    plt.ylabel("Primary task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST primary task FT.",fontsize=fs)
    plt.savefig(prefix+"ft_1.png")
    
    mem1=history[11]
    mem2=history[12]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem1,linewidth=2,color="blue",label="Host's WM.")
    plt.plot(mem2,linewidth=2,color="red",label="Adversary's WM.")
    plt.legend(fontsize=fs)
    plt.ylim([0,100])
    plt.ylabel("WM task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem1),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST primary task FT.",fontsize=fs)
    plt.savefig(prefix+"ft_2.png")
    
    mem=history[15]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem,linewidth=2,color="deeppink")
    plt.ylim([0,20])
    plt.ylabel("Primary task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST WM overwriting.",fontsize=fs)
    plt.savefig(prefix+"attack_1.png")
    
    mem1=history[16]
    mem2=history[17]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem1,linewidth=2,color="blue",label="Host's WM.")
    plt.plot(mem2,linewidth=2,color="red",label="Adversary's WM.")
    plt.legend(fontsize=fs)
    plt.ylim([0,100])
    plt.ylabel("WM task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem1),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST WM overwriting (host).",fontsize=fs)
    plt.savefig(prefix+"attack_2.png")
    
    mem1=history[18]
    mem2=history[19]
    plt.figure(figsize=(8,6),dpi=200)
    plt.plot(mem1,linewidth=2,color="blue",label="Host's WM.")
    plt.plot(mem2,linewidth=2,color="red",label="Adversary's WM.")
    plt.legend(fontsize=fs)
    plt.ylim([0,100])
    plt.ylabel("WM task error rate (%).",fontsize=fs)
    x_ticks = np.arange(0,len(mem1),10)
    plt.xticks(x_ticks)
    plt.xlabel("No. of epochs.",fontsize=fs)
    plt.title("FashionMNIST WM overwriting (adversary).",fontsize=fs)
    plt.savefig(prefix+"attack_3.png")
    return 0

mm,history=run(0.2,100,200,20,60)
paint_detail(history,"./figure/fashion_mnist_figures_model1_0.2_100_200_20_60_DA/")
torch.save(mm,"./model/fashion_mnist_model1_0.2_100_200_20_60_DA.pkl")          
        
