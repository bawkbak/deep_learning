
from dataloader import read_bci_data
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import matplotlib.pyplot as plt

class EEG(nn.Module):
    def __init__(self,act_func='ELU'):
        super(EEG, self).__init__()
        if act_func == 'ELU': self.act_func = nn.ReLU()
        if act_func == 'LeakyReLU': self.act_func = nn.LeakyReLU()
        if act_func == 'ReLU': self.act_func = nn.ReLU()
        self.pipe0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51), stride=(1,1),padding=(0,25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.pipe1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.pipe2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(in_features=736,out_features=2,bias=True)
        )
    def forward(self,x):
        x = self.pipe0(x)
        x = self.act_func(x)
        x = self.pipe1(x)
        x = self.act_func(x)
        x = self.pipe2(x)
        return x


class DeepConvNet(nn.Module):
    def __init__(self,act_func='ELU'):
        super(DeepConvNet, self).__init__()
        if act_func == 'ELU': self.act_func = nn.ReLU()
        if act_func == 'LeakyReLU': self.act_func = nn.LeakyReLU()
        if act_func == 'ReLU': self.act_func = nn.ReLU()
        C, T, N = 2, 750, 2
        self.pipe0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5)),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(C,1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        )
        self.pipe1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5)),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        )
        self.pipe2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5)),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        )
        self.pipe3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,5)),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        )
        self.pipe4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=8600,out_features=2)
        )

    def forward(self,x):
        x = self.pipe0(x)
        x = self.act_func(x)
        x = self.pipe1(x)
        x = self.act_func(x)
        x = self.pipe2(x)
        x = self.act_func(x)
        x = self.pipe3(x)
        x = self.act_func(x)
        x = self.pipe4(x)
        return x


i = 0
def train( model, train_data, train_label, optimizer, batchsize):
    global i
    count = 0
    model.train()
    while count<1080:
        data = torch.cuda.FloatTensor( train_data[i:i+batchsize] )
        target = torch.cuda.LongTensor( train_label[i:i+batchsize] )
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()
        loss = loss(output, target)
        loss.backward()
        optimizer.step()

        i = (i+batchsize)%1080
        count += batchsize

def test(model, test_data, test_label):
    model.eval()
    data = torch.cuda.FloatTensor( test_data )
    target = torch.cuda.LongTensor( test_label )
    output = model(data)
    loss = nn.CrossEntropyLoss()
    test_loss = loss(output, target)  # sum up batch loss
    pred = output.argmax(dim=1)  # get the index of the max log-probability
    correct = 0
    for i,pred_ans in enumerate(pred):
        if pred[i] == target[i]: correct += 1
    return test_loss.item()/1080.0 , correct/1080.0

def max(old_record, new_record):
    if (old_record >= new_record):
        return old_record
    else:
        return new_record


if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda:0')
    train_data, train_label, test_data, test_label = read_bci_data()

    
    for task in ['EEG', 'DeepConvNet']: #for task in ['EEG', 'DeepConvNet']:
        hyper_lr = 0.00131 #0.00132
        hyper_epoch = 300
        hyper_batchsize = 128 #128


        plt_array_loss = []
        plt_array_accuracy = []
        for act_func in ['ReLU', 'LeakyReLU', 'ELU']:
            for testset in ['train','test']:
                print(str(task+'_'+act_func+'_'+testset))
                plt_array_loss_tmp = []
                plt_array_accuracy_tmp = []
                max_accu = 0

                if testset == 'train':
                    m_data, m_label = train_data, train_label
                elif testset == 'test':
                    m_data, m_label = test_data, test_label

                if task == 'EEG':
                    model = EEG(act_func=act_func)
                elif task == 'DeepConvNet':
                    model = DeepConvNet(act_func=act_func)

                model.to(device)
                optimizer = optim.Adam(model.parameters(),lr = float(hyper_lr)) 
                
                for epoch in range(hyper_epoch + 1):
                    train(model, train_data, train_label, optimizer, batchsize = int(hyper_batchsize))
                    test_loss, correct = test(model, m_data, m_label)
                    max_accu = max(max_accu, correct)
                    plt_array_accuracy_tmp.append(correct*100)
                    plt_array_loss_tmp.append(test_loss)
                    if (epoch * 2) % hyper_epoch == 0: 
                        print('epoch: {val_epoch}, loss: {val_loss} accuracy: {val_accu}'.format(val_epoch = epoch, val_loss = test_loss, val_accu = correct))
                if testset == 'test':
                    print('best accuracy: {accuracy}'.format(accuracy = max_accu))
                plt_array_accuracy.append(plt_array_accuracy_tmp)
                plt_array_loss.append(plt_array_loss_tmp)
               



        for arr in plt_array_accuracy: plt.plot(arr)
        plt.title(str("Activation Functions comparision ("+task+')'))
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend(['relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test', 'elu_train', 'elu_test',])
        plt.savefig(str(task+'_accuracy.png'))
        plt.close()
        plt.show()

        for arr in plt_array_loss: plt.plot(arr)
        plt.grid()
        plt.title(str("Learning curve comparision ("+task+')'))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['relu_train', 'relu_test', 'leaky_relu_train', 'leaky_relu_test', 'elu_train', 'elu_test',])
        plt.savefig(str(task+'_loss.png'))
        plt.close()
        plt.show()