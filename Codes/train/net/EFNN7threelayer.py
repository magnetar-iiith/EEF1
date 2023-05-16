# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

from utils import util

## Imports
import torch

import os
import numpy as np
# import pandas as pd
# import pandas.util.testing as tm
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim

import torch.nn.functional as F

from random import shuffle

import time
import logging
logging.getLogger('').setLevel(logging.DEBUG)
logger1 = logging.getLogger('1')

from easydict import EasyDict as edict
# import copy
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device :  ',device)

class EFNN7threelayer(nn.Module):
    def __init__(self,config,logger_file_name):
        super(EFNN7threelayer, self).__init__()
        # self.config = config
        # self.config = edict(config.deepcopy())
        self.config = edict(config.copy())
        self.config.net = edict(self.config.net.copy())
        # self.config = copy.deepcopy(config)

        self.logger_file_name = logger_file_name
        self.util =  util.Util(config,logger_file_name)
        self.model_path = 'models/' + logger_file_name + '.pth'
        self.softmaxfn = nn.Softmax(dim=1).cuda(device)

        self.sigmoid = nn.Sigmoid().to(device)
        self.tanh = nn.Tanh().to(device)
        self.relu = nn.ReLU().to(device)
        
        # self.values = torch.tensor(self.config.values).to(device)

        self.maxpoollayer = nn.MaxPool2d((self.config.num_agents, 1) , return_indices=True)

        # self.tanh = torch.tanh(a)

        self.activation = {
            'relu':self.relu,
            'tanh':self.tanh
        }


        # self.conv = []
        # self.deconv = []

        # self.conv1 = nn.Conv2d(1, 8, 3).to(device)
        # self.conv2 = nn.Conv2d(8, 16, 3).to(device)
        # self.conv3 = nn.Conv2d(16, 32, 3).to(device)
        # self.conv4 = nn.Conv2d(32, 64, 3).to(device)
        # self.conv5 = nn.Conv2d(64, 128, 2).to(device)
        # # self.conv6 = nn.Conv2d(128, 256, 1).to(device)
        # # self.reverseconv6 = nn.ConvTranspose2d(256, 128, 1).to(device)
        # self.reverseconv5 = nn.ConvTranspose2d(128, 64, 2).to(device)
        # self.reverseconv4 = nn.ConvTranspose2d(64, 32, 3).to(device)
        # self.reverseconv3 = nn.ConvTranspose2d(32, 16, 3).to(device)
        # self.reverseconv2 = nn.ConvTranspose2d(16, 8, 3).to(device)
        # self.reverseconv1 = nn.ConvTranspose2d(8, 1, 3).to(device)

    

        self.conv = nn.ModuleList([ nn.Conv2d( self.config.net.convlayerunits[i],self.config.net.convlayerunits[i+1],self.config.net.convkernelsize[i]).to(device)   for i in range(len(self.config.net.convlayerunits)-1)])
        self.deconv = nn.ModuleList([ nn.ConvTranspose2d(   self.config.net.deconvlayerunits[i],self.config.net.deconvlayerunits[i+1],self.config.net.deconvkernelsize[i] ).to(device)   for i in range(len(self.config.net.deconvlayerunits)-1)])
        
        self.conv1 = nn.ModuleList([ nn.Conv2d( self.config.net.convlayerunits[i],self.config.net.convlayerunits[i+1],self.config.net.convkernelsize[i]).to(device)   for i in range(len(self.config.net.convlayerunits)-1)])
        self.deconv1 = nn.ModuleList([ nn.ConvTranspose2d(   self.config.net.deconvlayerunits[i],self.config.net.deconvlayerunits[i+1],self.config.net.deconvkernelsize[i] ).to(device)   for i in range(len(self.config.net.deconvlayerunits)-1)])
        
        self.conv2 = nn.ModuleList([ nn.Conv2d( self.config.net.convlayerunits[i],self.config.net.convlayerunits[i+1],self.config.net.convkernelsize[i]).to(device)   for i in range(len(self.config.net.convlayerunits)-1)])
        self.deconv2 = nn.ModuleList([ nn.ConvTranspose2d(   self.config.net.deconvlayerunits[i],self.config.net.deconvlayerunits[i+1],self.config.net.deconvkernelsize[i] ).to(device)   for i in range(len(self.config.net.deconvlayerunits)-1)])
        self.conv3 = nn.ModuleList([ nn.Conv2d( self.config.net.convlayerunits[i],self.config.net.convlayerunits[i+1],self.config.net.convkernelsize[i]).to(device)   for i in range(len(self.config.net.convlayerunits)-1)])
        self.deconv3 = nn.ModuleList([ nn.ConvTranspose2d(   self.config.net.deconvlayerunits[i],self.config.net.deconvlayerunits[i+1],self.config.net.deconvkernelsize[i] ).to(device)   for i in range(len(self.config.net.deconvlayerunits)-1)])

        self.last = nn.ConvTranspose2d(8,1,3).to(device)
        # print('___________________________-')
        # print(self.conv)
        # print('___________________________-')
        # print(self.deconv)
        # print()

        for i in range(len(self.config.net.convlayerunits)-1):
            nn.init.xavier_uniform_(self.conv[i].weight)
            nn.init.xavier_uniform_(self.conv1[i].weight)
            nn.init.xavier_uniform_(self.conv2[i].weight)
            nn.init.xavier_uniform_(self.conv3[i].weight)
            self.conv[i].bias.data.fill_(0.0)
            self.conv1[i].bias.data.fill_(0.0)
            self.conv2[i].bias.data.fill_(0.0)
            self.conv3[i].bias.data.fill_(0.0)

        for i in range(len(self.config.net.deconvlayerunits)-1):
            nn.init.xavier_uniform_(self.deconv[i].weight)
            nn.init.xavier_uniform_(self.deconv1[i].weight)
            nn.init.xavier_uniform_(self.deconv2[i].weight)
            nn.init.xavier_uniform_(self.deconv3[i].weight)
            self.deconv[i].bias.data.fill_(0.0)
            self.deconv1[i].bias.data.fill_(0.0)
            self.deconv2[i].bias.data.fill_(0.0)
            self.deconv3[i].bias.data.fill_(0.0)
            

        # self.pool = nn.MaxPool2d(2, 2).to(device)

        # self.dropoutHidden = nn.Dropout(self.config.train.dropoutratehidden)
        # self.dropoutVisible = nn.Dropout(self.config.train.dropoutratevisible)

    def forward(self,x, temperature=0.1 ):

        # print()
        # print(self.config)
        # print()
        # print()

        num_agents = x.shape[2]
        num_items = x.shape[3]

        # x = x [:,None,:,:]

        # x1 = F.tanh(self.conv1(x))
        # x2 = F.tanh(self.conv2(x1))
        # x3 = F.tanh(self.conv3(x2))
        # x4 = F.tanh(self.conv4(x3))
        # x5 = F.tanh(self.conv5(x4))
        # x = F.tanh(self.reverseconv5(x5))
        # # x = torch.cat((x4,x),1)
        # x = F.tanh(self.reverseconv4(x))
        # # x = torch.cat((x3,x),1)
        # x = F.tanh(self.reverseconv3(x))
        # # x = torch.cat((x2,x),1)
        # x = F.tanh(self.reverseconv2(x))
        # # x = torch.cat((x1,x),1)
        # x = F.tanh(self.reverseconv1(x))

        skippedconnectionsave = []    
        for i in range(len(self.config.net.convlayerunits)-1):
            # if(self.config.train.dropout):
                # x = self.dropoutHidden(self.activation[self.config.net.convactivation[i]](self.conv[i](x)))
            # else:
            x = self.activation[self.config.net.convactivation[i]](self.conv[i](x))


            if(self.config.net.skippedconnection[i]):
                skippedconnectionsave.append(x.clone())
            else:
                skippedconnectionsave.append([])

        for i in range(len(self.config.net.deconvlayerunits)-1):

            if(self.config.net.skippedconnection[i]):
                x = torch.cat((skippedconnectionsave[i],x),1)

            # if(self.config.train.dropout):
            #     if( i == len(self.config.net.deconvlayerunits)):
            #         x = (self.activation[self.config.net.deconvactivation[i]](self.deconv[i](x)))
            #     else:
            #         x = self.dropoutHidden(self.activation[self.config.net.deconvactivation[i]](self.deconv[i](x)))
            # else:
            x = self.activation[self.config.net.deconvactivation[i]](self.deconv[i](x))

        # print('_____________________')
        # print(x.shape)
        # print('_____________________')
        # x = x.view(-1, self.config.num_agents, self.config.num_items)
        # x = x [:,None,:,:]
        # print('_____________________')
        # print(x.shape)
        # print('_____________________')

        for i in range(len(self.config.net.convlayerunits)-1):
            x = self.activation[self.config.net.convactivation[i]](self.conv1[i](x))
        for i in range(len(self.config.net.deconvlayerunits)-1):
            x = self.activation[self.config.net.deconvactivation[i]](self.deconv1[i](x))

        # print('_____________________')
        # print(x.shape)
        # print('_____________________')
        # x = x.view(-1, self.config.num_agents, self.config.num_items)
        # x = x [:,None,:,:]
        # print('_____________________')
        # print(x.shape)
        # print('_____________________')

        for i in range(len(self.config.net.convlayerunits)-1):
            x = self.activation[self.config.net.convactivation[i]](self.conv2[i](x))
        for i in range(len(self.config.net.deconvlayerunits)-1):
            x = self.activation[self.config.net.deconvactivation[i]](self.deconv2[i](x))

        # x = x.view(-1, self.config.num_agents, self.config.num_items)
        # x = x [:,None,:,:]

        for i in range(len(self.config.net.convlayerunits)-1):
            x = self.activation[self.config.net.convactivation[i]](self.conv3[i](x))
        for i in range(len(self.config.net.deconvlayerunits)-2):
            x = self.activation[self.config.net.deconvactivation[i]](self.deconv3[i](x))

        # print('shape of x before ', x.shape)
        x = self.activation['tanh'](self.last(x))
        # print('shape of x after ', x.shape)
        x = x.view(-1, num_agents, num_items)
        # print('shape of x after ', x.shape)

        # if(self.config.valuesmul==1):
        #     permvalue = torch.randperm(self.config.num_agents)
        #     # values=self.values[permvalue]
        #     x = x * values


        if(self.config.softmax == 1):
            x =  self.softmaxfn(x)

        allocation = x * (1/temperature)

        allocation = self.softmaxfn(allocation)
        # allocation = self.sigmoid(allocation)

        # print('Allocation SHape', allocation.shape)
        # maxvaluation , indicies = self.maxpoollayer(allocation)

        # zeross = torch.zeros(allocation.shape).to(device)

        # zerosflat = torch.flatten(zeross, 1)

        # zerosflat[:,indicies[0][0]] = 1


        # zerosflat = x.reshape(allocation.shape)

        # allocation = allocation * zerosflat

        return allocation



        # "convlayerunits" : [1,8,16,32,64,128],
        # "convkernelsize":[3,3,3,3,2],
        # "convactivation" : ["tanh","tanh","tanh","tanh","tanh"],
        # "deconvlayerunits" : [128,64,32,16,8,1],
        # "deconvkernelsize":[2,3,3,3,3],
        # "deconvactivation" : ["tanh","tanh","tanh","tanh","tanh"],
        # "skippedconnection":[0,0,0,0]
