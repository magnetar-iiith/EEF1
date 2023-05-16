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
# from tqdm import tqdm
# import seaborn as sns
# from pylab import rcParams
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EFNN6_1(nn.Module):
    def __init__(self,config,logger_file_name):
        super(EFNN6_1, self).__init__()
        self.config = config
        self.logger_file_name = logger_file_name
        self.util =  util.Util(config,logger_file_name)
        self.model_path = 'models/' + logger_file_name + '.pth'


        self.layers = nn.ModuleList([ nn.Linear(  config.net.activation_units[i] , config.net.activation_units[i+1]  ).to(device)   for i in range(config.net.num_layers)])
        
        logger_path = 'logs/' + logger_file_name +'.log'
        logger1.addHandler(logging.FileHandler(filename=logger_path))

        self.softmaxfn = nn.Softmax(dim=1).to(device)
        self.values = torch.tensor(self.config.values).to(device)
        
        self.dropout = nn.Dropout(0.25)

        self.activation_functions  =  edict({
            'identity' : nn.Identity().to(device), 
            'softmax'  : nn.Softmax(dim=1).to(device),
            'logsoftmax': nn.LogSoftmax(dim=1).to(device),
            'lrelu'     : nn.LeakyReLU().to(device),
            'relu'     : nn.ReLU().to(device),
            'hardtanh' : nn.Hardtanh().to(device),
            'sigmoid'  : nn.Sigmoid().to(device),
            'tanh' : nn.Tanh().to(device),
            })



        ## initialization of weights and bias is left
        for k in range(config.net.num_layers):
            # nn.init.uniform_(self.layers[k].weight)
            nn.init.xavier_uniform_(self.layers[k].weight)
            self.layers[k].bias.data.fill_(0.0)


    def forward(self,x, mulfactor=9 ):
        x = self.util.preprocessing(x)
        for i in range(self.config.net.num_layers):
            # x = self.activation_functions[self.config.net.activation[i]](self.layers[i](x))
            x = self.activation_functions['tanh'](self.layers[i](x))

            x = self.dropout(x)

        x = self.util.revert_preprocessing(x)

        # values = torch.tensor([[0.999] , [1]]).to(device)
        if(self.config.valuesmul==1):
            permvalue = torch.randperm(self.config.num_agents)
            values=self.values[permvalue]
            x = x * values

        x = x * mulfactor
        x = self.softmaxfn(x)

        # print('x shape', x.size())

        return x


