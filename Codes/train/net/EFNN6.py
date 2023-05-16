# # import matplotlib
# # import matplotlib.pyplot as plt
# import numpy as np

# from utils import util

# ## Imports
# import torch

# import os
# import numpy as np
# # import pandas as pd
# # import pandas.util.testing as tm
# from tqdm import tqdm
# # import seaborn as sns
# from pylab import rcParams
# # import matplotlib.pyplot as plt
# # from matplotlib import rc
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import confusion_matrix, classification_report
# from torch import nn, optim

# import torch.nn.functional as F

# from random import shuffle

# import time
# import logging
# logging.getLogger('').setLevel(logging.DEBUG)
# logger1 = logging.getLogger('1')

# from easydict import EasyDict as edict

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# class EFNN6(nn.Module):
#     def __init__(self,config,logger_file_name):
#         super(EFNN6, self).__init__()
#         self.config = config
#         self.logger_file_name = logger_file_name
#         self.util =  util.Util(config,logger_file_name)
#         self.model_path = 'models/' + logger_file_name + '.pth'


#         self.layers = nn.ModuleList([ nn.Linear(  config.net.activation_units[i] , config.net.activation_units[i+1]  ).cuda(device)   for i in range(config.net.num_layers)])
        
#         logger_path = 'logs/' + logger_file_name +'.log'
#         logger1.addHandler(logging.FileHandler(filename=logger_path))

#         self.softmaxfn = nn.Softmax(dim=1).cuda(device)


#         self.activation_functions  =  edict({
#             'identity' : nn.Identity().cuda(device), 
#             'softmax'  : nn.Softmax(dim=1).cuda(device),
#             'logsoftmax': nn.LogSoftmax(dim=1).cuda(device),
#             'lrelu'     : nn.LeakyReLU().cuda(device),
#             'relu'     : nn.ReLU().cuda(device),
#             'hardtanh' : nn.Hardtanh().cuda(device),
#             'sigmoid'  : nn.Sigmoid().cuda(device),
#             'tanh' : nn.Tanh().cuda(device),
#             })



#         ## initialization of weights and bias is left
#         for k in range(config.net.num_layers):
#             # nn.init.uniform_(self.layers[k].weight)
#             nn.init.xavier_uniform_(self.layers[k].weight)
#             self.layers[k].bias.data.fill_(0.0)


#     def forward(self,x, mulfactor=9 ):
#         x = self.util.preprocessing(x)
#         for i in range(self.config.net.num_layers):
#             x = self.activation_functions[self.config.net.activation[i]](self.layers[i](x))
        
#         x = self.util.revert_preprocessing(x)

#         values = torch.tensor(self.config.values).to(device)

#         permvalue = torch.randperm(self.config.num_agents)
#         values=values[permvalue]
#         x = x * values

#         x =  self.softmaxfn(x)

#         x = x * mulfactor
#         x = self.softmaxfn(x)

#         return x

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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('device :  ',device)

class EFNN6(nn.Module):
    def __init__(self,config,logger_file_name):
        super(EFNN6, self).__init__()
        # self.config = config.copy()

        self.config = edict(config.copy())

        # print(self.config)
        # print(self.config.items)
        # print(self.config['items'])
        
        # print(self.config['temperature'])

        self.config['net']['convlayerunits'] = [1,8,16,32,64,128]
        print('okay')
        self.config['net']['convkernelsize'] = [3,3,3,3,2]
        self.config['net']['deconvlayerunits'] = [128,64,32,16,8,1]
        self.config['net']['deconvkernelsize'] = [2,3,3,3,3]
        
        # print()
        # print(self.config)
        # print()

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
        # self.conv2 = nn.ModuleList([ nn.Conv2d( self.config.net.convlayerunits[i],self.config.net.convlayerunits[i+1],self.config.net.convkernelsize[i]).to(device)   for i in range(len(self.config.net.convlayerunits)-1)])
        # self.deconv2 = nn.ModuleList([ nn.ConvTranspose2d(   self.config.net.deconvlayerunits[i],self.config.net.deconvlayerunits[i+1],self.config.net.deconvkernelsize[i] ).to(device)   for i in range(len(self.config.net.deconvlayerunits)-1)])
        # self.conv3 = nn.ModuleList([ nn.Conv2d( self.config.net.convlayerunits[i],self.config.net.convlayerunits[i+1],self.config.net.convkernelsize[i]).to(device)   for i in range(len(self.config.net.convlayerunits)-1)])
        # self.deconv3 = nn.ModuleList([ nn.ConvTranspose2d(   self.config.net.deconvlayerunits[i],self.config.net.deconvlayerunits[i+1],self.config.net.deconvkernelsize[i] ).to(device)   for i in range(len(self.config.net.deconvlayerunits)-1)])

        print('___________________________-')
        print(self.conv)
        print('___________________________-')
        print(self.deconv)
        print()

        for i in range(len(self.config.net.convlayerunits)-1):
            nn.init.xavier_uniform_(self.conv[i].weight)
            nn.init.xavier_uniform_(self.conv1[i].weight)
            # nn.init.xavier_uniform_(self.conv2[i].weight)
            # nn.init.xavier_uniform_(self.conv3[i].weight)
            self.conv[i].bias.data.fill_(0.0)
            self.conv1[i].bias.data.fill_(0.0)
            # self.conv2[i].bias.data.fill_(0.0)
            # self.conv3[i].bias.data.fill_(0.0)

        for i in range(len(self.config.net.deconvlayerunits)-1):
            nn.init.xavier_uniform_(self.deconv[i].weight)
            nn.init.xavier_uniform_(self.deconv1[i].weight)
            # nn.init.xavier_uniform_(self.deconv2[i].weight)
            # nn.init.xavier_uniform_(self.deconv3[i].weight)
            self.deconv[i].bias.data.fill_(0.0)
            self.deconv1[i].bias.data.fill_(0.0)
            # self.deconv2[i].bias.data.fill_(0.0)
            # self.deconv3[i].bias.data.fill_(0.0)
            

        # self.pool = nn.MaxPool2d(2, 2).to(device)

        # self.dropoutHidden = nn.Dropout(self.config.train.dropoutratehidden)
        # self.dropoutVisible = nn.Dropout(self.config.train.dropoutratevisible)

    def forward(self,x, mulfactor=9 ):

        x = x [:,None,:,:]

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


        x = x.view(-1, self.config.num_agents, self.config.num_items)
        x = x [:,None,:,:]

        for i in range(len(self.config.net.convlayerunits)-1):
            x = self.activation[self.config.net.convactivation[i]](self.conv1[i](x))
        for i in range(len(self.config.net.deconvlayerunits)-1):
            x = self.activation[self.config.net.deconvactivation[i]](self.deconv1[i](x))

        x = x.view(-1, self.config.num_agents, self.config.num_items)
        x = x [:,None,:,:]

        # for i in range(len(self.config.net.convlayerunits)-1):
        #     x = self.activation[self.config.net.convactivation[i]](self.conv2[i](x))
        # for i in range(len(self.config.net.deconvlayerunits)-1):
        #     x = self.activation[self.config.net.deconvactivation[i]](self.deconv2[i](x))

        # x = x.view(-1, self.config.num_agents, self.config.num_items)
        # x = x [:,None,:,:]

        # for i in range(len(self.config.net.convlayerunits)-1):
        #     x = self.activation[self.config.net.convactivation[i]](self.conv3[i](x))
        # for i in range(len(self.config.net.deconvlayerunits)-1):
        #     x = self.activation[self.config.net.deconvactivation[i]](self.deconv3[i](x))

            
        x = x.view(-1, self.config.num_agents, self.config.num_items)

        # if(self.config.valuesmul==1):
        #     permvalue = torch.randperm(self.config.num_agents)
        #     values=self.values[permvalue]
        #     x = x * values


        if(self.config.softmax == 1):
            x =  self.softmaxfn(x)

        allocation = x * mulfactor

        allocation = self.softmaxfn(allocation)
        # allocation = self.sigmoid(allocation)


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



