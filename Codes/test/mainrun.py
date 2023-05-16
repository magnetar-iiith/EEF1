import sys
import numpy as np
import logging

from net import  EFNN6 , EFNN7 , EFNN6_1 , EFNN5

import json
from easydict import EasyDict as edict
import logging
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

from utils import util, trainer
from test.testing import Testing

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

from easydict import EasyDict as edict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


nn_name = sys.argv[1]
config_file_name =  sys.argv[2]
trialno =  sys.argv[3]
reloadnn = sys.argv[4]
logger_file_name = nn_name + '_' +  config_file_name +  '_' + trialno 

print('nn_name')
print(nn_name)

if(reloadnn == 'Test'):
    batchindex = int(sys.argv[5])
    batchsize = int(sys.argv[6])
    print('batchindex', batchindex)

    config_path = 'config/test/' + config_file_name

    with open(config_path) as json_file:
        cfg = json.load(json_file)
        cfg =  edict(cfg)

    netList = []
    cfglist = []

    for netindex,net_name in enumerate(cfg.pretrainnednet):
        print('net name ', net_name)
        print('..',cfg.indivconfig[netindex])
        config_path = 'config/indiv/' + cfg.indivconfig[netindex]


        with open(config_path) as json_file:
            cfg1 = json.load(json_file)
            cfglist.append(edict(cfg1))



    for net_index,net_name in enumerate(cfg.pretrainnednet):
        model_path_ = 'models/test/' + net_name
        nn_name = net_name[:5]
        print(nn_name)

        if nn_name == "EFNN5":
            print('EFNN5')
            currentnet = EFNN5.EFNN5(cfg,logger_file_name)
        if nn_name == "EFNN6":
            print('EFNN6')
            currentnet = EFNN6.EFNN6(cfg,logger_file_name)
        elif nn_name == "EFNN6_1":
            currentnet = EFNN6_1.EFNN6_1(cfg,logger_file_name)
        elif nn_name == "EFNN7":
            print('EFNN7')
            currentnet = EFNN7.EFNN7(cfglist[net_index],logger_file_name)

        state_dict = torch.load(model_path_)
        print()
        # print('state dict ')
        # print(state_dict)
        print()
        currentnet.load_state_dict(state_dict)
        netList.append(currentnet)

    t = Testing(netList, cfg,logger_file_name,batchindex,batchsize)
    t.testTheNetwork()
elif(reloadnn == 'Reload'):
    config_path = 'config/train/' + config_file_name
    with open(config_path) as json_file:
        cfg = json.load(json_file)
        cfg =  edict(cfg)

    if nn_name == "EFNN5":
        net = EFNN5.EFNN5(cfg,logger_file_name)
    elif nn_name == "EFNN6":
        net = EFNN6.EFNN6(cfg,logger_file_name)
    elif nn_name == "EFNN6_1":
        net = EFNN6_1.EFNN6_1(cfg,logger_file_name)
    elif nn_name == "EFNN7":
        net = EFNN7.EFNN7(cfg,logger_file_name)

    model_path_ = 'models/reload/' + logger_file_name + '.pth'

    state_dict = torch.load(model_path_)
    net.load_state_dict(state_dict)

    t = trainer.Trainer(net, cfg,logger_file_name)
    t.runTheNetwork()

else:
    config_path = 'config/train/' + config_file_name

    with open(config_path) as json_file:
        cfg = json.load(json_file)
        cfg =  edict(cfg)

    if nn_name == "EFNN5":
        net = EFNN5.EFNN5(cfg,logger_file_name)
    elif nn_name == "EFNN6":
        net = EFNN6.EFNN6(cfg,logger_file_name)
    elif nn_name == "EFNN6_1":
        net = EFNN6_1.EFNN6_1(cfg,logger_file_name)
    elif nn_name == "EFNN7":
        net = EFNN7.EFNN7(cfg,logger_file_name)

    t = trainer.Trainer(net, cfg,logger_file_name)
    t.runTheNetwork()
    # net.runTheNetwork()
    # torch.save(net.state_dict(), model_path)




    
