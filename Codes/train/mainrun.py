import sys
import numpy as np
import logging

from net import  EFNN6 , EFNN7 , EFNN6_1 , EFNN5,EFNN7onelayer,EFNN7zerolayer,EFNN7threelayer

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
# import seaborn as sns
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
    config_path = 'config/test/' + config_file_name

    with open(config_path) as json_file:
        cfg = json.load(json_file)
        cfg =  edict(cfg)

    netList = []
    for net_name in cfg.pretrainnednet:
        print('net name ', net_name)

    for net_name in cfg.pretrainnednet:
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
            currentnet = EFNN7.EFNN7(cfg,logger_file_name)
        elif nn_name == "EFNN7onelayer":
            print('EFNN7-onelayer')
            currentnet = EFNN7onelayer.EFNN7onelayer(cfg,logger_file_name)
        elif nn_name == "EFNN7zerolayer":
            print('EFNN7-onelayer')
            currentnet = EFNN7zerolayer.EFNN7zerolayer(cfg,logger_file_name)
        elif nn_name == "EFNN7threelayer":
            print('EFNN7-threelayer')
            currentnet = EFNN7threelayer.EFNN7threelayer(cfg,logger_file_name)


        state_dict = torch.load(model_path_)
        print()
       
        print()
        currentnet.load_state_dict(state_dict)
        netList.append(currentnet)

    t = Testing(netList, cfg,logger_file_name)
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
    elif nn_name == "EFNN7onelayer":
        print('EFNN7-onelayer')
        net = EFNN7onelayer.EFNN7onelayer(cfg,logger_file_name)
    elif nn_name == "EFNN7zerolayer":
        print('EFNN7-zerolayer')
        net = EFNN7zerolayer.EFNN7zerolayer(cfg,logger_file_name)
    elif nn_name == "EFNN7threelayer":
        print('EFNN7-threelayer')
        net = EFNN7threelayer.EFNN7threelayer(cfg,logger_file_name)

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
    elif nn_name == "EFNN7onelayer":
        print('EFNN7-onelayer')
        net = EFNN7onelayer.EFNN7onelayer(cfg,logger_file_name)
    elif nn_name == "EFNN7zerolayer":
        print('EFNN7-zerolayer')
        net = EFNN7zerolayer.EFNN7zerolayer(cfg,logger_file_name)
    elif nn_name == "EFNN7threelayer":
        print('EFNN7-threelayer')
        net = EFNN7threelayer.EFNN7threelayer(cfg,logger_file_name)

    

    t = trainer.Trainer(net, cfg,logger_file_name)
    t.runTheNetwork()
    # net.runTheNetwork()
    # torch.save(net.state_dict(), model_path)




    
