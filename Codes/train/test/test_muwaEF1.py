# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join('', 'utils')))
sys.path.append(os.path.abspath(os.path.join('', '')))

from utils.util import Util
from utils.efficiencyutil import EfficiencyUtil 
from utils.fairnessutil import FairnessUtil

## Imports
import torch

import os
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc
from torch import nn, optim
import torch.nn.functional as F
from random import shuffle
import time
import logging
logging.getLogger('').setLevel(logging.DEBUG)
logger1 = logging.getLogger('1')

from easydict import EasyDict as edict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cfg = edict({})
config = cfg
util =  Util(config,'test')

efficiencyUtil = EfficiencyUtil(cfg)
fairnessUtil = FairnessUtil(cfg)





n_samples = 20000
cfg.num_agents = 10
cfg.valuationtype = 'additive'
# for numberin in range(160,90,-10):
for numberin in range(20,31,5):

    print(numberin)
    cfg.num_items = numberin
    n_items = cfg.num_items 
    n_agents = cfg.num_agents

    ##Uniform
    X_test = np.random.rand(*[n_samples,n_agents,n_items])
    X_test = torch.from_numpy(X_test).float()
    X_test = X_test.to(device)

    print(X_test.shape)
    muwallocation = util.findMUW_all(X_test)
    muwsumenvy, muwcurrentenvy = util.calculateEnvySamplewise(muwallocation,X_test)
    muwcurrentsw = util.calculate_u_social_welfare(muwallocation,X_test)
    muwenvyindex = torch.nonzero(muwcurrentenvy)
    count =  n_samples - len(muwenvyindex)
    print('___________________________________________')
    print(' Total Envy Free samples ', count)
    print(' Total Envy Free samples percent', (count/n_samples))
    print(' USW ', torch.mean(muwcurrentsw))
    print()
    print('MUW.....done')


    greedyallocation = util.ef1envyfree_greedy_output(X_test)
    # greedysumenvy, greedycurrentenvy = calculateEnvySamplewise(greedyallocation,X_test)
    greedycurrentsw = util.calculate_u_social_welfare(greedyallocation,X_test)
    # greedyenvyindex = torch.nonzero(greedysumenvy)
    # count =  n_samples - len(greedyenvyindex)
    # print('___________________________________________')
    # print(' Total Envy Free samples ', count)
    # print(' Total Envy Free samples percent', (count/n_samples))
    # okayindex = torch.zero(muwcurrentsw - greedycurrentsw)
    okayindex = (muwcurrentsw == greedycurrentsw).nonzero() 
    print('__________________________________',)
    print(muwcurrentsw[okayindex])
    print(greedycurrentsw[okayindex])
    count__ =  n_samples - len(okayindex)
    print('...', count)
    print('__________________________________',)

    print(' Greedy ', torch.mean(greedycurrentsw))
    print()
    print('Greedy.....done')


    wcrrallocation = util.findWCRR_all(X_test)
    # wcrrsumenvy, wcrrcurrentenvy = calculateEnvySamplewise(wcrrallocation,X_test)
    wcrrcurrentsw = util.calculate_u_social_welfare(wcrrallocation,X_test)
    # wcrrenvyindex = torch.nonzero(wcrrsumenvy)
    # count =  n_samples - len(wcrrenvyindex)
    # print('___________________________________________')
    # print(' Total Envy Free samples ', count)
    # print(' Total Envy Free samples percent', (count/n_samples))
    okayindex = torch.nonzero(muwcurrentsw - wcrrcurrentsw)
    print('__________________________________',)
    print(muwcurrentsw[okayindex])
    print(wcrrcurrentsw[okayindex])
    count__ =  n_samples - len(okayindex)
    print('...', count)
    print('__________________________________',)

    print(' WCRR ', torch.mean(wcrrcurrentsw))
    print()
    print('WCRR.....done')



