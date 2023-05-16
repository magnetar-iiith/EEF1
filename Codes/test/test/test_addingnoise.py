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
# from tqdm import tqdm
# from pylab import rcParams
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

cfg.num_agents = 10
cfg.num_items = 200
cfg.valuationtype = 'additive'
n_items = cfg.num_items 
n_agents = cfg.num_agents
tic = time.time()      
time_elapsed = 0


n_samples = 10000
noisequotient = 0.01
# distributiontype = 'Dirichlet'
distributiontype = 'Uniform'
noiseType = 'Uniform'
# noiseType = 'Normal'

concentrationParameter = 100

if(distributiontype == 'Uniform'):
    #Uniform
    X_test_all = np.random.rand(*[n_samples,n_agents,n_items])
    X_test_all = torch.from_numpy(X_test_all).float()
elif(distributiontype == 'Dirichlet'):
    print('Concentration param', concentrationParameter)
    concentration = np.ones(n_items) * concentrationParameter
    X_test_all = np.random.dirichlet(concentration, (n_samples,n_agents))
    X_test_all = torch.from_numpy(X_test_all).float()
X_test_all = X_test_all.to(device)




print('Distribution  ',distributiontype)
# print('Concentration 100')
print('________________________________________________')
print('Noise type', noiseType)
print('Noise quotient', noisequotient)
print('num_items', n_items)
print('________________________________________________')

count_list_withzero = []
redflagcount = 0
difference_1_muw = []
difference_2_rr = []
difference_3_wcrr = []


muwsw = []
muwsw_1 = []
greedy = []
wcrr = []

print(X_test_all.shape)
for X_test in X_test_all:
    muwallocation = util.findMUW(X_test)
    muwcurrentsw = torch.sum(muwallocation*X_test)
    muwcurrentsw_1 = torch.sum(muwallocation*X_test)

    isEF1 = fairnessUtil.isSampleEF1Free(muwallocation,X_test)
    count = 0
    while (not isEF1):
        if(noiseType == 'Uniform'):
            # noise = np.random.normal(-noisequotient, noisequotient, X_test.shape)
            noise = np.random.uniform(-noisequotient, noisequotient, X_test.shape)
            noise = torch.from_numpy(noise).float()
            noise = noise.to(device)
        elif(noiseType == 'Normal'):
            noise = np.random.normal(0, noisequotient, X_test.shape)
            noise = torch.from_numpy(noise).float()
            noise = noise.to(device)
            

        X_test_1 = X_test + noise

        muwallocation_1 = util.findMUW(X_test_1)
        isEF1 = fairnessUtil.isSampleEF1Free(muwallocation_1,X_test)
        muwcurrentsw_1 = torch.sum(muwallocation_1*X_test)
        count += 1 

        if(count > 50):
            redflagcount += 1
            print('Red FLAG!!!!')
            break
    
    if(count > 0 ):
        greedyallocation = efficiencyUtil.greedy_EF1Envifree_allocation(X_test)
        greedycurrentsw = torch.sum(greedyallocation*X_test)

        wcrrallocation = efficiencyUtil.findWCRR(X_test)
        wcrrcurrentsw = torch.sum(wcrrallocation*X_test)

        difference_1_muw.append(muwcurrentsw - muwcurrentsw_1)
        difference_2_rr.append(muwcurrentsw_1 - greedycurrentsw)
        difference_3_wcrr.append(muwcurrentsw_1 - wcrrcurrentsw)
        
        print('muwcurrentsw_1   ', muwcurrentsw_1)
        print('greedycurrentsw  ', greedycurrentsw )
        print(' wcrr sw         ', wcrrcurrentsw)

        muwsw.append(muwcurrentsw)
        muwsw_1.append(muwcurrentsw_1)
        greedy.append(greedycurrentsw)
        wcrr.append(wcrrcurrentsw)


        
    count_list_withzero.append(count)
    print('Count ', count)
    print('muwcurrentsw     ', muwcurrentsw)


toc = time.time()
time_elapsed += (toc - tic)

summary = "t ={:.4f}"
print(summary.format(time_elapsed))
  
uniquecounts = torch.unique(torch.tensor(count_list_withzero), return_counts=True)

print(uniquecounts)
print()
print('mean', sum(difference_1_muw) / len(difference_1_muw) )
print('min', min(difference_1_muw))
print('max', max(difference_1_muw))
print()
print()
print('mean', sum(difference_2_rr) / len(difference_2_rr) )
print('min', min(difference_2_rr))
print('max', max(difference_2_rr))
print()
print('mean', sum(difference_3_wcrr) / len(difference_3_wcrr) )
print('min', min(difference_3_wcrr))
print('max', max(difference_3_wcrr))
print('...................')
print('mean muwcurrentsw', sum(muwsw) / len(muwsw) )
print('mean muwcurrentsw_1', sum(muwsw_1) / len(muwsw_1) )
print('mean greedycurrentsw', sum(greedy) / len(greedy) )
print('mean wcrr ', sum(wcrr) / len(wcrr) )
print('')
