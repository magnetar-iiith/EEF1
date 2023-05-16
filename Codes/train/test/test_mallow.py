import profile_generator
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from scipy import stats
import itertools

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
cfg.num_agents = 10
cfg.valuationtype = 'additive'
cfg.num_items = 20

util =  Util(config,'test')

efficiencyUtil = EfficiencyUtil(cfg)
fairnessUtil = FairnessUtil(cfg)



num_agents = cfg.num_agents
num_items = cfg.num_items
n_samples = 10000

mixture = ['p70',0.1,0.9]
ps = np.arange(0.1,1,0.1)
p = ps[3]
agents = list(range(num_agents))


for numberin in range(20,51,5):

    print(numberin)
    print('_____________________')
    cfg.num_items = numberin
    num_items = cfg.num_items 
    num_agents = cfg.num_agents
    items = list(range(num_items))
    items_rev = items[::-1]


    valuation_profile = []
    allocation_profile = []
    for i in range(n_samples):
        profile = profile_generator.generate_mallows_mixture_profile(agents, items, [mixture[1], mixture[2]], [items, items_rev], [p, p])
        valuation = []
        for n_agent in profile.keys():
            # print(profile[n_agent])
            valuation.append((profile[n_agent]) )
        valuation = torch.FloatTensor(valuation).to(device)
        valuation = (valuation + 1)
        valuation = valuation/100
        allocation = util.findMUW(valuation)
        allocation_profile.append(allocation)
        valuation_profile.append(valuation)

    valuation_profile = torch.stack(valuation_profile)
    allocation_profile = torch.stack(allocation_profile)
    valuation_profile = valuation_profile.to(device)
    allocation_profile = allocation_profile.to(device)

    sumenvy, envy = util.calculateEnvySamplewise(allocation_profile,valuation_profile)
    sw = util.calculate_u_social_welfare(allocation_profile,valuation_profile)
    envyindex = torch.nonzero(envy)
    count =  n_samples - len(envyindex)
    print(' Total Envy Free upto one item samples ', count)
    print(' Total Envy Free upto one item samples percent', (count/n_samples))
    print(' USW ', torch.mean(sw))

    print()
    print('greedy')
    greedyallocation = util.ef1envyfree_greedy_output(valuation_profile)
    greedysumenvy, greedycurrentenvy = util.calculateEnvySamplewise(greedyallocation,valuation_profile)
    greedycurrentsw = util.calculate_u_social_welfare(greedyallocation,valuation_profile)
    print('grredy sw mean ', torch.mean(greedycurrentsw))
    print()
    greedyenvyindex = torch.nonzero(greedycurrentenvy)
    count =  n_samples - len(greedyenvyindex)
    print('___________________________________________')
    print(' Total Envy Free samples ', count)
    print(' Total Envy Free samples percent', (count/n_samples))
    # okayindex = torch.zero(muwcurrentsw - greedycurrentsw)
    # okayindex = (muwcurrentsw == greedycurrentsw).nonzero() 
    # print('__________________________________',)
    # print(muwcurrentsw[okayindex])
    # print(greedycurrentsw[okayindex])
    # count__ =  n_samples - len(okayindex)
    # print('...', count)
    # print('__________________________________',)
    # print()


    wcrrallocation = util.findWCRR_all(valuation_profile)
    wcrrsumenvy, wcrrcurrentenvy = util.calculateEnvySamplewise(wcrrallocation,valuation_profile)
    wcrrcurrentsw = util.calculate_u_social_welfare(wcrrallocation,valuation_profile)
    wcrrenvyindex = torch.nonzero(wcrrcurrentenvy)
    count =  n_samples - len(wcrrenvyindex)
    print('___________________________________________')
    print(' Total Envy Free samples ', count)
    print(' Total Envy Free samples percent', (count/n_samples))
    print(' WCRR ', torch.mean(wcrrcurrentsw))
    print()
    print('WCRR.....done')

    print()
    print()
    # borda_ranking = [k for k in sorted(inverse_borda_scores, key=inverse_borda_scores.__getitem__)]

    # print(borda_ranking)

