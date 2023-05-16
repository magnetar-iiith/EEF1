# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

import os,sys
sys.path.append(os.path.abspath(os.path.join('', 'utils')))
sys.path.append(os.path.abspath(os.path.join('', '')))

from utils.util import Util

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

import statistics 

from easydict import EasyDict as edict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Testing():

    def __init__(self, netList, config,logger_file_name,currentindex = 0,batchszie=1000):
        self.batchszie = batchszie
        self.currentindex = currentindex
        self.cfg = config
        self.logger_file_name = logger_file_name
        self.netList = netList 
        self.util =  Util(config,logger_file_name)


    def testTheNetwork(self):

        time_elapsed = 0
        print('Pretrainned net')
        print(self.cfg.pretrainnednet)
        print()

        num_test_batches = self.cfg.num_test_batches

        # currentindex = 0
        # batchszie = 600
        currentrange = range(self.currentindex*self.batchszie , (self.currentindex+1)*self.batchszie)
        print(currentrange)
        
        # currentrange = range(0,1000)
        # currentrange = range(1000,2000)
        # currentrange = range(2000,3000)
        # currentrange = range(3000,4000)
        # currentrange = range(4000,5000)
        # currentrange = range(6000,7000)
        # currentrange = range(7000,8000)
        # currentrange = range(8000,9000)
        # currentrange = range(9000,10000)
        
        self.cfg.num_agents = 7
        print(self.cfg.num_agents)
        for numberin in range(135,180,3):
        # for numberin in range(1,2,2):
            self.cfg.num_items = numberin
            print(self.cfg.num_items)
            print(numberin)
            # print('------')
            # print('1000')

            # print(self.cfg.train.dropout)
            for i in range(num_test_batches):
                # n_samples = 10000

                if(True):
                    n_samples = 5000
                    test_data_shape = [n_samples , self.cfg.num_agents , self.cfg.num_items]
                    X_test = self.util.generate_test_data(test_data_shape,self.cfg.valuationDistribution, self.cfg.valuationtype , self.cfg.items)

                # print(test_data_shape)
                # # X_test = self.util.generate_test_data(test_data_shape,'dirichlet', self.cfg.valuationtype , self.cfg.items)
                # X_test = self.util.generate_test_data(test_data_shape,self.cfg.valuationDistribution, self.cfg.valuationtype , self.cfg.items)
                # print('lognormal')
                # X_test = np.random.lognormal(mean=0.5, sigma=1.0, size=test_data_shape)
                
                # print('wxp')
                # X_test = np.random.exponential(scale=1.0, size=test_data_shape)

                # mu, sigma = 0.5, 1 # mean and standard deviation
                # mu, sigma = 0.5, 0.1 # mean and standard deviation
                # X_test = np.random.normal(mu, sigma, test_data_shape)

                # X_test = (X_test - 0.5) / 1

                # X_test = torch.from_numpy(X_test).float()
                # X_test = X_test.to(device)    
                
                # filename = 'data/X_testGauss' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                # print('guass')
                # filename = 'data/X_testExp' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                # print('exp')
                # filename = 'data/X_testLognormal' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                # print('lognormal')
                    # filename = 'data/X_test' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                    # torch.save(X_test, filename)
                # if(numberin == 32):
                #     X_test = self.util.generate_test_data(test_data_shape,self.cfg.valuationDistribution, self.cfg.valuationtype , self.cfg.items)
                #     torch.save(X_test, filename)
                # # else:            
                # X_test = torch.load(filename)
                # X_test = X_test[currentrange]


                # print("Mean of the sample is % s " %(statistics.mean(X_test))) 
                # print("Standard Deviation of the sample is % s "%(statistics.stdev(X_test)))
                # print("Mean of the sample is % s " %(statistics.mean(X_test))) 

                if(True):
                    # tic = time.time()              
                    # countRandom, aEf1Random , randomsw , randomallocation= self.util.calculateaEF1ForWCRR(X_test)
                    # toc = time.time()
                    # time_elapsed = (toc - tic)
                    # print('wcrr.....done ', time_elapsed )
                    # print('countMUW', countRandom)
                    # print('aEF1MUW' , aEf1Random)
                    # # print('Random SW', randomsw )


                    tic = time.time()              
                    countRandom, aEf1Random , randomsw , randomallocation= self.util.calculateaEF1ForRR(X_test)
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    print('greedy.....done ', time_elapsed )
                    print('countMUW', countRandom)
                    print('aEF1MUW' , aEf1Random)
                    # print('Random SW', randomsw )

                    tic = time.time()              
                    countMUW, aEf1MUW , muwsw , muwallocation= self.util.calculateaEF1ForMUW(X_test)
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    print('MUW.....done ', time_elapsed )
                    print('countMUW', countMUW)
                    print('aEF1MUW' , aEf1MUW)



                if(False):
                    tic = time.time()              
                    countMUW, aEf1MUW , muwsw , muwallocation= self.util.calculateaEF1ForMUW(X_test)
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    print('MUW.....done ', time_elapsed )
                    print('countMUW', countMUW)
                    print('aEF1MUW' , aEf1MUW)
                    # print('MUW sw', muwsw)

                    XY = X_test * muwallocation
                    XY = XY.float()

                    # # # m = 21
                    layersnum = 5
                    # num_items = 
                    eachlayerm = self.cfg.num_items//layersnum
                    print(eachlayerm)

                    layersss = []
                    for k in range(layersnum-1):
                        agents = []
                        for j in range(eachlayerm):
                            agents.append(k+j*(layersnum))
                        layersss.append(agents)
                    print(layersss)
                    layers = []

                    zerostensor = torch.zeros(muwallocation.shape).to(device)
                    print('......',zerostensor.shape)
                    for i in range(layersnum-1):
                        layers.append(zerostensor.clone().detach())
                        if(i != 0):
                            layers[i] = layers[i-1].clone().detach()

                        for k in layersss[i]:
                            layers[i][:,:,k] = XY[:,:,k].clone().detach()

                    print('..layers', len(layers))
                    layers.append(XY.clone().detach())
                    print('..layers', len(layers))
                    layers  = torch.stack(layers)
                    print(layers.shape)
                    Y = layers.reshape(X_test.shape[0],layersnum,self.cfg.num_agents,self.cfg.num_items)

                    # # filename = 'data/Y_testGuass' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                    # # filename = 'data/Y_testExp' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                    # # filename = 'data/Y_testLognormal' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'

                    # filename = 'data/Y_test' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                    # torch.save(Y,filename)
                    # Y = torch.load(filename)
                    # # Y = []
                    Y=Y[currentrange]

                    # # print('jjjjj',XY.shape)
                    # # tic = time.time() 

                # filename = 'data/Y_test' + str(self.cfg.num_agents) + '_' + str(self.cfg.num_items) + '.pt'
                # # torch.save(Y,filename)
                # Y = torch.load(filename)
                # # # # # # # Y = []
                # Y=Y[currentrange]

                # self.util.getCombineAllocation(self.netList,X_test,Y)            
                
                # # combinedallocation , combineddiscreteallocation  , combinedenvy = self.util.getCombineAllocation(self.netList,X_test,Y)            

                # # nn_allocation_discrete = combineddiscreteallocation.clone()
                # # nn_allocation = combinedallocation.clone()
                # # nn_sw = self.util.calculate_u_social_welfare(nn_allocation_discrete,X_test)





                # # print('Just checking for MUW neural network    ')
                # # print('Comment this later')
                # # print()
                # # # okayindex = torch.nonzero(muwsw - nn_sw)
                # # print('__________________________________',)
                # # # print(muwsw[okayindex])
                # # # print(wcrrcurrentsw[okayindex])
                # # count__ =  n_samples - len(okayindex)
                # # print('MUW = NN', count__ )
                # # print('MUW = NN', count__  / n_samples)


                # toCompareAllocation = {}
                # ####
                # ##### 1) NN
                # removeindex = torch.nonzero(combinedenvy)
                # removeindex = removeindex.reshape(removeindex.shape[0])
                # if(len(removeindex) !=0):
                #     totalindex = torch.tensor(range(n_samples))
                #     ef1index  = np.setxor1d(totalindex,removeindex.to('cpu'))
                #     nn_allocation  = nn_allocation[ef1index].clone()
                #     nn_allocation_discrete  = nn_allocation_discrete[ef1index].clone()
                #     X_test = X_test[ef1index].clone()
                # updatednumsamples = X_test.shape[0]
                # print('******************************************************')
                # print()
                # print('updatednumsamples', updatednumsamples)
                # print('******************************************************')
                # toCompareAllocation['nn'] = nn_allocation_discrete.clone()

                listnames = self.cfg.listnames
                # toc = time.time()
                # time_elapsed1 = (toc - tic)
                # print('NN allocation done ', time_elapsed1)
                    
                ####  2) Greedy
                if('greedy' in listnames):
                    tic = time.time()
                    print('greedy....')
                    tic1 = time.time()
                    greedyallocation = self.util.ef1envyfree_greedy_output(X_test)
                    toCompareAllocation['greedy'] = greedyallocation.clone()
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    greedysw = self.util.calculate_u_social_welfare(greedyallocation,X_test)
                    print(' USW ', torch.mean(greedysw))

                    print('greedy done ', time_elapsed1)


                ### 2) doubleRoundRobin_all
                if('doubleroundrobin' in listnames):
                    doubleRoundRobinalloc = self.util.doubleRoundRobin_all(X_test)
                    toCompareAllocation['doubleroundrobin'] = doubleRoundRobinalloc.clone()
                    # toCompareAllocation.append(doubleRoundRobinalloc)

                ####  3) MNW
                if('mnw' in listnames):
                    optimal_mnw_allocation = self.util.findMNW_all(allallocation, X_test)
                    optimalnwlist = self.util.calculate_n_social_welfare(optimal_mnw_allocation,X_test)
                    toCompareAllocation['mnw'] = optimal_mnw_allocation.clone()

                # ####  4) Barman 2018
                # if('barman 2018' in listnames):
                #     toCompareAllocation.append([])

                ####  5) Aziz 2019 
                if('wcrr' in listnames):
                    tic = time.time()
                    wcrrallocation = self.util.findWCRR_all(X_test)
                    toCompareAllocation['wcrr'] = wcrrallocation.clone()
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    wcrrsw = self.util.calculate_u_social_welfare(wcrrallocation,X_test)
                    print(' USW ', torch.mean(wcrrsw))
                    print('wcrr done ', time_elapsed1)
                    testsumenvy, testcurrentenvy = self.util.calculateEnvySamplewise(wcrrallocation,X_test)
                    testenvyindex = torch.nonzero(testcurrentenvy)
                    count =  wcrrallocation.shape[0] - len(testenvyindex)
                    print('WCRR EF1 count ', count)


                ###### 7 ) cyclice ele
                if('cyclic' in listnames):
                    tic = time.time()
                    cycliceliminationallocation = self.util.cylicElimination_all(X_test)
                    toCompareAllocation['cyclic'] = cycliceliminationallocation.clone()
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    print('cyclic done ', time_elapsed1)

                ####  6) Max SW in EF1  

                if('max sw in EF1' in listnames):
                    tic = time.time()
                    optimal_usw_allocation_in_EF1 = self.util.findMUWinEF1_all(allallocation, X_test)
                    optimaluwlist = self.util.calculate_u_social_welfare(optimal_usw_allocation_in_EF1,X_test)
                    toCompareAllocation['max sw in EF1'] = optimal_usw_allocation_in_EF1.clone()
                    toc = time.time()
                    time_elapsed = (toc - tic)
                    print('max sw in EF1 done ', time_elapsed1)

                ### To compare




