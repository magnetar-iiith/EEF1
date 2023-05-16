
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
from itertools import combinations 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EfficiencyUtil():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.softmaxfn = nn.Softmax(dim=1).to(device)
        self.relufn = nn.ReLU()

    def calculate_u_social_welfare(self,allocation, valuation):
        if (self.cfg.valuationtype == 'additive'):
            return self.calculate_u_social_welfare_additive(allocation,valuation)
        elif (self.cfg.valuationtype == 'general'):
            return self.calculate_u_social_welfare_general(allocation,valuation)

    def calculate_u_social_welfare_additive(self,allocation , valuation):
        ## allocation : num_samples x num_agents x num_items
        ## valuation : num_samples x num_agents x num_items
        sw = allocation * valuation
        sw = torch.sum( sw, (1, 2))  
        return sw


    ## x = n_Samples x n_Agents x n_items
    ## 
    def findMUW_all(self,x):
        maxswallocation, indices = torch.max(x , dim=1)
        allocation_ = torch.zeros(x.shape).to(device)
        for n_sample in range(indices.shape[0]):
            for n_item in range(indices.shape[1]):
                agentid = indices[n_sample][n_item]
                allocation_[n_sample][agentid][n_item] =1
        # X_test = torch.from_numpy(X_test).float()

        return allocation_.double()

    def findMUW(self,x):
        maxswallocation, indices = torch.max(x , dim=0)
        n_items = indices.shape[0]
        allocation_ = torch.zeros(x.shape).to(device)
        for n_item in range(n_items):
            agentid = indices[n_item]
            allocation_[agentid][n_item] =1 
        return allocation_

    def findRandomAllocation_all(self,x):
        n_samples = x.shape[0]
        n_agents = x.shape[1]
        n_items = x.shape[2]
        # maxswallocation, indices = torch.max(x , dim=1)
        allocation_ = torch.zeros(x.shape).to(device)
        for n_sample in range(n_samples):
            for n_item in range(n_items):
                # agentid = indices[n_sample][n_item]
                agentid = np.random.randint(n_agents)
                allocation_[n_sample][agentid][n_item] =1
        # X_test = torch.from_numpy(X_test).float()
        return allocation_.double()


    def findRandomAllocation(self,x):
        # maxswallocation, indices = torch.max(x , dim=0)
        n_items = x.shape[1]
        n_agents = x.shape[0]
        allocation_ = torch.zeros(x.shape).to(device)
        for n_item in range(n_items):
            agentid = np.random.randint(n_agents)
            allocation_[agentid][n_item] =1 
        return allocation_


    def ef1envyfree_greedy_output(self,x):
        allocation_output = []
        for valuation in x:
            a = self.greedy_EF1Envifree_allocation(valuation)
            allocation_output.append(a)
        return torch.stack(allocation_output)


    def greedy_EF1Envifree_allocation(self,valuation):
        num_agents = valuation.shape[0]
        num_items = valuation.shape[1]
        sorted, val_indices = torch.sort(valuation, descending=True)
        ## To maintain which item are remaining
        unallocated_arr = torch.zeros(num_items,  dtype=torch.int).to(device)
        num_unallocated_items = num_items
        count = torch.zeros(num_agents, dtype=torch.int).to(device)
        x_allocation = torch.zeros((num_agents , num_items) , dtype=torch.int).to(device)

        while (num_unallocated_items > 0):
            ## Order is fixed
            for i in range(num_agents):
                if(num_unallocated_items != 0):
                    ## Finding index of next highest valuated unallocated item 
                        item_index = val_indices[i][count[i]]
                        while (unallocated_arr[item_index] == 1):
                            count[i] += 1
                            item_index = val_indices[i][count[i]]
                        
                        x_allocation[i][item_index] = 1
                        unallocated_arr[item_index] = 1
                        num_unallocated_items -= 1
        return torch.tensor(x_allocation, dtype=torch.float).to(device)


    def doubleRoundRobin(self,valuation):
        num_agents = valuation.shape[0]
        num_items = valuation.shape[1]
        chores = []
        ## Find First chores - which all agents consider chores
        for j in range(num_items):
            count = 0
            for i in range(num_agents):
                    if(valuation[i][j]  < 0):
                        count += 1
            if(count == num_agents):
                chores.append(j)
    
        # print('chores...', chores)
        sorted, val_indices = torch.sort(valuation, descending=True)

        choresvaluation = valuation[:,chores].clone()
        choresvaluationsorted , choresval_index = torch.sort(choresvaluation, descending=True)
    
        unallocated_arr = torch.zeros(num_items,  dtype=torch.int).to(device)
        num_unallocated_items = num_items
        ## To maintain the last item they received + 1 , =>  next item 
        count = torch.zeros(num_agents, dtype=torch.int).to(device)

        x_allocation = torch.zeros((num_agents , num_items) , dtype=torch.int).to(device)

        
        num_chores = len(chores)
        count_chores = torch.zeros(num_agents, dtype=torch.int).to(device)

        i = 0
        while(num_chores > 0):
            for i in range(num_agents):
                    # print('i', i)
                    item_index = choresval_index[i][count_chores[i]]
                    chore_index_index = chores[item_index]
                    while (unallocated_arr[chore_index_index] == 1):
                        count_chores[i] += 1
                        item_index = choresval_index[i][count_chores[i]]
                        chore_index_index = chores[item_index]
                    x_allocation[i][chore_index_index] = 1
                    unallocated_arr[chore_index_index] = 1
                    num_unallocated_items -= 1
                    num_chores -= 1

                    if(num_chores==0):
                        # print('i:: ',i)
                        break
        # print(' i is', i)
        # print(' x_allocation', x_allocation.clone())

        if(num_unallocated_items == 0):
            return torch.tensor(x_allocation, dtype=torch.float).to(device)


        while (num_unallocated_items > 0):
        ## Order is fixed
            while(i < num_agents):
            # print('i combo', i)
                if(num_unallocated_items != 0):
                    ## Finding index of next highest valuated unallocated item 
                    item_index = val_indices[i][count[i]]
                    while (unallocated_arr[item_index] == 1):
                        count[i] += 1
                        item_index = val_indices[i][count[i]]
                    
                    if(valuation[i,item_index]>0):            
                        x_allocation[i][item_index] = 1
                        unallocated_arr[item_index] = 1
                        num_unallocated_items -= 1

                    i += 1
                    if(i == num_agents):
                        i = 0
                else:
                    # print('okayokay', x_allocation)
                    return torch.tensor(x_allocation, dtype=torch.float).to(device)



    def doubleroundrobin_all(self,x):
        allocation_output = []
        for valuation in x:
            a = self.doubleRoundRobin(valuation)
            allocation_output.append(a)
        return torch.stack(allocation_output)



    def findWCRR(self,valuation):
        n_agents = valuation.shape[0]
        n_items = valuation.shape[1]
        remaining_items = n_items
        remaining_agents = n_agents

        smallnumber = -99999999
        valuationclone = valuation.clone()
        allocation = torch.zeros(valuationclone.shape).to(device)

        while (remaining_items > 0):
            modifiedvaluation0 = valuationclone.clone()
            remaining_agents = n_agents
            while (remaining_agents > 0):
                if (remaining_items > 0):

                    maxitem_ , maxitemidx= torch.max(modifiedvaluation0,dim=1)
                    maxitemholdingagent, maxagentidx = torch.max(maxitem_, dim=0 )

                    agentno = maxagentidx
                    itemno = maxitemidx[agentno]
                    allocation[maxagentidx][itemno] = 1


                    valuationclone[:,itemno] = smallnumber
                    modifiedvaluation0[:,itemno] = smallnumber
                    modifiedvaluation0[agentno,:] = smallnumber

                    remaining_agents -= 1
                    remaining_items -= 1

                else:
                    remaining_agents = 0

        return allocation

    def findWCRR_all(self,valuation_profile ):
        allocations = []
        n_samples = valuation_profile.shape[0]
        for i in range(n_samples):
            allocations.append(self.findWCRR(valuation_profile[i]))
        return torch.stack(allocations)


    def calculate_nash_social_welfare(self,allocation , valuation):
        ## allocation : num_samples x num_agents x num_items
        ## valuation : num_samples x num_agents x num_items
        bundlereceivedvaluation = allocation * valuation
        utilities = torch.sum( bundlereceivedvaluation, dim=2)

        n_agents = allocation.shape[1]
        nashsw = 1
        for i in range(n_agents):
            nashsw *=  utilities[:,i]
        ###Take the value from config
        # nashsw = nashsw**(0.5)
        return nashsw

