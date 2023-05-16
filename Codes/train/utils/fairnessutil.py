
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
from itertools import combinations 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FairnessUtil():

    def __init__(self, cfg):
        self.cfg = cfg
        self.softmaxfn = nn.Softmax(dim=1).to(device)
        self.relufn = nn.ReLU()

    # def isSampleEF1Free(self,allocation_profile,valuation_profile):
    #     if(self.cfg.valuationtype == 'additive'):
    #         return self.isSampleEF1Free_additive(allocation_profile,valuation_profile)
    #     elif(cfg.valuationtype == 'general'):
    #         return self.isSampleEF1Free_general(allocation_profile,valuation_profile)
    
    def isSampleEF1Free(self,allocation_profile,valuation_profile):
        n_agents = allocation_profile.shape[0]
        n_items = allocation_profile.shape[1]

        sumenvy = 0

        ## V_i A_j for all j
        multiply_vi_aj =  valuation_profile[:,None,:] * allocation_profile 

        ## V_i A_i
        multiply_vi_ai = valuation_profile * allocation_profile

        ## Max item in jth bundle according to agent i
        ## dim = 1, inner most dim, i.e. => choosing the most valued item 
        ## n_agents, n_items
        maxitemfori_in_jthbundle, temp = torch.max(multiply_vi_aj, 2)

        ## social welfare calculation for agent i corresponding to its own bundle and other
        ## agent's bundle
        sw_i_j = torch.sum(multiply_vi_aj , dim=2)
        sw_i = torch.sum(multiply_vi_ai , dim=1)

        ## substraction of max valued  from social welfare calculating using other agent's bundle
        sw_i_j_minus_maxitem = sw_i_j - maxitemfori_in_jthbundle

        ## Min item in ith bundle, remove from its bundle
        ## i.e. when chores are there, we can remove a chore from agent i bundle
        ## if they are envying
        vi_min_item, k = torch.min(multiply_vi_ai , dim=1)
        sw_i_afterremovingminitem = sw_i - vi_min_item

        ### Envy is calculated as the minimum envy by either removing item from other agent
        ### or by removing item from agent i bundle 
        envy = torch.min( sw_i_j_minus_maxitem -  torch.reshape(sw_i, (n_agents,1)) , sw_i_j - torch.reshape(sw_i_afterremovingminitem, (n_agents,1)) )

        ## filter out and only consider the positive envy value
        # envy =  self.relufn(envy)
        envy[envy < 0] = 0
        sumenvy += torch.sum(envy)

        if (sumenvy == 0):
            return True
        else:
            return False

    def findEF1envy_i(self,allocation_profile,valuation_profile,i=0,trainMode=True):
        valuation_i = valuation_profile[:,i,:]
        allocation_i = allocation_profile[:,i,:]
        n_samples = allocation_profile.shape[0]
        n_agents = allocation_profile.shape[1]
        n_items = allocation_profile.shape[2]

        ## V_i A_j for all j
        multiply_vi_aj =  valuation_i[:,None,:] * allocation_profile 
        ## V_i A_i
        multiply_vi_ai = valuation_i * allocation_i
        ## Max item in jth bundle according to agent i
        ## dim = 2, inner most dim, i.e. => choosing the most valued item 
        ## n_samples, n_agents, n_items
        maxitemfori_in_jthbundle, temp = torch.max(multiply_vi_aj, 2)
        ## social welfare calculation for agent i corresponding to its own bundle and other
        ## agent's bundle
        sw_i_j = torch.sum(multiply_vi_aj , dim=2)
        sw_i = torch.sum(multiply_vi_ai , dim=1)
        ## substraction of max valued  from social welfare calculating using other agent's bundle
        sw_i_j_minus_maxitem = sw_i_j - maxitemfori_in_jthbundle
        ## Min item in ith bundle, remove from its bundle
        ## i.e. when chores are there, we can remove a chore from agent i bundle
        ## if they are envying
        vi_min_item, k = torch.min(multiply_vi_ai , dim=1)
        sw_i_afterremovingminitem = sw_i - vi_min_item

        ### Envy is calculated as the minimum envy by either removing item from other agent
        ### or by removing item from agent i bundle 
        envy = torch.min( sw_i_j_minus_maxitem - sw_i[:,None] , sw_i_j - sw_i_afterremovingminitem[:,None]   )
        
        ## filter out and only consider the positive envy value
        # envy =  self.relufn(envy)
        envy[envy < 0] = 0

        sumenvy = torch.sum(envy)

        ### envy array of agent i corresponding to agent j 
        agents = range(n_agents)
        remainingagent = np.delete(agents, i)
        envy = envy[:,remainingagent]
        ## envy is n_Samples x n_agents - 1
        ## envy contains envy of agent i , each samples, corresponding to all other agents
        ## sum envy is total sum of envy across all agents and all samples
        if (trainMode):
            return sumenvy 
        else:
            return sumenvy , envy.T

    def findEFenvy_i(self,allocation_profile,valuation_profile,i=0,trainMode=True):
        valuation_i = valuation_profile[:,i,:]
        allocation_i = allocation_profile[:,i,:]
        n_samples = allocation_profile.shape[0]
        n_agents = allocation_profile.shape[1]
        n_items = allocation_profile.shape[2]

        ## V_i A_j for all j
        multiply_vi_aj =  valuation_i[:,None,:] * allocation_profile 
        ## V_i A_i
        multiply_vi_ai = valuation_i * allocation_i

        ## social welfare calculation for agent i corresponding to its own bundle and other
        ## agent's bundle
        sw_i_j = torch.sum(multiply_vi_aj , dim=2)
        sw_i = torch.sum(multiply_vi_ai , dim=1)

        ### Envy 
        envy = sw_i_j - sw_i[:,None] 
        
        ## filter out and only consider the positive envy value
        # envy =  self.relufn(envy)
        envy[envy < 0] = 0

        sumenvy = torch.sum(envy)

        ### envy array of agent i corresponding to agent j 
        agents = range(n_agents)
        remainingagent = np.delete(agents, i)
        envy = envy[:,remainingagent]
        ## envy is n_Samples x n_agents - 1
        ## envy contains envy of agent i , each samples, corresponding to all other agents
        ## sum envy is total sum of envy across all agents and all samples
        if (trainMode):
            return sumenvy 
        else:
            return sumenvy , envy.T


    def findGeneralEnvy_i(self,allocation_profile,valuation_profile,i=0, trainMode=True, lossfunctionid=2):
        if(lossfunctionid == 4):
            return self.findEFXenvy_i(allocation_profile,valuation_profile,i, trainMode)
        else:
            return self.findEF1envy_i(allocation_profile,valuation_profile,i, trainMode)


    def calculateEnvySamplewise(self,allocation_profile, valuation_profile):
        print('*************************************************************')
        n_samples = allocation_profile.shape[0]
        n_agents = allocation_profile.shape[1]
        n_items = allocation_profile.shape[2]

        sumenvy = 0.0
        envy = torch.zeros(n_samples).to(device)

        for i in range(0,n_agents):      
            ## envy_ was n_Samples x n_agent - 1
            sumenvy_ , envy_ = self.findEF1envy_i(allocation_profile, valuation_profile,i,False)
            ## now summing across each samples
            ## envy's shape is n_samples 
            ## envy_ contains summation of envy of agent i wrt to other agents for each samples
            envy_ = torch.sum(envy_, dim=0)
            envyindex_ = torch.nonzero(envy_)
            count =  n_samples - len(envyindex_)

            envy += envy_
            sumenvy+= float(sumenvy_)
            summary = " Envy agent {} is {} , ef1 samples : {} , a_ef1 : {}"
            print(summary.format(i,sumenvy_,count,count/n_samples))
        print()

        ## sumenvy is summation of envy of all agents across all samples (scalar)
        ## envy is summation of envy of all agents for each samples (shape : n_samples)
        return sumenvy, envy

    def findenvy_i(self,allocation_profile,valuation_profile,i=0):
        valuation_i = valuation_profile[:,i,:]
        allocation_i = allocation_profile[:,i,:]
        n_agents = allocation_profile.shape[1]

        sw_i = torch.sum(valuation_i * allocation_i , dim=1)
        sw_list = []
        for j in range(n_agents):
            allocation_j = allocation_profile[:,j,:]
            sw_ij = torch.sum(valuation_i * allocation_j , dim=1)
            sw_list.append(sw_ij)
        sw_list = torch.stack(sw_list)

        difference = sw_list - sw_i
        relufn = nn.ReLU()
        envy = relufn(difference)
        sumenvy = torch.sum(envy)
        return sumenvy



