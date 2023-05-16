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


from utils.efficiencyutil import EfficiencyUtil 
from utils.fairnessutil import FairnessUtil

# from efficiencyutil import EfficiencyUtil 
# from fairnessutil import FairnessUtil

from random import shuffle

import time
from itertools import combinations 

# import profile_generator
from utils.profile_generator import *

import numpy as np
from scipy import stats
import itertools

import gc

import logging
logging.getLogger('').setLevel(logging.DEBUG)
logger1 = logging.getLogger('1')
logger2 = logging.getLogger('2')
logger3 = logging.getLogger('3')
logger4 = logging.getLogger('4')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



class Util (FairnessUtil,EfficiencyUtil):
    def __init__(self, cfg,logger_file_name):
        super().__init__(cfg)
        self.cfg = cfg
        self.logger_file_name = logger_file_name
        self.softmaxfn = nn.Softmax(dim=1).to(device)
        self.relufn = nn.ReLU()
        self.crossentropyloss = nn.CrossEntropyLoss().to(device)

        self.mseloss = nn.MSELoss()

        if(True):
            filename = '6agnets.pt'
            logger_path = 'logs/' + filename +'.log'

            ####
            logger1.addHandler(logging.FileHandler(filename=logger_path))

            filename = '15CRR1.pt'
            logger_path = 'logs/' + filename +'.log'
            logger2.addHandler(logging.FileHandler(filename=logger_path))

            filename = '15RR1.pt'
            logger_path = 'logs/' + filename +'.log'
            logger3.addHandler(logging.FileHandler(filename=logger_path))

            filename = '15MUW1.pt'
            logger_path = 'logs/' + filename +'.log'
            logger4.addHandler(logging.FileHandler(filename=logger_path))



            filename = '10agentlambda-network lambda0.1.pt'

            logger_path = 'logs/' + filename +'.log'

            logger1.addHandler(logging.FileHandler(filename=logger_path))


    def generate_random_X_all_uniform(self,shape):
        if(self.cfg.items =='all'):
            shapeclone_ = shape.copy()
            shapeclone_2 = shape.copy()
            value = int(shapeclone_[0]/3)
            shapeclone_[0] = int((value))
            shapeclone_2[0] = int((value))
            
            positive_valuations =np.random.uniform(0.0, 1.0, size = shapeclone_)
            positive_valuations =  torch.from_numpy(positive_valuations).float()

            negative_valuations =np.random.uniform(-1.0, 0.0, size = shapeclone_)
            negative_valuations =  torch.from_numpy(negative_valuations).float()

            comb_valuations =np.random.uniform(-1.0, 1.0, size = shapeclone_2)
            comb_valuations =  torch.from_numpy(comb_valuations).float()

            X = torch.cat([positive_valuations, negative_valuations,comb_valuations], dim=0)

        elif(self.cfg.valuationtype =='good'):
            positive_valuations =np.random.uniform(0.0, 1.0, size = shape)
            positive_valuations =  torch.from_numpy(positive_valuations).float()
            X = torch.cat([positive_valuations], dim=0)

        elif(self.cfg.valuationtype =='chore'):
            negative_valuations =np.random.uniform(-1.0, 0.0, size = shape)
            negative_valuations =  torch.from_numpy(negative_valuations).float()
            X = torch.cat([negative_valuations], dim=0)

        elif(self.cfg.valuationtype =='comb'):
            comb_valuations =np.random.uniform(-1.0, 1.0, size = shape)
            comb_valuations =  torch.from_numpy(comb_valuations).float()
            X = torch.cat([comb_valuations], dim=0)

        return X.to(device)


    def generate_random_X_good_uniform(self,shape):
        X = np.random.uniform(0.0, 1.0, size = shape)
        X = torch.from_numpy(X).float()
        return X.to(device)

    def generate_mallow_data(self,shape):
        num_items = shape[2] 
        num_agents = shape[1]
        n_samples = shape[0]
        agents = list(range(num_agents))
        items = list(range(num_items))
        items_rev = items[::-1]
        mixture = ['p70',0.1,0.9]
        ps = np.arange(0.1,1,0.1)
        p = ps[3]


        valuation_profile = []
        for i in range(n_samples):
            profile = generate_mallows_mixture_profile(agents, items, [mixture[1], mixture[2]], [items, items_rev], [p, p])
            valuation = []
            for n_agent in profile.keys():
                # print(profile[n_agent])
                valuation.append((profile[n_agent]) )
            valuation = torch.FloatTensor(valuation).to(device)
            valuation = (valuation + 1)
            valuation = valuation/100
            valuation_profile.append(valuation)
        valuation_profile = torch.stack(valuation_profile)
        return valuation_profile.to(device)


    def generate_random_X_chore_uniform(self,shape):
        X = np.random.uniform(-1.0, 0.0, size = shape)
        X = torch.from_numpy(X).float()
        return X.to(device)

    def generate_random_X_comb_uniform(self,shape):
        X = np.random.uniform(-1.0, 1.0, size = shape)
        X = torch.from_numpy(X).float()
        return X.to(device)

    def generate_test_data(self,shape,valuationDistribution,valuationtype,items):
        if(self.cfg.valuationtype == 'additive'):
            if(valuationDistribution == 'Uniform'):
                if(self.cfg.items == 'good'):
                    return self.generate_random_X_good_uniform(shape)
                elif(self.cfg.items == 'chore'):
                    return self.generate_random_X_chore_uniform(shape)
                elif(self.cfg.items == 'comb'):
                    return self.generate_random_X_comb_uniform(shape)
                elif(self.cfg.items == 'all'):
                    return self.generate_random_X_all_uniform(shape)
            elif(valuationDistribution == 'mallow'):
                return self.generate_mallow_data(shape)
            elif(valuationDistribution == 'dirichlet'):
                n_samples = shape[0]
                n_agents = shape[1]
                n_items = shape[2]
                concentrationParameter = 10
                print('Concentration param', concentrationParameter)
                concentration = np.ones(n_items) * concentrationParameter
                X_test_all = np.random.dirichlet(concentration, (n_samples,n_agents))
                X_test_all = torch.from_numpy(X_test_all).float()
                return X_test_all.to(device)
    


    def generate_batch_data(self,X, mode='train'):
        i=0
        config = self.cfg
        if(self.cfg.lossfunctionId == 2):
            Y = self.findMUW_all(X).float()
        if(config.numberofchannels == 21):
            print('Starting to find MUW')
            Y = self.findMUW_all(X).float()
            print('Done to find MUW')
        elif(config.numberofchannels == 22):
            print('Starting to find Random ALlocation')
            Y = self.findRandomAllocation_all(X).float()
            print('Done find Random ALlocation')

        # if(config.numberofchannels == 2):
        #     print('Starting to find MUW')
        #     Y = self.findMUW_all(X).float()
        #     # Y = self.findRandomAllocation_all(X).float()
        #     print(Y.shape)
        #     print('done finding MUW')
        elif(config.numberofchannels == 31):
            print('Starting to find MUW')
            Y = self.findMUW_all(X).float()
            print(Y.shape)
            print('done finding MUW')
            print('Starting with RR')
            # Z = self.ef1envyfree_greedy_output(X).float()
            Z = self.findRandomAllocation_all(X).float()
            print('done finding with RR')
        elif(config.numberofchannels == 32):
            print('Starting to find MUW')
            Y = self.findMUW_all(X).float()
            print(Y.shape)
            print('done finding MUW')
            print('Starting with WCRR')
            Z = self.findWCRR_all(X).float()
            print('done finding with WCRR')
        elif(config.numberofchannels == 33):
            print('Starting to find MUW')
            Y = self.findMUW_all(X).float()
            print(Y.shape)
            print('done finding MUW')
            print('Starting with RR')
            Z = self.ef1envyfree_greedy_output(X).float()
            print('done finding with RR')

        perm = np.random.permutation(config.num_samples) 
        idx = perm[i * config.batch_size: (i + 1) * config.batch_size]
        while True:      
            idx = perm[i * config.batch_size: (i + 1) * config.batch_size]
        # if( self.cfg.lossfunctionId == 4 ):
            if(config.numberofchannels == 21):
                yield X[idx], Y[idx]
            elif(config.numberofchannels == 22):
                yield X[idx], Y[idx]
            elif(config.numberofchannels == 31):
                yield X[idx], [Y[idx],Z[idx]]
            elif(config.numberofchannels == 32):
                yield X[idx], [Y[idx],Z[idx]]
            elif(config.numberofchannels == 33):
                yield X[idx], [Y[idx],Z[idx]]
            
            # else:
            # yield X[idx], idx
            # yield X[None,idx], idx
            i += 1
            if (i * config.batch_size == config.num_samples):
                i = 0
                perm = np.random.permutation(config.num_samples) 


    def preprocessing(self,X):
        n_samples = X.shape[0]
        n_agents = X.shape[1]
        n_items = X.shape[2]
        return torch.reshape(X, (n_samples, n_agents*n_items))

    def revert_preprocessing(self,X):
        n_samples = X.shape[0]
        return torch.reshape(X, (n_samples, n_agents, n_items))

        
    
    def loss_function_max_sw_wrt_ef1(self, allocation , valuation,lambda_lag):
        # print('loss_function_max_sw_wrt_ef1')
        sw = self.calculate_u_social_welfare(allocation, valuation)
        loss_1 = -torch.sum(sw)

        n_agents = allocation.shape[1]

        envyness = []
        loss_2 = 0
        for i in range(n_agents):
            lossef1  = self.findEF1envy_i(allocation,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)


        # for i in range(n_agents):
        #     loss_2 *= (envyness[i]+ envyness[((i+1)%n_agents)])/2


        loss_1 = loss_1 / (self.cfg.num_agents * self.cfg.num_items )

        # if(self.cfg.loss==0):
        #     loss_2 = torch.sum(lambda_lag * envyness)
        # elif(self.cfg.loss==1):
        #     for i in range(n_agents):
        #         loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
        # elif(self.cfg.loss==2):
        #     for i in range(n_agents):
        #         loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
        #     loss_2 += torch.sum(lambda_lag * envyness)

        loss_2 = torch.sum(self.cfg.train.lambda_init_val * envyness)

        # loss_2 = torch.prod(lambda_lag * envyness)

        loss_2 = (2*loss_2) / (self.cfg.num_agents * (self.cfg.num_agents-1) * self.cfg.num_items ) 

        loss = loss_1 + loss_2
        lag_loss = -loss_2
        
        # loss = loss_1
        ###
        return loss , lag_loss , torch.sum(envyness) , torch.mean(sw)
        # return loss , lag_loss , loss_2 , torch.mean(sw)


    def loss_function_max_sw_wrt_ef(self, allocation , valuation,lambda_lag):
        # print('loss_function_max_sw_wrt_ef1')
        sw = self.calculate_u_social_welfare(allocation, valuation)
        loss_1 = -torch.sum(sw)

        n_agents = allocation.shape[1]

        envyness = []
        loss_2 = 0
        for i in range(n_agents):
            lossef1  = self.findEFenvy_i(allocation,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)


        # for i in range(n_agents):
        #     loss_2 *= (envyness[i]+ envyness[((i+1)%n_agents)])/2


        loss_1 = loss_1 / (self.cfg.num_agents * self.cfg.num_items )

        # if(self.cfg.loss==0):
        #     loss_2 = torch.sum(lambda_lag * envyness)
        # elif(self.cfg.loss==1):
        #     for i in range(n_agents):
        #         loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
        # elif(self.cfg.loss==2):
        #     for i in range(n_agents):
        #         loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
        #     loss_2 += torch.sum(lambda_lag * envyness)

        loss_2 = torch.sum(self.cfg.train.lambda_init_val * envyness)

        # loss_2 = torch.prod(lambda_lag * envyness)

        loss_2 = (2*loss_2) / (self.cfg.num_agents * (self.cfg.num_agents-1) * self.cfg.num_items ) 

        loss = (loss_1) + loss_2
        lag_loss = -loss_2
        
        # loss = loss_1
        ###
        return loss , lag_loss , torch.sum(envyness) , torch.mean(sw)
        # return loss , lag_loss , loss_2 , torch.mean(sw)


    def loss_function_max_sw(self, allocation , valuation,lambda_lag):
        # print('loss_function_max_sw')
        sw = self.calculate_u_social_welfare(allocation, valuation)
        loss_1 = -torch.sum(sw)

        n_agents = allocation.shape[1]

        envyness = []
        loss_2 = 0
        for i in range(n_agents):
            lossef1  = self.findEF1envy_i(allocation,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)


        # for i in range(n_agents):
        #     loss_2 *= (envyness[i]+ envyness[((i+1)%n_agents)])/2


        loss_1 = loss_1 / (self.cfg.num_agents * self.cfg.num_items )

        if(self.cfg.loss==0):
            loss_2 = torch.sum(lambda_lag * envyness)
        elif(self.cfg.loss==1):
            for i in range(n_agents):
                loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
        elif(self.cfg.loss==2):
            for i in range(n_agents):
                loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
            loss_2 += torch.sum(lambda_lag * envyness)
 
        # loss_2 = torch.prod(lambda_lag * envyness)

        loss_2 = (2*loss_2) / (self.cfg.num_agents * (self.cfg.num_agents-1) * self.cfg.num_items ) 

        # loss = loss_1 + loss_2
        lag_loss = -loss_2
        
        loss = loss_1
        ###
        return loss , lag_loss , torch.sum(envyness) , torch.mean(sw)
        # return loss , lag_loss , loss_2 , torch.mean(sw)



    def loss_function_ef1(self, allocation , valuation,lambda_lag):
        # print('loss_function_ef1')
        sw = self.calculate_u_social_welfare(allocation, valuation)
        # loss_1 = -torch.sum(sw)

        n_agents = allocation.shape[1]

        envyness = []
        loss_2 = 0
        for i in range(n_agents):
            lossef1  = self.findEF1envy_i(allocation,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)


        # for i in range(n_agents):
        #     loss_2 *= (envyness[i]+ envyness[((i+1)%n_agents)])/2


        # loss_1 = loss_1 / (self.cfg.num_agents * self.cfg.num_items )

        if(self.cfg.loss==0):
            loss_2 = torch.sum(lambda_lag * envyness) 
        elif(self.cfg.loss==1):
            for i in range(n_agents):
                loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
        elif(self.cfg.loss==2):
            for i in range(n_agents):
                loss_2 += envyness[i]* envyness[((i+1)%n_agents)]
            loss_2 += torch.sum(lambda_lag * envyness)
 
        # loss_2 = torch.prod(lambda_lag * envyness)

        loss_2 = (2*loss_2) / (self.cfg.num_agents * (self.cfg.num_agents-1) * self.cfg.num_items ) 

        # loss = loss_1 + loss_2
        lag_loss = -loss_2
        
        loss = loss_2
        # loss = loss_2  -torch.sum(sw) * 0.01

        ###
        return loss , lag_loss , torch.sum(envyness) , torch.mean(sw)
        # return loss , lag_loss , loss_2 , torch.mean(sw)


    def labellearningMUW(self, allocation_pred, allocation_true, valuation):
        # print(' allocation pred' , allocation_pred.shape)
        n_samples = allocation_pred.shape[0]
        n_items = allocation_true.shape[2]
        n_agents = allocation_true.shape[1]
        # print(' allocation true ', allocation_true.shape)
        # allocation_pred = allocation_pred.long()
        # loss = self.crossentropyloss(allocation_true.reshape(n_samples, n_items*n_agents),allocation_pred.reshape(n_samples, n_items*n_agents))
        swpred = self.calculate_u_social_welfare(allocation_pred, valuation)
        # swtrue = self.calculate_u_social_welfare(allocation_true, valuation)
        loss = self.mseloss(allocation_true, allocation_pred)

        envyness = []
        for i in range(n_agents):
            lossef1  = self.findEF1envy_i(allocation_pred,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)


        return loss , torch.sum(envyness) , torch.mean(swpred)

    def loss_function_max_ef1envyfree(self,allocation,valuation):
        loss1 = 0
        n_agents = allocation.shape[1]
        for i in range(n_agents):
            currentenvy = self.findEF1envy_i(allocation,valuation,i)
            loss1 += currentenvy
        # loss1 = loss1**2
        # loss1 = (2*loss1) / (self.cfg.num_agents * (self.cfg.num_agents-1) * self.cfg.num_items ) 

        return loss1


    def loss_function_max_nashsw( self,allocation , valuation):
        nashsw = self.calculate_nash_social_welfare(allocation, valuation)
        loss_1 = -torch.mean(nashsw)
        return loss_1

    def getDiscreteAllocation(self,x):
        maxswallocation, indices = torch.max(x , dim=1)
        allocation_ = torch.zeros(x.shape).to(device)
        for n_sample in range(indices.shape[0]):
            for n_item in range(indices.shape[1]):
                agentid = indices[n_sample][n_item]
                allocation_[n_sample][agentid][n_item] =1 
        return allocation_

    def getDiscreteAllocationFast(self,x):
        n_agents = x.shape[1]
        values = torch.tensor(self.cfg.values).to(device)
        permvalue = torch.randperm(n_agents)
        values=values[permvalue]
        x = x * values
        mulfactor = 9999999
        x = x * mulfactor
        x =  self.softmaxfn(x)
        return x


    def getDiscreteAllocationParam(self,allocation,valuation, fast=False):
        if(fast):
            discreateallocation = self.getDiscreteAllocationFast(allocation)
        else:
            discreateallocation = self.getDiscreteAllocation(allocation)
        return discreateallocation


    def calculateaEF1ForMUW(self,X):
        n_samples = X.shape[0]
        allocation = self.findMUW_all(X)
        print(X.shape)
        sw = self.calculate_u_social_welfare(allocation,X)
        print('MUW')
        print(' USW ', torch.mean(sw))
        meansw = torch.mean(sw)

        propvalue = self.calculateProportionality(allocation,X)
        notpropindex = torch.nonzero(propvalue)
        count1 =  n_samples - len(notpropindex)
        print(' Total Prop Samples samples ', count1)
        print(' Total Prop samples percent', (count1/n_samples))
        aprop = count1/n_samples
        

        sumenvy, envy = self.calculateEnvySamplewise(allocation,X)
        envyindex = torch.nonzero(envy)
        count1 =  n_samples - len(envyindex)
        print(' Total Envy Free upto one item samples ', count1)
        print(' Total Envy Free upto one item samples percent', (count1/n_samples))
        aef1 = (count1/n_samples)

        sumenvy, envy = self.calculateEnvySamplewiseEFX(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free upto any item samples ', count)
        print(' Total Envy Free upto any item samples percent', (count/n_samples))
        aefx = (count/n_samples)

        sumenvy, envy = self.calculateEnvySamplewiseEF(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/n_samples))
        aef = (count/n_samples)


        print('copy hereeeee')
        summary = "{} {} {} {} {}"
        print(summary.format(aef1,aefx,aef,aprop,meansw))
        logger4.info(summary.format(aef1,aefx,aef,aprop,meansw))
        print('copy hereeeee')



        return count1, count1/n_samples , sw , allocation

    def calculateaEF1ForRandomAllocation(self,X):
        n_samples = X.shape[0]
        allocation = self.findRandomAllocation_all(X)
        print(X.shape)
        sumenvy, envy = self.calculateEnvySamplewise(allocation,X)
        sw = self.calculate_u_social_welfare(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free upto one item samples ', count)
        print(' Total Envy Free upto one item samples percent', (count/n_samples))
        print(' USW ', torch.mean(sw))
        return count, count/n_samples , sw , allocation

    def calculateaEF1ForRR(self,X):
        n_samples = X.shape[0]
        if(self.cfg.items == 'comb'):
            allocation = self.doubleroundrobin_all(X)
        else:
            allocation = self.ef1envyfree_greedy_output(X)
        print(X.shape)
        print('greedy')
        sw = self.calculate_u_social_welfare(allocation,X)
        print(' USW ', torch.mean(sw))
        meansw = torch.mean(sw)

        propvalue = self.calculateProportionality(allocation,X)
        notpropindex = torch.nonzero(propvalue)
        count1 =  n_samples - len(notpropindex)
        print(' Total Prop Samples samples ', count1)
        print(' Total Prop samples percent', (count1/n_samples))
        aprop = count1/n_samples


        sumenvy, envy = self.calculateEnvySamplewise(allocation,X)
        envyindex = torch.nonzero(envy)
        count1 =  n_samples - len(envyindex)
        print(' Total Envy Free upto one item samples ', count1)
        print(' Total Envy Free upto one item samples percent', (count1/n_samples))
        aef1 = (count1/n_samples)


        sumenvy, envy = self.calculateEnvySamplewiseEFX(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free upto any item samples ', count)
        print(' Total Envy Free upto any item samples percent', (count/n_samples))
        aefx = (count/n_samples)


        sumenvy, envy = self.calculateEnvySamplewiseEF(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/n_samples))
        aef = (count/n_samples)


        print('copy hereeeee')
        summary = "{} {} {} {} {}"
        print(summary.format(aef1,aefx,aef,aprop,meansw))
        print('copy hereeeee')

        logger3.info(summary.format(aef1,aefx,aef,aprop,meansw))


        return count1, count1/n_samples , sw , allocation


    def calculateaEF1ForWCRR(self,X):
        n_samples = X.shape[0]
        # if(self.cfg.items == 'comb'):
        #     allocation = self.doubleroundrobin_all(X)
        # else:
        allocation = self.findWCRR_all(X)
        print(X.shape)
        print('greedy')
        sw = self.calculate_u_social_welfare(allocation,X)
        print(' USW ', torch.mean(sw))
        meansw = torch.mean(sw)

        propvalue = self.calculateProportionality(allocation,X)
        notpropindex = torch.nonzero(propvalue)
        count1 =  n_samples - len(notpropindex)
        print(' Total Prop Samples samples ', count1)
        print(' Total Prop samples percent', (count1/n_samples))
        aprop = count1/n_samples


        sumenvy, envy = self.calculateEnvySamplewise(allocation,X)
        envyindex = torch.nonzero(envy)
        count1 =  n_samples - len(envyindex)
        print(' Total Envy Free upto one item samples ', count1)
        print(' Total Envy Free upto one item samples percent', (count1/n_samples))
        aef1 = (count1/n_samples)


        sumenvy, envy = self.calculateEnvySamplewiseEFX(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free upto any item samples ', count)
        print(' Total Envy Free upto any item samples percent', (count/n_samples))
        aefx = (count/n_samples)


        sumenvy, envy = self.calculateEnvySamplewiseEF(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/n_samples))
        aef = (count/n_samples)


        print('copy hereeeee')
        summary = "{} {} {} {} {}"
        print(summary.format(aef1,aefx,aef,aprop,meansw))
        print('copy hereeeee')

        logger2.info(summary.format(aef1,aefx,aef,aprop,meansw))

        return count1, count1/n_samples , sw , allocation



    def plot_loss(self,plotdetails):
        iter_list = plotdetails[0]
        loss_list = plotdetails[1]
        abs_loss_list = plotdetails[2]
        plt.figure()

        plt.subplot(211)
        plt.plot(iter_list[1:], loss_list[1:], 'bo',iter_list[1:], loss_list[1:], 'k')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.suptitle(' Fractional Allocation - Loss')

        plt.subplot(212)
        plt.plot(iter_list[1:], abs_loss_list[1:], 'bo',iter_list[1:], abs_loss_list[1:], 'k')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.suptitle(' 0/1 Allocation - Loss')

        plt.savefig("plots/"+self.logger_file_name+'.png')


    def getCombineAllocation(self,netList, X_test,Y):
        tic1 = time.time()

        allocation = []
        envy = []
        usw = []
        # filename = 'Testing' + str(X_test.shape[1]) + '_' + str(X_test.shape[2]) + '.pt'



        # muwallocation = self.findMUW_all(X_test).float()
        # print('...muwallocation shape', muwallocation.shape)
        print('X test...', X_test.shape)
        # XY = X_test * muwallocation

        num_agents = X_test.shape[1]
        num_items = X_test.shape[2]

        # tempXtest =   (X_test - 0.5) / 0.1

        # twoDvaluations = torch.cat((tempXtest[:,None,:,:],Y[:,:,:,:].float()),1)
        twoDvaluations = torch.cat((X_test[:,None,:,:],Y[:,:,:,:].float()),1)
        # twoDvaluations = X_test[:,None,:,:]


        for netindex, net in enumerate(netList):
            # print(' netindex ', netindex)
            # print(' ........................')
            # print(net['config'])
            # print()

            # if()


            # if(self.cfg.numberofchannels[netindex]==2):
            #     twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:].float()),1)
            # if(self.cfg.numberofchannels[netindex]==6):

            #     # # m = 21
            #     # layersnum = 5
            #     # # num_items = 
            #     # eachlayerm = num_items//layersnum
            #     # print(eachlayerm)

            #     # layersss = []
            #     # for k in range(layersnum-1):
            #     #     agents = []
            #     #     for j in range(eachlayerm):
            #     #         agents.append(k+j*(layersnum))
            #     #     layersss.append(agents)
            #     # print(layersss)
            #     # layers = []

            #     # zerostensor = torch.zeros(muwallocation.shape).to(device)
            #     # print('......',zerostensor.shape)
            #     # for i in range(layersnum-1):
            #     #     layers.append(zerostensor.clone().detach())
            #     #     if(i != 0):
            #     #         layers[i] = layers[i-1].clone().detach()

            #     #     for k in layersss[i]:
            #     #         layers[i][:,:,k] = XY[:,:,k].clone().detach()

            #     # print('..layers', len(layers))
            #     # layers.append(XY.clone().detach())
            #     # print('..layers', len(layers))
            #     # layers  = torch.stack(layers)
            #     # print(layers.shape)
            #     # Y = layers.reshape(X_test.shape[0],layersnum,self.cfg.num_agents,self.cfg.num_items)

            #     # filename = 'Y_test' + str(num_agents) + '_' + str(num_items) + '.pt'
            #     # torch.save(Y,filename)
            #     # print('jjjjj',XY.shape)
            #     # layers = []
            #     # zerostensor = torch.zeros(muwallocation.shape).to(device)
            #     # print('......',zerostensor.shape)
            #     # for i in range(self.cfg.num_items):
            #     #     layers.append(zerostensor.clone().detach())
            #     #     if(i != 0):
            #     #         layers[i] = layers[i-1].clone().detach()

            #     #     layers[i][:,:,i] = XY[:,:,i].clone().detach()
                
            #     # layers  = torch.stack(layers)
            #     # Y = layers.reshape(X_test.shape[0],num_items,num_agents,num_items)
            # elif(self.cfg.numberofchannels[netindex]==21):
            #     print('jjjjj',XY.shape)
            #     layers = []
            #     zerostensor = torch.zeros(muwallocation.shape).to(device)
            #     print('......',zerostensor.shape)
            #     for i in range(self.cfg.num_items):
            #         layers.append(zerostensor.clone().detach())
            #         if(i != 0):
            #             layers[i] = layers[i-1].clone().detach()

            #         layers[i][:,:,i] = XY[:,:,i].clone().detach()
                
            #     layers  = torch.stack(layers)
            #     Y = layers.reshape(X_test.shape[0],num_items,num_agents,num_items)
            #     twoDvaluations = torch.cat((X_test[:,None,:,:],Y[:,:,:,:].float()),1)
            # elif(self.cfg.numberofchannels[netindex]==11):
            #     print('jjjjj',XY.shape)
            #     layers = []
            #     zerostensor = torch.zeros(muwallocation.shape).to(device)
            #     print('......',zerostensor.shape)
            #     for i in range(self.cfg.num_agents):
            #         layers.append(zerostensor.clone().detach())
            #         if(i != 0):
            #             layers[i] = layers[i-1].clone().detach()

            #         layers[i][:,i,:] = XY[:,i,:].clone().detach()
                
            #     layers  = torch.stack(layers)
            #     Y = layers.reshape(X_test.shape[0],num_agents,num_agents,num_items)
            #     twoDvaluations = torch.cat((X_test[:,None,:,:],Y[:,:,:,:].float()),1)

            currentnetallocation = net(twoDvaluations,9999)
            # print('did it pass')

            ### CHANGE HERE to change discrete allocation method
            # currentnetallocation = self.getDiscreteAllocation(currentnetallocation)
            currentnetallocation = currentnetallocation * 9999999
            currentnetallocation = self.softmaxfn(currentnetallocation)


            sumenvy, currentenvy = self.calculateEnvySamplewise(currentnetallocation,X_test)
            currentsw = self.calculate_u_social_welfare(currentnetallocation,X_test)
            n_samples = currentnetallocation.shape[0]
            envyindex = torch.nonzero(currentenvy)
            count =  n_samples - len(envyindex)
            print('___________________________________________')
            print(' Total Envy Free samples ', count)
            print(' Total Envy Free samples percent', (count/n_samples))
            print(' USW ', torch.mean(currentsw))
            print()
            
            ## First network
            if(netindex == 0):
                combinedallocation = currentnetallocation.clone()
                combinedenvy = currentenvy.clone()
                combinedusw = currentsw.clone()
            else:
                combined_allocation_has_envy = (combinedenvy !=0 ) & (currentenvy ==0 )  
                
                combined_allocation_has_envy_index = torch.nonzero(combined_allocation_has_envy)
                combinedallocation[combined_allocation_has_envy_index] = currentnetallocation[combined_allocation_has_envy_index]

                allocation_with_max_sw = (combinedenvy ==0 ) & (currentenvy ==0 ) & (combinedusw < currentsw)
                allocation_with_max_sw_index = torch.nonzero(allocation_with_max_sw)
                combinedallocation[allocation_with_max_sw_index] = currentnetallocation[allocation_with_max_sw_index]
                del combined_allocation_has_envy
                del allocation_with_max_sw

            # del combinedenvy
            del currentenvy
            del currentnetallocation
            del sumenvy
            del envyindex
            del net
            del currentsw

            # del muwallocation
            # del XY
            gc.collect()


        toc1 = time.time()
        time_elapsed1 = (toc1 - tic1)
        print('time elapsed  for NN', time_elapsed1)
        print()
        print()
        print('****************************************************************')

        print('Neural Network::')
        finalusw = self.calculate_u_social_welfare(combinedallocation,X_test)
        print(' USW ', torch.mean(finalusw))
        meansw = torch.mean(finalusw)


        propvalue = self.calculateProportionality(combinedallocation,X_test)
        notpropindex = torch.nonzero(propvalue)
        count1 =  n_samples - len(notpropindex)
        print(' Total Prop Samples samples ', count1)
        print(' Total Prop samples percent', (count1/n_samples))
        aprop = count1/n_samples
        

        sumenvy, envy = self.calculateEnvySamplewise(combinedallocation,X_test)
        envyindex = torch.nonzero(envy)
        count1 =  n_samples - len(envyindex)
        print(' Total Envy Free upto one item samples ', count1)
        print(' Total Envy Free upto one item samples percent', (count1/n_samples))
        aef1 = (count1/n_samples)


        sumenvy, envy = self.calculateEnvySamplewiseEFX(combinedallocation,X_test)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free upto any item samples ', count)
        print(' Total Envy Free upto any item samples percent', (count/n_samples))
        aefx = (count/n_samples)


        sumenvy, envy = self.calculateEnvySamplewiseEF(combinedallocation,X_test)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/n_samples))
        aef = (count/n_samples)



        print('copy hereeeee')

        summary = "{} {} {} {} {}"
        print(summary.format(aef1,aefx,aef,aprop,meansw))
        logger1.info(summary.format(aef1,aefx,aef,aprop,meansw))
        print('copy hereeeee')



        # sumenvy, finalenvy = self.calculateEnvySamplewise(combinedallocation,X_test)
        # finalusw = self.calculate_u_social_welfare(combinedallocation,X_test)
        # envyindex = torch.nonzero(finalenvy)
        # count =  n_samples - len(envyindex)
        # print(' Total Envy Free samples ', count)
        # print(' Total Envy Free samples percent', (count/n_samples))
        # print(' USW ', torch.mean(finalusw))
        print()

        print('****************************************************************')


        # print()
        # print()
        # print('*********************Complete Discrete Allocation*******************************************')
        # combineddiscreteallocation = self.getDiscreteAllocation(combinedallocation)
        # sumenvy, finalenvy = self.calculateEnvySamplewise(combineddiscreteallocation,X_test)
        # finalusw = self.calculate_u_social_welfare(combineddiscreteallocation,X_test)
        # envyindex = torch.nonzero(finalenvy)
        # count =  n_samples - len(envyindex)
        # print(' Total Envy Free samples ', count)
        # print(' Total Envy Free samples percent', (count/n_samples))
        # print(' USW ', torch.mean(finalusw))
        # print()

        print('****************************************************************')

        # return combinedallocation, combineddiscreteallocation , finalenvy 
        # return combinedallocation, combinedallocation , finalenvy 




 
 