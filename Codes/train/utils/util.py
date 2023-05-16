## Imports
import torch

import os
import numpy as np
# import pandas as pd
# import pandas.util.testing as tm
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
import logging
from itertools import combinations 

# import profile_generator
from utils.profile_generator import *

import numpy as np
import itertools


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

    def generate_random_X_all_uniform(self,shape):
        if(self.cfg.items =='all'):
            shapeclone_ = shape.copy()
            shapeclone_2 = shape.copy()
            value = int(shapeclone_[0]/3)
            # shapeclone_[0] = int((307200))
            # shapeclone_2[0] = int((115200))
            
            shapeclone_[0] = int((value * 1.5))
            shapeclone_2[0] = int((value * 0.75))

            # shapeclone_[0] = int((value ))
            # shapeclone_2[0] = int((value ))
            
            positive_valuations =np.random.uniform(0.0, 1.0, size = shapeclone_)
            positive_valuations =  torch.from_numpy(positive_valuations).float()

            negative_valuations =np.random.uniform(-1.0, 0.0, size = shapeclone_2)
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
        if(config.numberofchannels == 21):
            print('Starting to find MUW')
            # Y = self.findMUW_all(X).float()
            print('Done to find MUW')
        # if(config.numberofchannels == 2):
        #     print('Starting to find MUW')
        #     Y = self.findMUW_all(X).float()
        #     # Y = self.findRandomAllocation_all(X).float()
        #     print(Y.shape)
        #     print('done finding MUW')


        #Set True to load data
        if(True):
            print('7,10,13')
            print('1:1:1')
            X1 = torch.load('X-10-1.pt')
            Y1 = torch.load('Y-10-1.pt')
            # X2 = torch.load('X-10-1.pt')
            # Y2 = torch.load('Y-10-1.pt')

            # X21 = X2[:76800]
            # X22 = X2[76800:]
            # Y21 = Y2[:76800]
            # Y22 = Y2[76800:]
            # del X2
            # del Y2
            # X3 = torch.load('X-13-1.pt')
            # Y3 = torch.load('Y-13-1.pt')
            # X31 = X3[:76800]
            # X32 = X3[76800:]
            # Y31 = Y3[:76800]
            # Y32 = Y3[76800:]
            # del X3
            # del Y3


            # X3 = torch.load('X-20-1.pt')
            # Y3 = torch.load('Y-20-1.pt')

        ## Uncoment to generate data
        if(False):
            train_data_shape1 = [config.num_samples , 10 , 20]
            # X1 = self.generate_random_X_good_uniform(train_data_shape1)
            X1 = self.generate_random_X_all_uniform(train_data_shape1)
            print('Start MUW')
            print('X1 shape',X1.shape)
            Y1 = self.findMUW_all(X1).float()

            XY = X1 * Y1
            zerostensor = torch.zeros(XY.shape).to(device)

            layers = []
            for i in range(config.num_agents):
                # layers.append(XY.clone().detach())
                # layers[i][:,:,i+1:] = 0
                layers.append(zerostensor.clone().detach())
                if(i != 0):
                    layers[i] = layers[i-1].clone().detach()
                layers[i][:,i,:] = XY[:,i,:].clone().detach()

            layers  = torch.stack(layers)
            Y1 = layers.reshape(X1.shape[0],10,10,20)
            torch.save(X1, 'X-10-1.pt')
            torch.save(Y1, 'Y-10-1.pt')



            # XY1 = X1 * Y1 
            # n_items = X1.shape[2]
            # n_agents = X1.shape[1]
            # layersnum = 5
            # #Per layer, no of items
            # eachlayerm = n_items//layersnum
            # layersss = []
            # for k in range(layersnum-1):
            #     agents = []
            #     for j in range(eachlayerm):
            #         agents.append(k+j*(layersnum))
            #     layersss.append(agents)
            # ## Layer and item indexing
            # print(layersss)
            # layers = []
            # zerostensor = torch.zeros(Y1.shape).to(device)
            # print('......',zerostensor.shape)

            # for i in range(layersnum-1):
            #     layers.append(zerostensor.clone().detach())
            #     if(i != 0):
            #         layers[i] = layers[i-1].clone().detach()

            #     for k in layersss[i]:
            #         layers[i][:,:,k] = XY1[:,:,k].clone().detach()

            # print('..layers', len(layers))
            # layers.append(XY1.clone().detach())
            # print('..layers', len(layers))
            # layers  = torch.stack(layers)
            # print(layers.shape)
            # Y1 = layers.reshape(X.shape[0],layersnum,n_agents,n_items)

            # torch.save(X1, 'X-10-1.pt')
            # torch.save(Y1, 'Y-10-1.pt')

            print('done saving')
        num_samples = 153600
        perm = np.random.permutation(num_samples) 
        idx = perm[i * config.batch_size: (i + 1) * config.batch_size]
        while True:      
            idx = perm[i * config.batch_size: (i + 1) * config.batch_size]

            # yield X2[idx],idx
            # yield X1[idx],idx

            # if(config.numberofchannels == 1):
            #     yield X2[idx],idx
            #     yield X1[idx],idx
            #     yield X3[idx],idx
            # elif(config.numberofchannels == 2):
            #     yield X2[idx],Y2[idx]
            yield X1[idx],Y1[idx]
            #     yield X3[idx],Y3[idx]
            #     # randomnum = np.random.randint(10)
            #     # if( (randomnum%2) == 0):
            #     #     yield X1[idx],idx
            #     # else:
            #     #     yield X2[idx],idx
            # elif(config.numberofchannels == 21):
            #     # yield X21[idx],Y21[idx]
            #     # yield X1[idx],Y1[idx]
            #     # yield X31[idx],Y31[idx]
            #     # yield X22[idx],Y22[idx]
            #     # yield X32[idx],Y32[idx]

            #     # yield X2[idx],Y2[idx]
            #     yield X1[idx],Y1[idx]
            #     # yield X3[idx],Y3[idx]
            # elif(config.numberofchannels == 22):
            #     yield X[idx], Y[idx]
            # i += 1
            # if (i * config.batch_size == num_samples):
            #     i = 0
            #     perm = np.random.permutation(num_samples) 


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
        n_items = allocation.shape[2]
        envyness = []
        loss_2 = 0
        for i in range(n_agents):
            lossef1  = self.findEFenvy_i(allocation,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)


        # for i in range(n_agents):
        #     loss_2 *= (envyness[i]+ envyness[((i+1)%n_agents)])/2


        loss_1 = loss_1 / (n_agents * n_items )

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

        loss_2 = (2*loss_2) / (n_agents * (n_agents-1) * n_items ) 

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
        sumenvy, envy = self.calculateEnvySamplewise(allocation,X)
        sw = self.calculate_u_social_welfare(allocation,X)
        envyindex = torch.nonzero(envy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free upto one item samples ', count)
        print(' Total Envy Free upto one item samples percent', (count/n_samples))
        print(' USW ', torch.mean(sw))
        return count, count/n_samples , sw , allocation

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


    def getCombineAllocation(self,netList, X_test):
        ##### I dont run this here, it runs in my test branch
        tic1 = time.time()

        allocation = []
        envy = []
        usw = []

        # muwallocation = self.findMUW_all(X_test).float()
        # muwallocation = self.findRandomAllocation_all(X_test).float()
        if(self.cfg.numberofchannels == 21):
            muwallocation = self.findMUW_all(X_test).float()
        elif(self.cfg.numberofchannels == 22):
            muwallocation = self.util.findRandomAllocation_all(X_test).float()
        elif(self.cfg.numberofchannels == 31):
            muwallocation = self.findMUW_all(X_test).float()
            # rr = self.findRandomAllocation_all(X_test).float()
            # rr = self.ef1envyfree_greedy_output(X_test).float()

        elif(self.cfg.numberofchannels == 32):
            wcrr = self.findWCRR_all(X_test)
        elif(self.cfg.numberofchannels == 33):
            rr = self.ef1envyfree_greedy_output(X_test).float()
            
        # print(torch.cat((batch_X[:,None,:,:],allocation_true[:,None,:,:].float()),1).shape)
        # twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:].float()),1)
        if(self.config.numberofchannels == 2):
            twoDvaluations = X_test[:,None,:,:]
        if(self.cfg.numberofchannels == 21 or self.cfg.numberofchannels == 22):
            twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:]),1)
        elif(self.cfg.numberofchannels == 31):
            twoDvaluations = torch.cat((muwallocation[:,None,:,:],X_test[:,None,:,:],rr[:,None,:,:]),1)
        elif(self.cfg.numberofchannels == 32):
            twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:],wcrr[:,None,:,:]),1)
        elif(self.cfg.numberofchannels == 33):
            twoDvaluations = torch.cat((X_test[:,None,:,:],rr[:,None,:,:],muwallocation[:,None,:,:]),1)


        for netindex, net in enumerate(netList):
            # print(' netindex ', netindex)
            # print(' ........................')
            # print(net['config'])
            # print()
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


        toc1 = time.time()
        time_elapsed1 = (toc1 - tic1)
        print('time elapsed  for NN', time_elapsed1)
        print()
        print()
        print('****************************************************************')
        sumenvy, finalenvy = self.calculateEnvySamplewise(combinedallocation,X_test)
        finalusw = self.calculate_u_social_welfare(combinedallocation,X_test)
        envyindex = torch.nonzero(finalenvy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/n_samples))
        print(' USW ', torch.mean(finalusw))
        print()

        print('****************************************************************')


        print()
        print()
        print('*********************Complete Discrete Allocation*******************************************')
        combineddiscreteallocation = self.getDiscreteAllocation(combinedallocation)
        sumenvy, finalenvy = self.calculateEnvySamplewise(combineddiscreteallocation,X_test)
        finalusw = self.calculate_u_social_welfare(combineddiscreteallocation,X_test)
        envyindex = torch.nonzero(finalenvy)
        count =  n_samples - len(envyindex)
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/n_samples))
        print(' USW ', torch.mean(finalusw))
        print()

        print('****************************************************************')

        return combinedallocation, combineddiscreteallocation , finalenvy 




 