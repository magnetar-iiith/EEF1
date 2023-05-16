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

from random import shuffle

import time
import logging
from itertools import combinations 


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Archive():
    def __init__(self, cfg,logger_file_name):
        self.cfg = cfg
        self.logger_file_name = logger_file_name

    # def generate_random_X(self,shape):
    #     return np.random.rand(*shape)

    ## enviness of agent i of agent j
    # def calculate_EF1_enviness_ij(self,sw,allocation, valuation, i , j):
    #     v_i =  valuation[:,i,:]
    #     sw_i = torch.sum( sw[:,i,:], dim=1) 
    #     x_j = allocation[:,j,:]

    #     vi_xj =  v_i * x_j 
    #     sw_i_xj =  torch.sum( v_i * x_j , dim=1) 

    #     sw_i_xj_k_list = []
    #     for k in range(self.cfg.num_items):
    #         sw_i_xj_k = sw_i_xj - vi_xj[:,k]
    #         sw_i_xj_k_list.append(sw_i_xj_k)
    #     sw_i_xj_k_list =  torch.stack(sw_i_xj_k_list)
    #     min_sw_i_xj , incidies = torch.min(sw_i_xj_k_list , dim=0)
    #     ef1envy_ij = min_sw_i_xj - sw_i

    #     relu = nn.ReLU()
    #     ef1envy_ij = relu(ef1envy_ij)
    #     expected_ef1envy_ij  = torch.mean(ef1envy_ij)
    #     return expected_ef1envy_ij


    # def calculate_EF1_enviness(self,allocation,valuation):
    #     sumvalue = 0
    #     sw = valuation * allocation
    #     ef1enviness = []
    #     for i in range(self.cfg.num_agents):
    #         envy_per_agent = []
    #         ## Agent i summed social welfare on all samples 
    #         for j in range(self.cfg.num_agents):
    #             if (i != j):
    #                 ef1envyij = self.calculate_EF1_enviness_ij(sw,allocation,valuation, i,j)
    #                 envy_per_agent.append(ef1envyij)
    #                 sumvalue += ef1envyij
    #             ef1enviness.append(envy_per_agent)  

    #     ef1enviness = torch.tensor(ef1enviness , requires_grad=True )
    #     return ef1enviness


    # def findEF1envy_i(self,allocation_profile,valuation_profile,i=0,trainMode=True):

    #     valuation_i = valuation_profile[:,i,:]
    #     allocation_i = allocation_profile[:,i,:]
    #     n_samples = allocation_profile.shape[0]
    #     n_agents = allocation_profile.shape[1]
    #     n_items = allocation_profile.shape[2]

    #     sw_i = torch.sum(valuation_i * allocation_i , dim=1)
    #     sw_i_clone = sw_i.clone()
    #     differenceList_bad = []
    #     sw_list = []
    #     for j in range(n_agents):
    #         if (i!=j):
    #             allocation_j = allocation_profile[:,j,:]
    #             sw_ij = torch.sum(valuation_i * allocation_j , dim=1)
    #             ## Removing one item i.e. the item that is valued maximum

    #             eachitemvaluationlist_i = [valuation_i[:,k]*allocation_i[:,k] for k in range(n_items)]
    #             eachitemvaluationlist_j= [valuation_i[:,k]*allocation_j[:,k] for k in range(n_items)]

    #             eachitemvaluationlist_i = torch.stack(eachitemvaluationlist_i)
    #             eachitemvaluationlist_j = torch.stack(eachitemvaluationlist_j)

    #             minvalueditem,idx = torch.min(eachitemvaluationlist_i, dim=0)
    #             maxvalueditem,idx = torch.max(eachitemvaluationlist_j, dim=0)


    #             # eachitemvaluationlist = [valuation_i[:,k]*allocation_i[:,k] for k in range(n_items)]
    #             # eachitemvaluation = torch.stack(eachitemvaluationlist)
    #             # minvalueditem,idx = torch.min(eachitemvaluation, dim=0)
    #             # maxvalueditem,idx = torch.max(eachitemvaluation, dim=0)

    #             ## Removing a bad from self
    #             sw_i_clone = sw_i_clone - minvalueditem
    #             difference = sw_ij - sw_i_clone
    #             differenceList_bad.append(difference)
        
    #             ## Removing a good from other
    #             sw_ij = sw_ij - maxvalueditem
    #             sw_list.append(sw_ij)
        

    #     sw_list = torch.stack(sw_list)
    #     difference_good = sw_list - sw_i
    #     differenceList_bad = torch.stack(differenceList_bad)

    #     relufn = nn.ReLU()
    #     envy_good = relufn(difference_good)
    #     envy_bad = relufn(differenceList_bad)
    #     envy = torch.min(envy_good,envy_bad)
    #     sumenvy = torch.sum(envy)

    #     if(trainMode):
    #         return sumenvy 
    #     else:
    #         return sumenvy , envy


    def findEFXenvy_i(self,allocation_profile,valuation_profile,i=0, trainMode=True):

        valuation_i = valuation_profile[:,i,:]
        allocation_i = allocation_profile[:,i,:]


        n_samples = allocation_profile.shape[0]
        n_agents = allocation_profile.shape[1]
        n_items = allocation_profile.shape[2]

        sw_i = torch.sum(valuation_i * allocation_i , dim=1)
        # sw_i = sw_i.to(device)

        sw_list = []
        for j in range(n_agents):
            if (i!=j):
                allocation_j = allocation_profile[:,j,:]
                sw_ij = torch.sum(valuation_i * allocation_j , dim=1)
                ## Removing one item i.e. the item that is valued maximum
                eachitemvaluationlist = [valuation_i[:,k]*allocation_j[:,k] for k in range(n_items)]
                eachitemvaluation = torch.stack(eachitemvaluationlist)
                minvalueditem,idx = torch.min(eachitemvaluation, dim=0)
                sw_ij = sw_ij - minvalueditem
                sw_list.append(sw_ij)
        
        sw_list = torch.stack(sw_list)

        difference = sw_list - sw_i
        relufn = nn.ReLU()
        envy = relufn(difference)

        sumenvy = torch.sum(envy)
        if(trainMode):
            return sumenvy 
        else:
            return sumenvy , envy
            

    def loss_function_max_envyfree(self, allocation , valuation):
        # enviness = calculate_EF1_enviness (allocation, valuation)
        # # loss = torch.mean(enviness) 
        
        loss = 0
        for i in range(self.cfg.num_agents):
            loss += self.findenvy_i(allocation,valuation,i)
        return loss

    def loss_function_max_ef1envyfree_v2(self,allocation,valuation):
        loss1 = 0
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            currentenvy = self.findEF1envy_i(allocation,valuation,i)
            loss1 += currentenvy**2
            # loss2 *= currentenvy
        return loss1

    def loss_function_max_ef1envyfree_v3(self,allocation,valuation):
        loss1 = 0
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            currentenvy = self.findEF1envy_i(allocation,valuation,i)
            loss1 += currentenvy + 1e-3*loss1*currentenvy
            # loss2 *= currentenvy
        return loss1
    
    def loss_function_max_ef1envyfree_v4(self,allocation,valuation):
        loss = 0
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            for j in range(self.cfg.num_agents):
                if( i != j):
                    loss += self.findEF1envy_ij(allocation,valuation,i,j)
        return loss**2

    def loss_function_max_ef1envyfree_v5(self,allocation,valuation):
        loss1 = 0
        loss2 = 100
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            for j in range(self.cfg.num_agents):
                if( i != j):
                    currentenvy = self.findEF1envy_ij(allocation,valuation,i,j)
                    loss1 += currentenvy
                    loss2 *= currentenvy
        return loss1 + loss2

    def getlogcosh(self,x):
        return abs(x) + torch.log1p(torch.exp(-2 * abs(x))) - torch.log(torch.tensor(2.0).to(device))

    def loss_function_max_ef1envyfree_v6(self,allocation,valuation):
        loss1 = 0
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            for j in range(self.cfg.num_agents):
                if( i != j):
                    x = self.findEF1envy_ij(allocation,valuation,i,j)
                    loss1 += self.getlogcosh(x)
        return loss1

    def loss_function_max_ef1envyfree_v7(self,allocation,valuation):
        loss1 = 0
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            for j in range(self.cfg.num_agents):
                if( i != j):
                    x = self.findEF1envy_ij(allocation,valuation,i,j)
                    loss1 += x
        loss1  =  self.getlogcosh(loss1)
        return loss1
    
    def loss_function_max_ef1envyfree_v8(self,allocation,valuation):
        loss1 = 0
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            loss2 = 1
            for j in range(self.cfg.num_agents):
                if( i != j):
                    x = self.findEF1envy_ij(allocation,valuation,i,j)
                    loss1 += x
                    loss2 *= x
            loss1 += loss2            
        return loss1
    

    def loss_function_max_ef1envyfree_v9(self,allocation,valuation):
        loss = 0
        # loss2 = 1e-3
        x , envyvaluesamplewise = self.findEF1envy_i(allocation,valuation,0,False)
        for i in range(1,self.cfg.num_agents):
            x , envyvaluesamplewise_i = self.findEF1envy_i(allocation,valuation,i,False)
            envyvaluesamplewise += envyvaluesamplewise_i
        
        loss = torch.sum(envyvaluesamplewise, dim=1)
        loss = torch.sum((10*loss)**2)
        return loss

    def loss_function_max_ef1envyfree_v10(self,allocation,valuation):
        loss1 = 0
        loss3 = 1
        # loss2 = 1e-3
        for i in range(self.cfg.num_agents):
            loss2 = 1
            for j in range(self.cfg.num_agents):
                if( i != j):
                    x = self.findEF1envy_ij(allocation,valuation,i,j)
                    loss1 += x
                    loss2 *= x
                    loss3 *= x
            loss1 += loss2   
        loss1 += loss3         
        return loss1
    

    def loss_function_max_efxenvyfree(self,allocation,valuation):
        loss = 0
        for i in range(self.cfg.num_agents):
            loss += self.findEFXenvy_i(allocation,valuation,i)
        return loss


    def abs_loss_function_max_efxenvyfree(self,allocation,valuation, fast=False):
        if(fast):
            discreateallocation = self.getDiscreteAllocationFast(allocation)
        else:
            discreateallocation = self.getDiscreteAllocation(allocation)

        loss = self.loss_function_max_efxenvyfree(discreateallocation,valuation)
        return loss

    # def loss_function_max_sw_wrt_ef1(self, allocation , valuation,lambda_lag):
    #     sw = self.calculate_social_welfare(allocation, valuation)
    #     # loss_1 = -torch.mean(sw)
    #     loss_1 = -torch.sum(sw)

    #     loss_2 = 0
    #     envyness = 0
    #     for i in range(self.cfg.num_agents):
    #         lossef1  = self.findEF1envy_i(allocation,valuation,i)
    #         loss_2 += lambda_lag[i]*lossef1
    #         envyness += lossef1

    #     loss = loss_1 + loss_2
    #     lag_loss = -loss_2

    #     return loss , lag_loss , envyness , torch.mean(sw)


    def loss_function_max_sw_wrt_ef1(self, allocation , valuation,lambda_lag):
        # sw = self.calculate_social_welfare(allocation, valuation)
        # loss_1 = -torch.sum(sw)

        # envyness = []
        # loss_2 = 0
        # for i in range(self.cfg.num_agents):
        #     lossef1  = self.findEF1envy_i(allocation,valuation,i)
        #     envyness.append(lossef1)

        # envyness = torch.stack(envyness)

        # loss_2 = torch.sum(lambda_lag * envyness)


        # loss = loss_1 + loss_2
        # lag_loss = -loss_2

        sw = self.calculate_social_welfare(allocation, valuation)
        loss_1 = -torch.sum(sw)

        # batchsize = allocation.shape[0]


        envyness = []
        loss_2 = 0
        for i in range(self.cfg.num_agents):
            lossef1  = self.findEF1envy_i(allocation,valuation,i)
            envyness.append(lossef1)

        envyness = torch.stack(envyness)
        # loss_2 = torch.sum(lambda_lag * envyness)

        loss_1 = loss_1 / (self.cfg.num_agents * self.cfg.num_items )
        # envyness1 = envyness / ((self.cfg.num_agents-1) * self.cfg.num_items)
        loss_2 = torch.sum(lambda_lag * envyness)
        loss_2 = (2*loss_2) / (self.cfg.num_agents * (self.cfg.num_agents-1) * self.cfg.num_items ) 


        loss = loss_1 + loss_2
        lag_loss = -loss_2



        return loss , lag_loss , torch.sum(envyness) , torch.mean(sw)



    def getAllAllocationFor2Agents(self,n_items):
        full_allocation = torch.ones(n_items)
        empty_allocation = torch.zeros(n_items)

        # tensor_a = torch.tensor(range(n_items))
        
        allocation_list = []

        # for i in range(1,n_items+1):
        for i in range(1,n_items+1):
            value = list(combinations(range(n_items), i))
            # value = torch.combinations(tensor_a,r=i)
            print(value)
            for j in range(len(value)):
                # print(value[j])
                combined_allocation = []
                allocation = torch.zeros(n_items)
                allocation[list(value[j])] = 1
                other_agent_allocation =  torch.ones(n_items) - allocation
                combined_allocation.append(allocation)
                combined_allocation.append(other_agent_allocation)
                combined_allocation = torch.stack(combined_allocation)
                allocation_list.append(combined_allocation)
                # print('__________________________________________________________')


        print('endddd')
        combined_allocation = []
        combined_allocation.append(empty_allocation)
        combined_allocation.append(full_allocation)
        combined_allocation = torch.stack(combined_allocation)
        allocation_list.append(combined_allocation)

            
        allocation_list = torch.stack(allocation_list).to(device)

        return allocation_list

    def findNashSWAllocation(self,allocation_profile,valuation_profile):

        n_samples = valuation_profile.shape[0]
        alloc_profiles = allocation_profile.shape[0]
        nashswallocations = []
        for i in range(n_samples):
                nashsw = 0
                valuation = valuation_profile[i]
                for j in range(alloc_profiles):
                    allocation = allocation_profile[j]

                    bundlereceivedvaluation = allocation * valuation
                    utilities = torch.sum( bundlereceivedvaluation, dim=1)

                    n_agents = allocation.shape[0]
                    nashswcurrent = 1
                    for i in range(n_agents):
                            nashswcurrent *=  utilities[i]


                    if (nashswcurrent >= nashsw):
                            nashsw = nashswcurrent
                            nash_allocation_max = allocation.clone()

                nashswallocations.append(nash_allocation_max)

        nashswallocations = torch.stack(nashswallocations)
        return nashswallocations

    def findEqualAllocationCount(self,allocationList, compareAllocation):
        equal = 0
        for i in range(0,allocationList.shape[0]):
            if (torch.all(torch.eq(compareAllocation,allocationList[i]))):
                equal += 1
        return equal

    def findEF1envy_ij(self,allocation_profile,valuation_profile,i=0,j=1):
        valuation_i = valuation_profile[:,i,:]
        allocation_i = allocation_profile[:,i,:]
        n_samples = allocation_profile.shape[0]
        n_agents = allocation_profile.shape[1]
        n_items = allocation_profile.shape[2]

        allocation_j = allocation_profile[:,j,:]

        sw_i = torch.sum(valuation_i * allocation_i , dim=1)
        sw_ij = torch.sum(valuation_i * allocation_j , dim=1)

        eachitemvaluationlist = [valuation_i[:,k]*allocation_j[:,k] for k in range(n_items)]
        eachitemvaluation = torch.stack(eachitemvaluationlist)
        maxvalueditem,idx = torch.max(eachitemvaluation, dim=0)
        sw_ij = sw_ij - maxvalueditem

        difference = sw_ij - sw_i
        envy = self.relufn(difference)
        sumenvy = torch.sum(envy)
        return sumenvy 
            



# if (lossfunctionId == 1):
#     loss = self.util.loss_function_max_envyfree(allocation,batch_X)
# elif (lossfunctionId == 2):
#     loss = self.util.loss_function_max_ef1envyfree(allocation,batch_X)
#     dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
#     abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
# elif ( lossfunctionId == 3):
#     loss ,lag_loss , envyness, sw = self.util.loss_function_max_sw_wrt_ef1(allocation, batch_X,lambda_lag[0])
#     dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
#     abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
#     envy_loss += float(envyness)
#     sw_total += float(sw)

# if (lossfunctionId == 3):
#     ## lagrange update
#     if (  ((_+1) % self.config.train.lambda_update_freq) == 0):
#         print('Updating labma' , lambda_lag[0])
#         optimizer_lag.zero_grad()
#         lag_loss.backward( retain_graph=True )
#         optimizer_lag.step()
#         print('lambda_lag[0]', lambda_lag[0])

# if((lossfunctionId == 3) ):
#     if((maxsw < sw_total)):
#         print('saving the best model for loss function 3')
#         print()
#         print(' Minimum Loss ', minimumloss, 'max sw ', maxsw)
#         print()

#         minimumloss = abs_running_loss
#         maxsw = sw_total
#         # bestmodel = self.net.state_dict()
#         # logger1.info('Saving the best model')
#         # logger1.info(minimumloss)
#         torch.save(self.net.state_dict(), self.model_path)
# else:

# summary = "TRAIN-BATCH Iter: {}, running : {:.4f} , running abs : {:.4f}, current batch loss : {:.4f}     t ={:.4f}"
# # logger1.info(summary.format(_,running_loss,abs_running_loss,loss,time_elapsed))
# print(summary.format(_,running_loss,abs_running_loss,loss,time_elapsed))

# bestmodel = self.net.state_dict()
# logger1.info('Saving the best model')
# logger1.info(minimumloss)

### ONE CHANGE



# if(schedulervalue == 1):
#     scheduler.step()
#     if(_ % self.config.train.step_size == 0):
#         # logger1.info('Scheduler')
#         for param_group in optimizer_net_w.param_groups:
#             print(param_group['lr'])
#             # logger1.info(param_group['lr'])
        
#         if( _ != 0):
#             # load best model
#             print('Loading the best model')
#             # self.net.load_state_dict(bestmodel)

#             state_dict = torch.load(self.model_path)
#             self.net.load_state_dict(state_dict)

# if( _ % 30000):
#     torch.save(self.net.state_dict(), self.model_path)


        # if (lossfunctionId == 1):
        #     loss = self.util.loss_function_max_envyfree(allocation,X_test)
        # elif (lossfunctionId == 2):
        #     loss = self.util.loss_function_max_ef1envyfree(allocation,X_test)
        # elif ( lossfunctionId == 3):
        #     loss ,lag_loss , envyness, sw = self.util.loss_function_max_sw_wrt_ef1(allocation, X_test,lambda_lag[0])
        # elif (lossfunctionId == 4):
        #     loss = self.util.loss_function_max_efxenvyfree(allocation,X_test)
        # elif (lossfunctionId == 5):
        #     loss = self.util.loss_function_max_ef1envyfree_v2(allocation,X_test)
        # elif (lossfunctionId == 6):
        #     loss = self.util.loss_function_max_ef1envyfree_v3(allocation,X_test)
        # elif (lossfunctionId == 7):
        #     loss = self.util.loss_function_max_ef1envyfree_v4(allocation,X_test)
        # elif (lossfunctionId == 8):
        #     loss = self.util.loss_function_max_ef1envyfree_v5(allocation,X_test)
        # elif (lossfunctionId == 9):
        #     loss = self.util.loss_function_max_ef1envyfree_v6(allocation,X_test)
        # elif (lossfunctionId == 10):
        #     loss = self.util.loss_function_max_ef1envyfree_v7(allocation,X_test)
        # elif (lossfunctionId == 11):
        #     loss = self.util.loss_function_max_ef1envyfree_v8(allocation,X_test)
        # elif (lossfunctionId == 12):
        #     loss = self.util.loss_function_max_ef1envyfree_v9(allocation,X_test)
        # elif (lossfunctionId == 13):
        #     loss = self.util.loss_function_max_ef1envyfree_v10(allocation,X_test)
        # elif (lossfunctionId == 14):
        #     loss = self.util.loss_function_max_nashsw(allocation,X_test)
        # elif (lossfunctionId == 15):
        #     loss = self.util.loss_function_max_ef1envyfree(allocation,X_test)


        ##For identical allocations
        # n_agents = X.shape[1]
        # for agentid in range(1,n_agents):
        #     X[:,agentid,:] = X[:,0,:].clone()

        # if(lossfunctionid == 15):
        #     X[:,0,:] = X[:,1,:].clone()



            # print(' ')
            # print('_______________________________________________________')
            # for i in range(self.config.num_agents):
            #     sumenvyvalue , envyvalue = self.util.findGeneralEnvy_i(allocation_test,X_test,i,False,self.config.lossfunctionId)
            #     summary = "Frac Ef1 agent {} is {} "
            #     # logger1.info(summary.format(i,envyvalue))
            #     print(summary.format(i,sumenvyvalue))
            
            # print('_______________________________________________________')

            # print(' With binary allocation - EF1 ')
            # print('_______________________________________________________')

            # print('')
            # print('_______________________________________________________')
            # print(self.logger_file_name)
            # binaryEnvyList = []
            # for i in range(self.config.num_agents):
            #     sumenvyvalue , envyvalue = self.util.findGeneralEnvy_i(discreteAllocationTest,X_test,i, False, self.config.lossfunctionId)
            #     summary = "01 Ef1 agent {} is {} "
            #     print(summary.format(i,sumenvyvalue))
            #     binaryEnvyList.append(envyvalue)
            # print('_______________________________________________________')
            # print('')


            # num_other_agents = self.config.num_agents - 1
            # count = 0
            # sample_envyfree_count = self.config.num_agents * num_other_agents
            # for k in range(n_samples):
            #     count_ = 0
            #     for i in range(self.config.num_agents):
            #         for j in range(num_other_agents):
            #             if(binaryEnvyList[i][j][k] == 0):
            #                 count_ += 1
            #     if( count_ == sample_envyfree_count ):
            #         count += 1

            # print('___________________________________________')
            # print('Total Envy Free samples ', count)
            # print(' Total Envy Free samples percent', (count/n_samples))

            # for i in range(self.config.num_agents):
            #     count = 0
            #     for k in range(n_samples):
            #         _count = 0
            #         for j in range(num_other_agents):
            #             if ( (binaryEnvyList[i][j][k] == 0) ):
            #                 _count += 1
            #         if (_count == num_other_agents):
            #             count += 1

            #     print(' For agent ', i , ' envy free samples : ', count)
            #     print(' Total Envy Free samples percent', (count/n_samples))

            # print('SW ::', torch.mean(self.util.calculate_u_social_welfare(discreteAllocationTest,X_test)) )
            # print()







        # self.conv1 = nn.Conv2d(1, 64, 3).to(device)
        # self.conv2 = nn.Conv2d(64, 128, 3).to(device)
        # self.conv3 = nn.Conv2d(128, 256, 3).to(device)
        # self.conv4 = nn.Conv2d(256, 512, 3).to(device)
        # # self.conv5 = nn.Conv2d(64, 128, 2).to(device)
        # # self.conv6 = nn.Conv2d(128, 256, 1).to(device)
        # # self.reverseconv6 = nn.ConvTranspose2d(256, 128, 1).to(device)
        # # self.reverseconv5 = nn.ConvTranspose2d(128, 64, 2).to(device)
        # self.reverseconv4 = nn.ConvTranspose2d(512, 256, 3).to(device)
        # self.reverseconv3 = nn.ConvTranspose2d(256, 128, 3).to(device)
        # self.reverseconv2 = nn.ConvTranspose2d(128, 64, 3).to(device)
        # self.reverseconv1 = nn.ConvTranspose2d(64, 1, 3).to(device)

        # self.conv1 = nn.Conv2d(1, 64, 3).to(device)
        # self.conv2 = nn.Conv2d(64, 128, 3).to(device)
        # self.conv3 = nn.Conv2d(128, 256, 3).to(device)
        # self.conv4 = nn.Conv2d(256, 512, 3).to(device)
        # self.reverseconv4 = nn.ConvTranspose2d(512, 256, 3).to(device)
        # self.reverseconv3 = nn.ConvTranspose2d(512, 128, 3).to(device)
        # self.reverseconv2 = nn.ConvTranspose2d(256, 64, 3).to(device)
        # self.reverseconv1 = nn.ConvTranspose2d(128, 1, 3).to(device)


        # self.conv1 = nn.Conv2d(1, 8, 3).to(device)
        # self.conv2 = nn.Conv2d(8, 16, 3).to(device)
        # self.conv3 = nn.Conv2d(16, 32, 3).to(device)
        # self.conv4 = nn.Conv2d(32, 64, 3).to(device)
        # self.conv5 = nn.Conv2d(64, 128, 2).to(device)
        # # self.conv6 = nn.Conv2d(128, 256, 1).to(device)
        # # self.reverseconv6 = nn.ConvTranspose2d(256, 128, 1).to(device)
        # self.reverseconv5 = nn.ConvTranspose2d(128, 64, 2).to(device)
        # self.reverseconv4 = nn.ConvTranspose2d(64, 32, 3).to(device)
        # self.reverseconv3 = nn.ConvTranspose2d(32, 16, 3).to(device)
        # self.reverseconv2 = nn.ConvTranspose2d(16, 8, 3).to(device)
        # self.reverseconv1 = nn.ConvTranspose2d(8, 1, 3).to(device)

        # self.conv1 = nn.Conv2d(1, 8, 3).to(device)
        # self.conv2 = nn.Conv2d(8, 16, 3).to(device)
        # self.conv3 = nn.Conv2d(16, 32, 3).to(device)
        # self.conv4 = nn.Conv2d(32, 64, 3).to(device)
        # self.conv5 = nn.Conv2d(64, 128, 2).to(device)
        # # self.conv6 = nn.Conv2d(128, 256, 1).to(device)
        # # self.reverseconv6 = nn.ConvTranspose2d(256, 128, 1).to(device)
        # self.reverseconv5 = nn.ConvTranspose2d(128, 64, 2).to(device)
        # self.reverseconv4 = nn.ConvTranspose2d(128, 32, 3).to(device)
        # self.reverseconv3 = nn.ConvTranspose2d(64, 16, 3).to(device)
        # self.reverseconv2 = nn.ConvTranspose2d(32, 8, 3).to(device)
        # self.reverseconv1 = nn.ConvTranspose2d(16, 1, 3).to(device)

        # self.fc1 = nn.Linear(16 * 2 * 2, 256).to(device)
        # self.fc1 = nn.Linear(200 , 256).to(device)
        # self.fc2 = nn.Linear(256, 256).to(device)
        # self.fc3 = nn.Linear(256, 200).to(device)

        # self.enc1 = nn.Linear(config.num_agents*config.num_items , 100).to(device)
        # self.enc2 = nn.Linear(100 , 50).to(device)
        # self.enc3 = nn.Linear(50 , 25).to(device)
        # self.dec3 = nn.Linear(25 , 50).to(device)
        # self.dec2 = nn.Linear(50 , 100).to(device)
        # self.dec1 = nn.Linear(100 , config.num_agents*config.num_items).to(device)


        # x1 = F.tanh(self.conv1(x))
        # x2 = F.tanh(self.conv2(x1))
        # x3 = F.tanh(self.conv3(x2))
        # x = F.tanh(self.conv4(x3))
        # x5 = F.tanh(self.conv5(x4))
        # x = F.tanh(self.reverseconv5(x5))
        # x = torch.cat((x4,x),1)
        # x = F.tanh(self.reverseconv4(x))
        # x = torch.cat((x3,x),1)
        # x = F.tanh(self.reverseconv3(x))
        # x = torch.cat((x2,x),1)
        # x = F.tanh(self.reverseconv2(x))
        # x = torch.cat((x1,x),1)
        # x = F.tanh(self.reverseconv1(x))

        # x1 = F.relu(self.conv1(x))
        # x2 = F.relu(self.conv2(x1))
        # x3 = F.relu(self.conv3(x2))
        # x = F.relu(self.conv4(x3))
        # # x5 = F.relu(self.conv5(x4))
        # # x = F.relu(self.reverseconv5(x5))
        # # x = torch.cat((x4,x),1)
        # x = F.relu(self.reverseconv4(x))
        # # x = torch.cat((x3,x),1)
        # x = F.relu(self.reverseconv3(x))
        # # x = torch.cat((x2,x),1)
        # x = F.relu(self.reverseconv2(x))
        # # x = torch.cat((x1,x),1)
        # x = F.relu(self.reverseconv1(x))

        # x1 = self.sigmoid(self.conv1(x))
        # x2 = self.sigmoid(self.conv2(x1))
        # x3 = self.sigmoid(self.conv3(x2))
        # x4 = self.sigmoid(self.conv4(x3))
        # x5 = self.sigmoid(self.conv5(x4))
        # x = self.sigmoid(self.reverseconv5(x5))
        # # x = torch.cat((x4,x),1)
        # x = self.sigmoid(self.reverseconv4(x))
        # # x = torch.cat((x3,x),1)
        # x = self.sigmoid(self.reverseconv3(x))
        # # x = torch.cat((x2,x),1)
        # x = self.sigmoid(self.reverseconv2(x))
        # # x = torch.cat((x1,x),1)
        # x = self.sigmoid(self.reverseconv1(x))



        # x1 = F.tanh(self.conv1(x))
        # x2 = F.tanh(self.conv2(x1))
        # x3 = F.tanh(self.conv3(x2))
        # x4 = F.tanh(self.conv4(x3))
        # x = F.tanh(self.reverseconv4(x4))
        # x = torch.cat((x3,x),1)
        # x = F.tanh(self.reverseconv3(x))
        # x = torch.cat((x2,x),1)
        # x = F.tanh(self.reverseconv2(x))
        # x = torch.cat((x1,x),1)
        # x = F.tanh(self.reverseconv1(x))


        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))

        # x = self.dropout(F.tanh(self.fc1(x)))
        # x = self.dropout(F.tanh(self.fc2(x)))
        # x = self.dropout(F.tanh(self.fc3(x)))
        # x = x.reshape(x.shape[0], self.config.num_agents*self.config.num_items)
        # x = F.relu(self.enc1(x))
        # x = F.relu(self.enc2(x))
        # x = F.relu(self.enc3(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec1(x))
        # x = x.reshape(x.shape[0], self.config.num_agents,self.config.num_items)

        # x = self.pool(F.tanh(self.conv1(x)))
        # x = self.pool(F.tanh(self.conv2(x)))
        # x = x.view(-1, 16 * 1 * 4)
        # x = F.tanh(self.fc1(x))
        # x = F.tanh(self.fc2(x))
        # x = self.fc3(x)
        # x = x.reshape(x.shape[0], self.config.num_agents,self.config.num_items)

        # n = len(netList)

        # conv00 = netList[0].state_dict()['conv.0.weight']
        # conv01 = netList[0].state_dict()['conv.1.weight']
        # conv02 = netList[0].state_dict()['conv.2.weight']
        # conv03 = netList[0].state_dict()['conv.3.weight']
        # deconv00 = netList[0].state_dict()['deconv.0.weight']
        # deconv01 = netList[0].state_dict()['deconv.1.weight']
        # deconv02 = netList[0].state_dict()['deconv.2.weight']
        # deconv03 = netList[0].state_dict()['deconv.3.weight']

        # conv10 = netList[0].state_dict()['conv1.0.weight']
        # conv11 = netList[0].state_dict()['conv1.1.weight']
        # conv12 = netList[0].state_dict()['conv1.2.weight']
        # conv13 = netList[0].state_dict()['conv1.3.weight']
        # deconv10 = netList[0].state_dict()['deconv1.0.weight']
        # deconv11 = netList[0].state_dict()['deconv1.1.weight']
        # deconv12 = netList[0].state_dict()['deconv1.2.weight']
        # deconv13 = netList[0].state_dict()['deconv1.3.weight']

        # conv20 = netList[0].state_dict()['conv2.0.weight']
        # conv21 = netList[0].state_dict()['conv2.1.weight']
        # conv22 = netList[0].state_dict()['conv2.2.weight']
        # conv23 = netList[0].state_dict()['conv2.3.weight']
        # deconv20 = netList[0].state_dict()['deconv2.0.weight']
        # deconv21 = netList[0].state_dict()['deconv2.1.weight']
        # deconv22 = netList[0].state_dict()['deconv2.2.weight']
        # deconv23 = netList[0].state_dict()['deconv2.3.weight']



        #     conv00 += netList[netindex].state_dict()['conv.0.weight']
        #     conv01 += netList[netindex].state_dict()['conv.1.weight']
        #     conv02 += netList[netindex].state_dict()['conv.2.weight']
        #     conv03 += netList[netindex].state_dict()['conv.3.weight']
        #     deconv00 += netList[netindex].state_dict()['deconv.0.weight']
        #     deconv01 += netList[netindex].state_dict()['deconv.1.weight']
        #     deconv02 += netList[netindex].state_dict()['deconv.2.weight']
        #     deconv03 += netList[netindex].state_dict()['deconv.3.weight']

        #     conv10 += netList[netindex].state_dict()['conv1.0.weight']
        #     conv11 += netList[netindex].state_dict()['conv1.1.weight']
        #     conv12 += netList[netindex].state_dict()['conv1.2.weight']
        #     conv13 += netList[netindex].state_dict()['conv1.3.weight']
        #     deconv10 += netList[netindex].state_dict()['deconv1.0.weight']
        #     deconv11 += netList[netindex].state_dict()['deconv1.1.weight']
        #     deconv12 += netList[netindex].state_dict()['deconv1.2.weight']
        #     deconv13 += netList[netindex].state_dict()['deconv1.3.weight']

        #     conv20 += netList[netindex].state_dict()['conv2.0.weight']
        #     conv21 += netList[netindex].state_dict()['conv2.1.weight']
        #     conv22 += netList[netindex].state_dict()['conv2.2.weight']
        #     conv23 += netList[netindex].state_dict()['conv2.3.weight']
        #     deconv20 += netList[netindex].state_dict()['deconv2.0.weight']
        #     deconv21 += netList[netindex].state_dict()['deconv2.1.weight']
        #     deconv22 += netList[netindex].state_dict()['deconv2.2.weight']
        #     deconv23 += netList[netindex].state_dict()['deconv2.3.weight']



        
        # netList[0].state_dict()['conv.0.weight'].data.copy_(( conv00/(n+1)  ))
        # netList[0].state_dict()['conv.1.weight'].data.copy_(( conv01/(n+1)  ))
        # netList[0].state_dict()['conv.2.weight'].data.copy_(( conv02/(n+1)  ))
        # netList[0].state_dict()['conv.3.weight'].data.copy_(( conv03/(n+1)  ))

        # netList[0].state_dict()['deconv.0.weight'].data.copy_(( deconv00/(n+1)  ))
        # netList[0].state_dict()['deconv.1.weight'].data.copy_(( deconv01/(n+1)  ))
        # netList[0].state_dict()['deconv.2.weight'].data.copy_(( deconv02/(n+1)  ))
        # netList[0].state_dict()['deconv.3.weight'].data.copy_(( deconv03/(n+1)  ))

        # netList[0].state_dict()['conv1.0.weight'].data.copy_(( conv10/(n+1)  ))
        # netList[0].state_dict()['conv1.1.weight'].data.copy_(( conv11/(n+1)  ))
        # netList[0].state_dict()['conv1.2.weight'].data.copy_(( conv12/(n+1)  ))
        # netList[0].state_dict()['conv1.3.weight'].data.copy_(( conv13/(n+1)  ))

        # netList[0].state_dict()['deconv1.0.weight'].data.copy_(( deconv10/(n+1))  )
        # netList[0].state_dict()['deconv1.1.weight'].data.copy_(( deconv11/(n+1) ) )
        # netList[0].state_dict()['deconv1.2.weight'].data.copy_(( deconv12/(n+1)  ))
        # netList[0].state_dict()['deconv1.3.weight'].data.copy_(( deconv13/(n+1)  ))


        # netList[0].state_dict()['conv2.0.weight'].data.copy_(( conv20/(n+1)  ))
        # netList[0].state_dict()['conv2.1.weight'].data.copy_(( conv21/(n+1)  ))
        # netList[0].state_dict()['conv2.2.weight'].data.copy_(( conv22/(n+1)  ))
        # netList[0].state_dict()['conv2.3.weight'].data.copy_(( conv23/(n+1)  ))

        # netList[0].state_dict()['deconv2.0.weight'].data.copy_(( deconv20/(n+1)  ))
        # netList[0].state_dict()['deconv2.1.weight'].data.copy_(( deconv21/(n+1)  ))
        # netList[0].state_dict()['deconv2.2.weight'].data.copy_(( deconv22/(n+1)  ))
        # netList[0].state_dict()['deconv2.3.weight'].data.copy_(( deconv23/(n+1)  ))

        # netList = [netList[0]]
