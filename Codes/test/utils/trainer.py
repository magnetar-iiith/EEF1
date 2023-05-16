# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

from utils import util

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


class Trainer():
    def __init__(self, net, config,logger_file_name):
        self.config = config
        self.logger_file_name = logger_file_name
        self.net = net 
        self.util =  util.Util(config,logger_file_name)
        self.model_path = 'models/' + logger_file_name + '.pth'
        logger_path = 'logs/' + logger_file_name +'.log'
        self._max_norm_val = 5
        logger1.addHandler(logging.FileHandler(filename=logger_path))
        self._max_norm_vals = 4
        self.softmaxfn = nn.Softmax(dim=1).to(device)


    def train(self,lossfunctionId):   


        print(self.logger_file_name)
        print()
        print('_____________________________________________________________________')
        print(self.config)
        print('_____________________________________________________________________')
        print()

        schedulervalue = self.config.train.scheduler 
        train_iter = self.config.train.max_iter
        if(self.config.optim == 0):
            optimizer_net_w  = optim.Adam(self.net.parameters(), self.config.train.lr)
        elif(self.config.optim == 1):
            optimizer_net_w  = optim.SGD(self.net.parameters(), lr=self.config.train.lr, momentum=0.99)        
        # torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99
        reloadbestmodel = self.config.train.reloadbestmodel
        fastAbsoluteLoss = self.config.train.fastAbsoluteLoss

        lambda_lag_init = (torch.ones([self.config.num_agents,1]).to(device) ) * self.config.train.lambda_init_val
        lambda_lag = [torch.tensor(  lambda_lag_init  ,requires_grad=True  )]
        
        optimizer_lag = optim.SGD(lambda_lag, lr=self.config.train.pho)

        
        time_elapsed = 0.0
        running_loss = 0.0
        abs_running_loss = 0.0
        envy_loss = 0.0
        sw_total = 0.0

        running_loss_list = []
        abs_running_loss_list = []
        itr_list = []

        
        minimumloss = 99999999999999
        maxsw = 0.0
        bestmodel = self.net.state_dict()

        mulfactor1 = 1 * self.config.train.basicmulfactor #9
        mulfactor2 = 11 * self.config.train.basicmulfactor #99
        mulfactor3 = 111 * self.config.train.basicmulfactor #999
        mulfactor4 = 1111 * self.config.train.basicmulfactor #9999
        mulfactor5 = 11111 * self.config.train.basicmulfactor #9999

        # scheduler = optim.lr_scheduler.StepLR(optimizer_net_w, step_size=self.config.train.step_size, gamma=self.config.train.gamma)

        ### Decide temperature factor based on the config 
        ### get the allocation using neural network
        ### calculate loss, and envy of frac allocation and discrete allocation, and social welfare
        ### loss.backward
        ### if the minimum abs allocation envy is greater than current , then save the current model

        startiter = 200000
        for _ in range(train_iter):
            tic = time.time()   
        
            # if( _ < 15000 ):
            #     self.config.valuesmul = 1
            # else:
            #     self.config.valuesmul = 0
                
            #1
            if(self.config.temperature == 1) :
                mulfactor = 1
            elif(self.config.temperature == 11) :
                mulfactor = 9
            elif(self.config.temperature == 111) :
                mulfactor = 99
            elif(self.config.temperature == 1111) :
                mulfactor = 999
            elif(self.config.temperature == 11111) :
                mulfactor = 9999
            elif(self.config.temperature == 2) :
                if ((_%100) < (50)):
                    mulfactor = mulfactor1
                else:
                    mulfactor = mulfactor2
            elif(self.config.temperature == 3) :
                if ((_%75) < (25)):
                    mulfactor = mulfactor2
                elif ((_%75) < (50)):
                    mulfactor = mulfactor3
                else:
                    mulfactor = mulfactor4
            elif(self.config.temperature == 4) :
                if ((_%100) < (25)):
                    mulfactor = 999
                elif ((_%100) < (50)):
                    mulfactor = 9999
                elif ((_%100) < (75)):
                    mulfactor = 99999
                else:
                    mulfactor = 999999
            elif(self.config.temperature == 5):
                # if(_ == 0):
                #     self.test()
                if( _ < 7000):
                    mulfactor = 999
                elif( _ < 20000):
                    if(_ == 7000):
                        print('__________________________')
                        self.test()
                        print('__________________________')
                        print('')
                        # for param_group in optimizer_net_w.param_groups:
                        #     param_group['lr'] = 0.00001
                        # for param_group in optimizer_net_w.param_groups:
                        #     print(param_group['lr'])
                    mulfactor = 9999
                elif( _ < 55000):
                    if(_ == 20000):
                        print('__________________________')
                        self.test()
                        print('__________________________')
                    mulfactor = 99999
                elif ( _ < 150000):
                    if(_ == 125000):
                        print('__________________________')
                        self.test()
                        print('__________________________')
                    mulfactor = 999999
                # elif( _ == 50000):
                #     print('__________________________')
                #     self.test()
                #     print('__________________________')
                
            elif(self.config.temperature == 6):
                if( _ < 100000):
                    mulfactor = 999
                elif( _ < 200000 ):
                    mulfactor = 99
                else:
                    mulfactor = 9
            elif(self.config.temperature == 9393):
                if( _ < 200000):
                    lossfunctionId = 3  ## Max sw + EF1
                    mulfactor = 99
                elif( _ < 300000):
                    if(_ == 200000):
                        print('')
                        for param_group in optimizer_net_w.param_groups:
                            param_group['lr'] = 0.0001
                        for param_group in optimizer_net_w.param_groups:
                            print(param_group['lr'])
                        self.test()
                    lossfunctionId = 2  ## EF1
                    if(_ < 250000):
                        mulfactor = 9999
                    else:
                        mulfactor = 99999
                elif( _ < 400000):
                    if(_ == 300000):
                        for param_group in optimizer_net_w.param_groups:
                            param_group['lr'] = 0.001
                        for param_group in optimizer_net_w.param_groups:
                            print(param_group['lr'])
                        self.test()
                    lossfunctionId = 3  ## Max sw + EF1
                    mulfactor = 99
                elif( _ < 500000):
                    if(_ == 400000):
                        for param_group in optimizer_net_w.param_groups:
                            param_group['lr'] = 0.0001
                        for param_group in optimizer_net_w.param_groups:
                            print(param_group['lr'])
                        self.test()
                    mulfactor = 99
                    lossfunctionId = 1  ## Max sw
                elif( _  < 550000):
                    if(_ == 400000):
                        for param_group in optimizer_net_w.param_groups:
                            param_group['lr'] = 0.001
                        for param_group in optimizer_net_w.param_groups:
                            print(param_group['lr'])
                        self.test()
                    lossfunctionId = 3                    
                    mulfactor = 99
                if( _ == 550000):
                    print(' _' , _)
                    self.test()
                    _ = 0
                    train_iter = 540000

            # batch_X, idx = next(self.get_batch_data_next)
            batch_X, allocation_true = next(self.get_batch_data_next)

            # print('batchx shape', batch_X.shape)
            n_samples = batch_X.shape[0]
            optimizer_net_w.zero_grad()

            loss = 0
            abs_loss = 0

            if(fastAbsoluteLoss == 0):
                fastAbsoluteLoss = False
            else:
                fastAbsoluteLoss = True


            # print(torch.cat((batch_X[:,None,:,:],allocation_true[:,None,:,:].float()),1).shape)
            if(self.config.numberofchannels == 21):
                twoDvaluations = torch.cat((batch_X[:,None,:,:],allocation_true[:,None,:,:]),1)
            elif(self.config.numberofchannels == 22):
                twoDvaluations = torch.cat((batch_X[:,None,:,:],allocation_true[:,None,:,:]),1)
            elif(self.config.numberofchannels == 31):
                # twoDvaluations = torch.cat((batch_X[:,None,:,:],allocation_true[0][:,None,:,:],allocation_true[1][:,None,:,:]),1)
                twoDvaluations = torch.cat((allocation_true[0][:,None,:,:],batch_X[:,None,:,:],allocation_true[1][:,None,:,:]),1)

            elif(self.config.numberofchannels == 32):
                twoDvaluations = torch.cat((batch_X[:,None,:,:],allocation_true[0][:,None,:,:],allocation_true[1][:,None,:,:]),1)
            elif(self.config.numberofchannels == 33):
                twoDvaluations = torch.cat((batch_X[:,None,:,:],allocation_true[1][:,None,:,:],allocation_true[0][:,None,:,:]),1)


            allocation = self.net(twoDvaluations,mulfactor )

            ## JUST EF1
            if(lossfunctionId == 1 ):
                loss ,lag_loss , envyness, sw = self.util.loss_function_ef1(allocation, batch_X,lambda_lag[0])
                dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
                abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
                envy_loss += float(envyness)
                sw_total += float(sw)
                running_loss += float(loss)
                abs_running_loss += float(abs_loss)
            ## MAX SW wrt EF1
            elif(lossfunctionId == 3 ):
                loss ,lag_loss , envyness, sw = self.util.loss_function_max_sw_wrt_ef1(allocation, batch_X,lambda_lag[0])
                dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
                abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
                envy_loss += float(envyness)
                sw_total += float(sw)
                running_loss += float(loss)
                abs_running_loss += float(abs_loss)
            ## MAX SW wrt EF1
            elif(lossfunctionId == 4 ):
                loss ,lag_loss , envyness, sw = self.util.loss_function_max_sw_wrt_ef(allocation, batch_X,lambda_lag[0])
                dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
                abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
                envy_loss += float(envyness)
                sw_total += float(sw)
                running_loss += float(loss)
                abs_running_loss += float(abs_loss)

            ## MAX SW
            elif(lossfunctionId == 2):
                loss ,lag_loss , envyness, sw = self.util.loss_function_max_sw(allocation, batch_X,lambda_lag[0])
                dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
                abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
                envy_loss += float(envyness)
                sw_total += float(sw)
                running_loss += float(loss)
                abs_running_loss += float(abs_loss)
            elif(lossfunctionId == 5):
                loss ,envyness, sw = self.util.labellearningMUW(allocation, allocation_true,batch_X)
                dis_alloc = self.util.getDiscreteAllocationParam(allocation, batch_X,fastAbsoluteLoss)
                abs_loss = self.util.loss_function_max_ef1envyfree(dis_alloc,batch_X)
                envy_loss += float(envyness)
                sw_total += float(sw)
                running_loss += float(loss)
                abs_running_loss += float(abs_loss)
                
            
            count = 0

            loss.backward(retain_graph=True)
            optimizer_net_w.step()


            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ( _ % self.config.num_batches) == 0:
                # running_loss_list.append(running_loss)
                # abs_running_loss_list.append(abs_running_loss)
                # itr_list.append(_)
                # if(lossfunctionId == 1) : 
                #     print('__________________________________________________')
                #     sw_total = sw_total / self.config.num_batches
                #     summary = "TRAIN-BATCH Iter: {}, running : {} , sw : {} , envyness : {}, running abs : {}, current batch loss : {}     t ={:.4f}"
                #     print(summary.format(_,running_loss,sw_total,envy_loss,abs_running_loss,loss,time_elapsed))                
                # elif (lossfunctionId == 3):
                #     print('__________________________________________________')
                #     sw_total = sw_total / self.config.num_batches
                #     summary = "TRAIN-BATCH Iter: {}, running : {} , sw : {} , envyness : {}, running abs : {}, current batch loss : {}     t ={:.4f}"
                #     print(summary.format(_,running_loss,sw_total,envy_loss,abs_running_loss,loss,time_elapsed))                
                # elif (lossfunctionId == 2):
                print('__________________________________________________')
                sw_total = sw_total / self.config.num_batches
                summary = "TRAIN-BATCH Iter: {}, running : {} , sw : {} , envyness : {}, running abs : {}, current batch loss : {}     t ={:.4f}"
                print(summary.format(_,running_loss,sw_total,envy_loss,abs_running_loss,loss,time_elapsed))
                print()
                #     # print(allocation[0])
                # else:
                #     summary = "TRAIN-BATCH Iter: {}, running : {:.4f} , running abs : {:.4f}, current batch loss : {:.4f}     t ={:.4f}"
                #     print(summary.format(_,running_loss,abs_running_loss,loss,time_elapsed))

                # print()
                # print('    Conv ')
                # print()
                # print('__________________________________________________')
                # print(self.net.conv[0].weight.grad)
                # print(self.net.conv[1].weight.grad)
                # print(self.net.conv[2].weight.grad)
                # print(self.net.conv[3].weight.grad)
                # print()
                # print('    DeConv ')
                # print()
                # print(self.net.deconv[0].weight.grad)
                # print(self.net.deconv[1].weight.grad)
                # print(self.net.deconv[2].weight.grad)
                # print(self.net.deconv[3].weight.grad)
                # print('__________________________________________________')
                # print()
                # print('    Conv1 ')
                # print()
                # print(self.net.conv1[0].weight.grad)
                # print(self.net.conv1[1].weight.grad)
                # print(self.net.conv1[2].weight.grad)
                # print(self.net.conv1[3].weight.grad)
                # print()
                # print('    DeConv1 ')
                # print()
                # print(self.net.deconv1[0].weight.grad)
                # print(self.net.deconv1[1].weight.grad)
                # print(self.net.deconv1[2].weight.grad)
                # print(self.net.deconv1[3].weight.grad)
                # print('__________________________________________________')
                # print()
                # print('    Conv2 ')
                # print()
                # print(self.net.conv2[0].weight.grad)
                # print(self.net.conv2[1].weight.grad)
                # print(self.net.conv2[2].weight.grad)
                # print(self.net.conv2[3].weight.grad)
                # print()
                # print('    DeConv2 ')
                # print()
                # print(self.net.deconv2[0].weight.grad)
                # print(self.net.deconv2[1].weight.grad)
                # print(self.net.deconv2[2].weight.grad)
                # print(self.net.deconv2[3].weight.grad)
                # print('__________________________________________________')
                # print()
                # print()
                # print(' Allocation   ')
                # print(allocation[0])

                if(lossfunctionId == 1 or lossfunctionId == 3 or lossfunctionId == 4):
                    if( (_ != 0) and (minimumloss > abs_running_loss) ):
                        minimumloss = abs_running_loss
                        print(' Minimum Loss ', minimumloss)
                        print('saving the best model ')
                        print()
                        print()
                        torch.save(self.net.state_dict(), self.model_path)
                else:
                    if( (_ != 0) and (maxsw < sw_total) ):
                        maxsw = sw_total
                        print(' Max SW ', maxsw)
                        print('saving the best model ')
                        print()
                        print()
                        print('saving net')
                        torch.save(self.net.state_dict(), self.model_path)



                running_loss = 0.0
                abs_running_loss = 0.0
                envy_loss = 0.0
                sw_total = 0.0




            if((_!=0) and schedulervalue == 1):

                # if( _ < 1000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.1
                # elif( _ < 5000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.01
                # elif( _ < 450000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.001
                # elif( _ < 750000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.0001
                # else:
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.00001

                # if(_ % 8000 == 0):
                #     self.config.temperature = 1111
                #     print('temperature change', self.config.temperature)
                # elif(_ % 11000 == 0):
                #     self.config.temperature = 111
                #     print('temperature change', self.config.temperature)

                # if( _ < 2000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.1
                # if( _ < 6000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.01
                # elif( _ < 405000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.001
                # elif( _ < 510000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.0001
                # elif( _ < 511000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.1
                # elif( _ < 516000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.01
                # elif( _ < 816000):
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.001
                # else:
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 0.0001
                    

                # if((_ % 510000) == 0):
                #     print('reseting')
                #     model_path = 'models/' + self.logger_file_name + '1' + '.pth'
                #     print('Loading back the best model')
                #     state_dict = torch.load(self.model_path)
                #     self.net.load_state_dict(state_dict)
                #     self.test()
                # if((_ % 900000) == 0):
                #     print('reseting')
                #     model_path = 'models/' + self.logger_file_name + '2' + '.pth'
                #     print('Loading back the best model')
                #     state_dict = torch.load(self.model_path)
                #     self.net.load_state_dict(state_dict)
                #     self.test()
                    

                    

                # if( _% self.config.train.resetlr == 0):
                #     count += 1
                #     for param_group in optimizer_net_w.param_groups:
                #         param_group['lr'] = 1

                #     print('reseting')
                #     model_path = 'models/' + self.logger_file_name + str(count) + '.pth'
                #     print('Loading back the best model')
                #     state_dict = torch.load(self.model_path)
                #     self.net.load_state_dict(state_dict)

                #     torch.save(self.net.state_dict(), model_path)
                #     self.test()
                #     print('next.......')
                #     # self.config.train.step_size = 0.1
                #     # scheduler = optim.lr_scheduler.StepLR(optimizer_net_w, step_size=self.config.train.step_size, gamma=self.config.train.gamma)
                #     # print()
                #     # print(scheduler)
                #     # print()
                #     for param_group in optimizer_net_w.param_groups:
                #         print(param_group['lr'])

                if(_ % self.config.train.step_size == 0):
                    # for g in optim.param_groups:
                    #     g['lr'] = g['lr'] * self.config.train.gamma
                    for param_group in optimizer_net_w.param_groups:
                        param_group['lr'] = param_group['lr'] * self.config.train.gamma

                    for param_group in optimizer_net_w.param_groups:
                        print(param_group['lr'])



                # scheduler.step()



            # if( (_ != 0) and ((_ % reloadbestmodel) == 0 )):
            #     print('Loading the best model')
            #     state_dict = torch.load(self.model_path)
            #     self.net.load_state_dict(state_dict)

        return allocation ,  loss , [itr_list,running_loss_list,abs_running_loss_list] , bestmodel, batch_X


    def test( self):
        print('Loading back the best model')
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)


        num_test_batches = 3
        b_EF1 = []
        a_EF1 = []
        for i in range(num_test_batches):
            n_samples = 20000
            test_data_shape = [n_samples , self.config.num_agents , self.config.num_items]

            ## CHANGE HERE
            # X_test = self.util.generate_random_X(test_data_shape)
            # X_test = self.util.generate_random_X(test_data_shape)
            X_test =np.random.uniform(0.0, 1.0, size = test_data_shape)
            X_test =  torch.from_numpy(X_test).float()

            # X_test = torch.from_numpy(X_test).float()
            X_test = X_test.to(device)

            # loss_test, allocation_test= self.test(lossfunctionid ,X_test)
            # loss_test, allocation_test= test(model, lossfunctionid ,X_test)
            lambda_lag_init = (torch.ones([self.config.num_agents,1]).to(device) ) * self.config.train.lambda_init_val
            lambda_lag = [torch.tensor(  lambda_lag_init  ,requires_grad=True  )]

            print('#######################################################################')

            # muwallocation = self.util.findMUW_all(X_test).float()
            # muwallocation = self.util.findRandomAllocation_all(X_test).float()
            if(self.config.numberofchannels == 21):
                muwallocation = self.util.findMUW_all(X_test).float()
            if(self.config.numberofchannels == 22):
                muwallocation = self.util.findRandomAllocation_all(X_test).float()
            if(self.config.numberofchannels == 31):
                muwallocation = self.util.findMUW_all(X_test).float()
                rr = self.util.findRandomAllocation_all(X_test).float()
                # rr = self.util.ef1envyfree_greedy_output(X_test).float()
            elif(self.config.numberofchannels == 32):
                wcrr = self.util.findWCRR_all(X_test).float()
            elif(self.config.numberofchannels == 33):
                rr = self.util.ef1envyfree_greedy_output(X_test).float()

            # print(torch.cat((batch_X[:,None,:,:],allocation_true[:,None,:,:].float()),1).shape)
            # twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:].float()),1)
            if(self.config.numberofchannels == 21):
                twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:]),1)
            elif(self.config.numberofchannels == 22):
                twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:]),1)
            elif(self.config.numberofchannels == 31):
                # twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:],rr[:,None,:,:]),1)
                twoDvaluations = torch.cat((muwallocation[:,None,:,:],X_test[:,None,:,:],rr[:,None,:,:]),1)
            elif(self.config.numberofchannels == 32):
                twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:],wcrr[:,None,:,:]),1)
            elif(self.config.numberofchannels == 33):
                twoDvaluations = torch.cat((X_test[:,None,:,:],rr[:,None,:,:],muwallocation[:,None,:,:]),1)

            print(twoDvaluations.shape)
            allocation_test = self.net(twoDvaluations)
            loss ,lag_loss , envyness, sw = self.util.loss_function_max_sw_wrt_ef1(allocation_test, X_test,lambda_lag[0])


            # discreteAllocationTest = 99999999999 * allocation_test
            # discreteAllocationTest = self.softmaxfn(discreteAllocationTest)
            ### change here
            discreteAllocationTest = self.util.getDiscreteAllocation(allocation_test)


            uniquevalues = torch.unique(discreteAllocationTest, dim=0)

            print(' ')
            print('_______________________________________________________')
            print(' Unique Allocation during test ')
            print(len(uniquevalues))
            # print(uniquevalues)
            print('_______________________________________________________')

            print(' With fractional allocation ')
            testsumenvy, testcurrentenvy = self.util.calculateEnvySamplewise(allocation_test,X_test)
            testenvyindex = torch.nonzero(testcurrentenvy)
            count =  n_samples - len(testenvyindex)
            print('___________________________________________')
            print(' Total Envy Free samples ', count)
            print(' Total Envy Free samples percent', (count/n_samples))

            print('SW  frac::', torch.mean(self.util.calculate_u_social_welfare(allocation_test,X_test)) )
            print()


            print(' With binary allocation - EF1 ')
            print('_______________________________________________________')

            testsumenvy, testcurrentenvy = self.util.calculateEnvySamplewise(discreteAllocationTest,X_test)
            testenvyindex = torch.nonzero(testcurrentenvy)
            count =  n_samples - len(testenvyindex)
            print('___________________________________________')
            print(' Total Envy Free samples ', count)
            print(' Total Envy Free samples percent', (count/n_samples))
            a_EF1.append(count/n_samples)
            _sw = torch.mean(self.util.calculate_u_social_welfare(discreteAllocationTest,X_test))

            print('SW ::', _sw )
            print()

            b_EF1.append(_sw)

        print('For n_batches', num_test_batches , 'a_EF1  ', sum(a_EF1)/num_test_batches)
        print('For n_batches', num_test_batches , 'b_EF1  ', sum(b_EF1)/num_test_batches)


    
        # return loss , allocation

    def runTheNetwork(self):
        np.random.seed(self.config.train.seed)
        train_data_shape = [self.config.num_samples , self.config.num_agents , self.config.num_items]
        X = self.util.generate_random_X_good_uniform(train_data_shape)
        X = X.to(device)

        lossfunctionid = self.config.lossfunctionId

        # self.test()
        self.get_batch_data_next = self.util.generate_batch_data(X)

        allocation, loss, plot_details , bestmodel, batch_X= self.train(lossfunctionid)


        print('Loading back the best model')
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict)

        # if(self.config.train.dropout):
        #     print('Named para')
        #     eps=1e-8
        #     for name, param in self.net.named_parameters():
        #         if 'bias' not in name:
        #             norm = param.norm(2, dim=0, keepdim=True)
        #             print('norm is', norm)
        #             desired = torch.clamp(norm, 0, self._max_norm_vals)
        #             param = param * (desired / (eps + norm))

        # self.config.train.dropout = 0

            # norm = w.norm(2, dim=0, keepdim=True).clamp(min=self._max_norm_val / 2)
            # desired = torch.clamp(norm, max=self._max_norm_val)
            # w *= (desired / norm)


        perm = np.random.permutation(X.shape[0]) 
        idx = perm[0 : 10000]
        batch_X = X[idx]

        if(self.config.numberofchannels == 21):
            muwallocation = self.util.findMUW_all(batch_X).float()
        if(self.config.numberofchannels == 22):
            muwallocation = self.util.findRandomAllocation_all(batch_X).float()
        if(self.config.numberofchannels == 31):
            muwallocation = self.util.findMUW_all(batch_X).float()
            rr = self.util.findRandomAllocation_all(batch_X).float()
        elif(self.config.numberofchannels == 32):
            wcrr = self.util.findWCRR_all(batch_X)
        elif(self.config.numberofchannels == 33):
            rr = self.util.ef1envyfree_greedy_output(batch_X).float()
            
        # print(torch.cat((batch_X[:,None,:,:],allocation_true[:,None,:,:].float()),1).shape)
        # twoDvaluations = torch.cat((X_test[:,None,:,:],muwallocation[:,None,:,:].float()),1)
        if(self.config.numberofchannels == 21):
            twoDvaluations = torch.cat((batch_X[:,None,:,:],muwallocation[:,None,:,:]),1)
        elif(self.config.numberofchannels == 22):
            twoDvaluations = torch.cat((batch_X[:,None,:,:],muwallocation[:,None,:,:]),1)
        elif(self.config.numberofchannels == 31):
            # twoDvaluations = torch.cat((batch_X[:,None,:,:],muwallocation[:,None,:,:],rr[:,None,:,:]),1)
            twoDvaluations = torch.cat((muwallocation[:,None,:,:],batch_X[:,None,:,:],rr[:,None,:,:]),1)
        elif(self.config.numberofchannels == 32):
            twoDvaluations = torch.cat((batch_X[:,None,:,:],muwallocation[:,None,:,:],wcrr[:,None,:,:]),1)
        elif(self.config.numberofchannels == 33):
            twoDvaluations = torch.cat((batch_X[:,None,:,:],rr[:,None,:,:],muwallocation[:,None,:,:]),1)


        allocation = self.net(twoDvaluations)

        print('allocation')
        print(allocation[0])
        print()
        print()
        print(allocation[1])
        print()
        print()
        print(allocation[2])
        print()
        print()

        print(' Training Loss ')
        print('_______________________________________________________')
        print('Frac loss')
        trainsumenvy, traincurrentenvy = self.util.calculateEnvySamplewise(allocation,batch_X)
        trainenvyindex = torch.nonzero(traincurrentenvy)
        count =  allocation.shape[0] - len(trainenvyindex)
        print('___________________________________________')
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/allocation.shape[0]))
        print('SW  frac::', torch.mean(self.util.calculate_u_social_welfare(allocation,batch_X)) )
        print()



        discreteAllocationTrain = self.util.getDiscreteAllocation(allocation)

        # print('_______________________________________________________')
        
        print()
        print('Discrete Loss')
        print()
        trainsumenvy, traincurrentenvy = self.util.calculateEnvySamplewise(discreteAllocationTrain,batch_X)
        trainenvyindex = torch.nonzero(traincurrentenvy)
        count =  allocation.shape[0] - len(trainenvyindex)
        print('___________________________________________')
        print(' Total Envy Free samples ', count)
        print(' Total Envy Free samples percent', (count/allocation.shape[0]))
        print('SW  dis::', torch.mean(self.util.calculate_u_social_welfare(discreteAllocationTrain,batch_X)) )
        print()

        print()
        print()


        self.test()

        









            #2
            # if ((_%100) < (50)):
            #     mulfactor = mulfactor1
            # else:
            #     mulfactor = mulfactor2

            #3
            # if ((_%150) < (50)):
            #     mulfactor = mulfactor1
            # elif (  (_%150) < (100) ):
            #     mulfactor = mulfactor2
            # else:
            #     mulfactor = mulfactor3


            #4
            # if ((_%200) < (100)):
            #     mulfactor = mulfactor1
            # else:
            #     mulfactor = mulfactor2

            #5
            # if ((_%150) < (50)):
            #     mulfactor = 5
            # elif (  (_%150) < (100) ):
            #     mulfactor = 50
            # else:
            #     mulfactor = 150

            #6
            # if ((_%20) < (10)):
            #     mulfactor = mulfactor1
            # else:
            #     mulfactor = mulfactor2


            # if ((_%100) < (25)):
            #     mulfactor = mulfactor1
            # elif (  (_%100) < (50) ):
            #     mulfactor = mulfactor2
            # elif (  (_%100) < (75) ):
            #     mulfactor = mulfactor3
            # else:
            #     mulfactor = mulfactor4

            
            ###############3
            # if ((_%75) < (25)):
            #     mulfactor = mulfactor1
            # elif (  (_%75) < (50) ):
            #     mulfactor = mulfactor2
            # # elif (  (_%100) < (75) ):
            # #     mulfactor = mulfactor3
            # else:
            #     mulfactor = mulfactor3


            #######################
            # if ((_%225) < (75)):
            #     mulfactor = mulfactor1
            # elif (  (_%225) < (150) ):
            #     mulfactor = mulfactor2
            # else:
            #     mulfactor = mulfactor3


            # if ((_%120) < (25)):
            #     mulfactor = mulfactor1
            # elif (  (_%120) < (50) ):
            #     mulfactor = mulfactor2
            # elif (  (_%120) < (75) ):
            #     mulfactor = mulfactor3
            # elif (  (_%120) < (100) ):
            #     mulfactor = mulfactor4
            # else:
            #     mulfactor = mulfactor5


            # if ((_%200) < (40)):
            #     mulfactor = 9
            # elif (  (_%200) < (80) ):
            #     mulfactor = 99
            # elif (  (_%200) < (120) ):
            #     mulfactor = 999
            # elif (  (_%200) < (160) ):
            #     mulfactor = 9999
            # else:
            #     mulfactor = 99999



            # if ((_%200) < (80)):
            #     mulfactor = mulfactor1
            # elif (  (_%200) < (150) ):
            #     mulfactor = mulfactor2
            # elif (  (_%200) < (175) ):
            #     mulfactor = mulfactor3
            # else:
            #     mulfactor = mulfactor4


