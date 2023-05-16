
import numpy as np
import gurobipy as gp
from gurobipy import *
from gurobipy import GRB

try:
    rangeofAgents = 7
    startagent = 7
    num_items = 100
    # num_instances = 1000
    infeasiblecount=0
    # num_agents = 15
    # num_items = 30
    
    runtimeaverage = []
    for num_agents in range(startagent,rangeofAgents+1):
        runtimeperinstance = []
        sw = []
        for instance in range(num_instances):
            shape =[num_agents,num_items]

            valuation =np.random.uniform(0.0, 1.0, size = shape)
            # valuation =np.random.uniform(-1.0, 0.0, size = shape)

            # valuation =np.random.uniform(-1.0, 1.0, size = shape)

            # print(valuation)
            # create empty model
            m = gp.Model()
            m.Params.timeLimit = 100.0
            agent= []
            orconstraintvariable = []
            for agentid in range(num_agents):
                agentname = str('agent'+ str(agentid))
                agent.append(m.addVars(num_items, vtype=GRB.BINARY, name=agentname))

            ## Objection function = social welfare
            objective = 0
            for i in range(num_agents):
                objective += agent[i].prod(list(valuation[i]))
            m.setObjective( objective  , GRB.MAXIMIZE)
            
            # One items goes to one agent and all are allocated
            for j in range(num_items):
                constraint = 0
                for i in range(num_agents):
                    constraint += agent[i][j]
                m.addConstr(constraint  == 1)


            # EF1
            ### Need to maintain it separately and assign separately so that gurobi works
            ## Maxitemforagent [i][j] stores max of { agent[j] * valuation[i] }
            ## Multiplication terms contains multiplication list of all agent[j]*valuation[i]
            ## length is n^2   
            ## GOODS
            maxitemforeachagent = []
            multiplicationterm = []
            count = 0
            for i in range(num_agents):
                for j in range(num_agents):
                    maxitemforeachagent.append(m.addVars(num_agents, vtype=GRB.CONTINUOUS))
                    multiplicationterm.append(m.addVars(num_items, vtype=GRB.CONTINUOUS))
                    m.addConstrs((multiplicationterm[count][k] == agent[j][k]*valuation[i][k]) for k in range(num_items) )

                    ### Constraints
                    m.addConstr(maxitemforeachagent[i][j] == max_(multiplicationterm[count]))

                    constraint = (agent[i].prod(list(valuation[i])) - (agent[j].prod(list(valuation[i]))  -maxitemforeachagent[i][j]  ) )

                    m.addConstr(  constraint >= 0)

                    count += 1


            ### CHORES
            ## Multiplication terms contains multiplication list of all agent[i]*valuation[i]
            ## length is n^2   
            # minitemforeachagent = []
            # minitemforeachagent = m.addVars(num_agents,  lb=-1.0,vtype=GRB.CONTINUOUS)
            # multiplicationterm = []
            # # count = 0
            # for i in range(num_agents):
            #     multiplicationterm.append(m.addVars(num_items, vtype=GRB.CONTINUOUS, lb=-1.0))
            #     m.addConstrs((multiplicationterm[i][k] == agent[i][k]*valuation[i][k]) for k in range(num_items) )
            #     m.addConstr(minitemforeachagent[i] == min_(multiplicationterm[i]))
            #     for j in range(num_agents):

            #         ### Constraints

            #         constraint = (agent[i].prod(list(valuation[i])) -minitemforeachagent[i] - (agent[j].prod(list(valuation[i])) ) )

            #         m.addConstr(  constraint >= 0)

            #         # count += 1
                    

            # EF1
            ### Need to maintain it separately and assign separately so that gurobi works
            ## Maxitemforagent [i][j] stores max of { agent[j] * valuation[i] }
            ## Multiplication terms contains multiplication list of all agent[j]*valuation[i]
            ## length is n^2   
            ## GOODS
            # maxitemforeachagent = []
            # minitemforeachagent = m.addVars(num_agents,  lb=-1.0,vtype=GRB.CONTINUOUS)
            # multiplicationterm = []
            # minamongmaxandminitem = []
            # minamoungmaxandminitemarry = []
            # count = 0
            # for i in range(num_agents):
            #     multiplicationterm.append(m.addVars(2*num_items, vtype=GRB.CONTINUOUS, lb=-1.0))
            #     m.addConstrs((multiplicationterm[i][k] == agent[i][k]*valuation[i][k]) for k in range(num_items) )
            #     # m.addConstr(minitemforeachagent[i] == min_(multiplicationterm[i]))
            #     # m.addConstr(minamongmaxandminitem[i][j] >= -multiplicationterm[i])
            #     m.addConstrs((multiplicationterm[count][k] == agent[i][k]*valuation[i][k]) for k in range(num_items) )

            #     for j in range(num_agents):
            #         maxitemforeachagent.append(m.addVars(num_agents, vtype=GRB.CONTINUOUS))
            #         minamongmaxandminitem.append(m.addVars(num_agents, vtype=GRB.CONTINUOUS, lb=-1.0))
            #         multiplicationterm.append(m.addVars(num_items, vtype=GRB.CONTINUOUS))
            #         print('ok')
            #         # m.addConstrs((multiplicationterm[count][k] == agent[j][k]*valuation[i][k]) for k in range(num_items) )
            #         print('ok')
            #         # m.addConstrs((multiplicationterm[count][k] == agent[k2][k]*valuation[i][k]) for k2, k in zip([i,j], range(num_items)) )
                    
            #         for k in range(num_items):
            #             m.addConstrs((multiplicationterm[count][0,0] == agent[j][k]*valuation[i][k]) )
                        
            #         # m.addConstrs((multiplicationterm[count][k] == agent[k2][k]*valuation[i][k]) for k2, k in zip([i,j], range(num_items)) )


            #         # m.addConstrs((multiplicationterm[count][num_items:] == agent[j][k]*valuation[i][k]) for k in range(num_items) )
            #         print('ok')

            #         minamoungminandmax = m.addVar(lb=-1.0)    
                        
            #         ### Constraints
            #         # m.addConstr(maxitemforeachagent[i][j] == max_(multiplicationterm[count]))

            #         # m.addConstr(minamongmaxandminitem[i][j] >= maxitemforeachagent[i][j])
            #         # m.addConstr(minamongmaxandminitem[i][j] >= -minitemforeachagent[i][j])
            #         # m.addConstr(minamongmaxandminitem[i][j] >= and_(-maxitemforeachagent[i][j], minitemforeachagent[i]))
            #         m.addConstr(minamongmaxandminitem[i][j] <= min_( multiplicationterm[i] ))
            #         m.addConstr(minamongmaxandminitem[i][j] <= min_( -multiplicationterm[count] ))

            #         constraint = (agent[i].prod(list(valuation[i])) - agent[j].prod(list(valuation[i])  -minamongmaxandminitem[i][j]  ) )

            #         m.addConstr(  constraint >= 0)

            #         count += 1




            # m.write('2x4.lp')


            # solve model
            m.optimize()
            # display solution
            # if m.SolCount > 0:
            #     m.printAttr('X')
            # # export model
            # m.write('2x4.lp')

            # print(valuation)
            # for v in m.getVars():
            #     print('%s %g' % (v.varName, v.x))

            # print('Obj: %g' % m.objVal)
            # print('...print m.Runtime ', m.Runtime)
            if m.SolCount > 0:
                sw.append(m.objval)
                runtimeperinstance.append(m.Runtime)
            else: 
                print('No solution')
                infeasiblecount += 1
        print('sum ', sum(runtimeperinstance) )
        print('avg run time', sum(runtimeperinstance) / len(runtimeperinstance))
        print('avg sw', sum(sw) / len(sw))
        print('infesaible', infeasiblecount)

        runtimeaverage.append(sum(runtimeperinstance) / len(runtimeperinstance) )

        print('____________________________')
        # print(runtimeperinstance)
        print('max time',max(runtimeperinstance))
        print('min time',min(runtimeperinstance))

        # print('here')
        # print(m.objval)
        
        print()
        print('____________________________')
    print('____________________________')
    print(runtimeaverage)
    print('____________________________')

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')