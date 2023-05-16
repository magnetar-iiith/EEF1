
import numpy as np
import gurobipy as gp
from gurobipy import *
from gurobipy import GRB

try:
    rangeofAgents = 4
    startagent = 4
    num_items = 100
    num_instances = 30
    # num_agents = 15
    # num_items = 30
    
    runtimeaverage = []
    for num_agents in range(startagent,rangeofAgents+1):
        runtimeperinstance = []
        for instance in range(num_instances):
            shape =[num_agents,num_items]
            valuation =np.random.uniform(0.0, 1.0, size = shape)


            # create empty model
            m = gp.Model()

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
            count = 0
            for i in range(num_agents):
                for j in range(num_agents):
                    indicatorid = str('indicator'+ str(i) + str(j))
                    orconstraintvariable.append(m.addVars(num_items, vtype=GRB.BINARY, name=indicatorid))
                    constraint = 0
                    for k in range(num_items):
                        constraint += orconstraintvariable[count][k] *(agent[i].prod(list(valuation[i])) - (agent[j].prod(list(valuation[i])) - agent[j][k]*valuation[i][k]))

                    ### Adding or constraints for EF1 for agenti and agentj
                    ### any item
                    m.addConstr(constraint >= 0)
                    m.addConstr(orconstraintvariable[count].sum() == 1)
                    count += 1

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

            print('Obj: %g' % m.objVal)
            print('...print m.Runtime ', m.Runtime)
            runtimeperinstance.append(m.Runtime)
        runtimeaverage.append(sum(runtimeperinstance) / len(runtimeperinstance) )

        print('____________________________')
        print(runtimeperinstance)
        print()
        print('____________________________')
    print('____________________________')
    print(runtimeaverage)
    print('____________________________')

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')