
import numpy as np
import gurobipy as gp
from gurobipy import *
from gurobipy import GRB

try:
    rangeofAgents = 15
    startagent = 2
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
            ### Need to maintain it separately and assign separately so that gurobi works
            ## Maxitemforagent [i][j] stores max of { agent[j] * valuation[i] }
            maxitemforeachagent = []

            ## Multiplication terms contains multiplication list of all agent[j]*valuation[i]
            ## length is n^2
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