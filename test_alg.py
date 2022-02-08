import copy
from gurobipy import Model, GRB, quicksum, Env
import numpy as np
from matplotlib import pyplot as plt

comp_mat = np.array([[1, 0, 0, 0, 1, 0, 1, 0],
                     [0, 1, 1, 0, 1, 0, 1, 0],
                     [0, 0, 1, 1, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 1]])

comp_mat = comp_mat + comp_mat.T - np.eye(8)

reductions = np.random.rand(8)*10

scores = [copy.copy(reductions)/max(reductions)]

runs = 200
for i in range(runs):
    new_scores = comp_mat @ scores[-1] / 8
    new_scores /= max(new_scores)
    scores.append(new_scores)

for i in range(8):
    plt.plot(np.array(scores)[:, i], label=str(reductions[i]))
plt.legend()
plt.show()

m = Model('CVRP')
# self.m.setParam('Method', 2) ###################testare == 2 !!!!!!!!!!!!111c
m.modelSense = GRB.MAXIMIZE
c = m.addMVar(8, vtype=GRB.BINARY)
A = (np.logical_not(comp_mat).astype(int))
m.addConstr(A@c <= (np.ones(8)-c)*8)
m.setObjective(c@reductions)
m.setParam('OutputFlag', 0)
m.optimize()
m.status
print(m.getObjective().getValue())
print(c.x)


comp_mat @ scores[-1]
scores
