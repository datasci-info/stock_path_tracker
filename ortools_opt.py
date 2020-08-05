import config as cfg
from optimize import *

import numpy as np
from ortools.sat.python import cp_model

n_tau = 10
n_lambd = 5
cutoff_cost = cfg.CUTOFF_COST

PROFIT = np.array([cal_profit(var_lambda) for var_lambda in range(1, n_lambd+1)]).reshape(1, n_lambd)

df = get_df()
df_cost = get_cost(df, cutoff_cost)
df_cost = df_cost[df_cost.lambd.map(lambda x: x.is_integer())]
COST = df_cost.query(f'(d2m<{n_tau}) and (1 <= lambd <= {n_lambd})').pivot(index='d2m', columns='lambd', values='opt_price').to_numpy()

PROB = np.zeros((n_tau, n_lambd))
prob_delta_cell = get_prob_single_day(df)
for var_tau in range(n_tau):
    for var_lambda in range(n_lambd):
        PROB[var_tau, var_lambda] = get_realizations(var_lambda, var_tau).pipe(get_prob, prob_delta_cell)

model = cp_model.CpModel()

tau = np.array([model.NewBoolVar(f'tau_{idx}') for idx in range(n_tau)]).reshape((n_tau, 1))
lambd = np.array([model.NewBoolVar(f'lambd_{idx}') for idx in range(n_lambd)]).reshape((n_lambd, 1))
xx = np.array([model.NewBoolVar(f'xx_{i}_{j}') for i in range(n_tau) for j in range(n_lambd)]).reshape((n_tau, n_lambd))
b = model.NewBoolVar('b')

model.Add(tau.sum() ==1)
model.Add(lambd.sum() ==1)
model.Add(xx.sum() ==1)
for i in range(n_tau):# sum == 2
    for j in range(n_lambd):
        model.Add(tau[i, 0]+ lambd[j, 0] == 2).OnlyEnforceIf(b)
        model.Add(xx[i, j] == 1).OnlyEnforceIf(b)
        
        model.Add(tau[i, 0]+ lambd[j, 0] < 2).OnlyEnforceIf(b.Not())
        model.Add(xx[i, j] != 1).OnlyEnforceIf(b.Not())

model_prob = (PROB * 100).astype(int)
model_profit = (PROFIT * 100).astype(int)
model_cost = (COST * 100).astype(int)

# cannot multiply (xx, lambd)  
model.Maximize(( model_prob * xx).sum() * (model_profit@lambd)[0,0] - (model_cost * xx).sum())

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print('Maximum of objective function: %i' % solver.ObjectiveValue())