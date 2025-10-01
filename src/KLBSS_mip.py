import numpy as np
import gurobipy as gp
from gurobipy import GRB

def klBSS(Y, X, s, bss=False, betamu=None, betaml=None, M=None,
          time_limit=600, mip_gap=1e-5, 
          hard_limit=900, hard_gap=1e-9, 
          nthread=8, verbose=0):
    def early_stopping(model, where):
        if where == GRB.Callback.MIP:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            if objbst == 0:
                gap = float('inf')
            else:
                gap = abs((objbst - objbnd) / objbst)
            if runtime > time_limit and gap < mip_gap:
                model.terminate()
    n, d = X.shape
    model = gp.Model('good')
    model.setParam('OutputFlag', verbose)
    model.setParam('Threads', nthread)
    model.setParam('TimeLimit', hard_limit)
    model.setParam('MIPGap', hard_gap)
    b = model.addMVar((d,), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="beta")
    nz = model.addMVar((d,), vtype=GRB.BINARY, name='nz')
    model.addConstr(nz.sum() == d-s)
    if not bss:
        w = model.addMVar((d,), vtype=GRB.BINARY, name='w')
        for k in range(d):
            model.addConstr(b[k] + M * w[k] >= betamu[k] * (1-nz[k]))
            model.addConstr(-b[k] + M * (1-w[k]) >= betaml[k] * (1-nz[k]))
    for j in range(d):
        model.addSOS(GRB.SOS_TYPE1, [b[j], nz[j]], [1,2])
    XTX = X.T @ X / n
    XTY = 2 * Y.T @ X / n
    obj = b.T @ XTX @ b - XTY @ b
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize(early_stopping)
    
    # Check if model has a valid solution
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        Shat = np.array([1-nz[i].X.item() for i in range(d)])
        Shat = np.where(Shat == 1)[0].tolist()
    else:
        # If no valid solution, return empty support
        print(f"No valid solution found.")
        Shat = []
    
    return Shat, model.Runtime, model