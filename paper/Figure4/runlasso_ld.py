import sys
sys.path.append('../..')
import os
os.makedirs('result_lasso', exist_ok=True)

import numpy as np
import random
from utils import *
from sklearn.linear_model import lasso_path

### 
import warnings
warnings.filterwarnings('ignore')
###

def metric(S,Shat):
    S = sorted(S)
    Shat = sorted(Shat)
    rec = S == Shat
    TP = len(np.intersect1d(S,Shat))
    FP = len(Shat) - TP
    FN = len(S) - TP
    fdr = FP / (FP + TP) if len(Shat) !=0 else 0
    tpr = TP / (TP + FN) if len(S) !=0 else 0
    hd = FP + FN
    return rec, hd, fdr, tpr


def lasso_check(X,Y,Sstar):
    _, coef, _ = lasso_path(X,Y,n_alphas=500)
    res = np.zeros((500,4))
    for i in range(coef.shape[1]):
        Shat = np.where(coef[:,i]!=0)[0].tolist()
        rec, hd, fdr, tpr = metric(Sstar, Shat)
        res[i,0] = rec
        res[i,1] = hd
        res[i,2] = fdr
        res[i,3] = tpr
    idx = np.argmin(res[:,1])
    return res[idx,:]

ds = [8,9,10]
ss = [2,3,4] # 0-2
s0s = [1,2,4]  # 0-2
graph_types = ['ER','SF','Bipartite','Complete'] # 0-3
err_dists = ['Gaussian', 't', 'Laplace', 'unif', 'mixed'] # 0-4


betamin = 0.1
betamax = 5
ns = np.arange(1000,8001,1000)
N = 200


def run_expt(d,s,s0,graph_type,err_dist):
    if graph_type in ['ER','SF']:
        string = f'{graph_type}-{s0}_{err_dist}_{d}_{s}'
    else:
        string = f'{graph_type}_{err_dist}_{d}_{s}'
    res = np.zeros((4,N,len(ns)))
    for i in range(N):
        ### fix data
        np.random.seed(1000+i)
        random.seed(1000+i)
        ###
        G = simulate_dag(d, s0*d, graph_type, s)
        X, Y, S = simulate_data(G, max(ns), s, betamin, betamax, betamin, err_dist=err_dist)

        for j, n in enumerate(ns):
            if graph_type in ['ER','SF']:
                print(f'd: {d}; s: {s}; graph_type: {graph_type}; s0: {s0}; error_dist: {err_dist}; N: {i}; n: {n}')
            else:
                print(f'd: {d}; s: {s}; graph_type: {graph_type}; error_dist: {err_dist}; N: {i}; n: {n}')
            Xt, Yt = X[:n,:], Y[:n]
            res[:,i,j] = lasso_check(Xt,Yt,S)

    np.savetxt('result_lasso/rec_lasso_' + string + '.csv', res[0,:,:])
    np.savetxt('result_lasso/hd_lasso_' + string + '.csv', res[1,:,:])
    np.savetxt('result_lasso/fdr_lasso_' + string + '.csv', res[2,:,:])
    np.savetxt('result_lasso/tpr_lasso_' + string + '.csv', res[3,:,:])


for d in ds:
    for s in ss:
        for s0 in s0s:
            for graph_type in graph_types[:2]:
                for err_dist in err_dists:
                    run_expt(d,s,s0,graph_type,err_dist)
    s0 = 1
    for s in ss:
        for graph_type in graph_types[2:]:
            for err_dist in err_dists:
                run_expt(d,s,s0,graph_type,err_dist)