import sys
sys.path.append('../..')
import os
os.makedirs('data_S', exist_ok=True)
os.makedirs('result', exist_ok=True)

import numpy as np
import random
from KLBSS import KLBSS
from utils import *

ds = [50]
ss = [10, 15, 20] # 0-2
s0s = [1,2,4]  # 0-2
graph_types = ['ER','SF','Bipartite','Complete'] # 0-3
err_dists = ['Gaussian', 't', 'Laplace', 'unif', 'mixed'] # 0-4
batches = [range(i*25,(i+1)*25) for i in range(8)] # 0-7

ns = np.arange(1000,8001,1000)
betamin = 0.1


def run_expt(d,s,s0,graph_type,err_dist,batch,b):
    betamax_SEM = 1 if graph_type == 'Complete' else 2
    betamin_SEM = betamax_SEM/10 if graph_type == 'Complete' else 0.1
    if graph_type in ['ER','SF']:
        string = f'{graph_type}-{s0}_{err_dist}_{d}_{s}_{b}'
    else:
        string = f'{graph_type}_{err_dist}_{d}_{s}_{b}'
    for i in batch:
        ### fix data
        np.random.seed(1000+i)
        random.seed(1000+i)
        ###
        G = simulate_dag(d, s0*d, graph_type, s)
        X, Y, S = simulate_data(G, max(ns), s, betamin, betamax_SEM, betamin_SEM, err_dist=err_dist)
        f = open('data_S/S_' + string + '.txt', 'a')
        f.write(str(S) + '\n')
        f.close()
        for j, n in enumerate(ns):
            if graph_type in ['ER','SF']:
                print(f'd: {d}; s: {s}; graph_type: {graph_type}; s0: {s0}; error_dist: {err_dist}; N: {i}; n: {n}')
            else:
                print(f'd: {d}; s: {s}; graph_type: {graph_type}; error_dist: {err_dist}; N: {i}; n: {n}')
            Xt, Yt = X[:n,:], Y[:n]

            res_BSS = KLBSS(Xt,Yt,s=s,bss=True,mip=True)
            res_klBSS = KLBSS(Xt,Yt,s=s,betam=betamin,mip=True)

            f_BSS = open('result/Shat_BSS_' + string + '.txt', 'a')
            f_BSS.write(str(res_BSS) + '\n')
            f_BSS.close()

            f_klBSS = open('result/Shat_klBSS_' + string + '.txt', 'a')
            f_klBSS.write(str(res_klBSS) + '\n')
            f_klBSS.close()



for d in ds:
    for s in ss:
        for err_dist in err_dists:
            for graph_type in graph_types:
                if graph_type in ['ER','SF']:
                    for s0 in s0s:
                        for b, batch in enumerate(batches):
                            run_expt(d,s,s0,graph_type,err_dist,batch,b)
                else:
                    for b, batch in enumerate(batches):
                        run_expt(d,s,s0,graph_type,err_dist,batch,b)