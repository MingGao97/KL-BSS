import sys
sys.path.append('../..')
import os
os.makedirs('result_iccv_hd', exist_ok=True)

import numpy as np
import random
from KLBSS import KLBSS
from utils import *


graph_types = ['ER','SF','Complete'] # 0-2
batches = [i for i in range(200)] # 0-200
samplesizes = np.arange(1000,8001,1000) # 0-7

d = 50
s = 10
s0 = 4
err_dist = 'mixed'
ns = np.arange(1000,8001,1000)
betamin = 0.1
ubs = 25
K = 5
betamins = 10**(np.arange(-2.4,0.8,0.2))

for graph_type in graph_types:
    betamax_SEM = 1 if graph_type == 'Complete' else 2
    betamin_SEM = betamax_SEM/10 if graph_type == 'Complete' else 0.1
    for b, i in enumerate(batches):
        ### fix data
        np.random.seed(1000+i)
        random.seed(1000+i)
        ###
        G = simulate_dag(d, s0*d, graph_type, s)
        X, Y, S = simulate_data(G, max(ns), s, betamin, betamax_SEM, betamin_SEM, err_dist=err_dist)
        for samplesize, n in enumerate(samplesizes):
            string = f'{graph_type}_{b}_{samplesize}'
            print(f'd: {d}; s: {s}; graph_type: {graph_type}; s0: {s0}; error_dist: {err_dist}; N: {i}; n: {n}')
            Xt, Yt = X[:n,:], Y[:n]

            res_klBSS = KLBSS(Xt,Yt,betams=betamins,ubs=ubs,ic='BIC',cv=True,K=K,mip=True)

            f_klBSS = open('result_iccv_hd/Shat_klBSS_' + string + '.txt', 'a')
            f_klBSS.write(str(res_klBSS) + '\n')
            f_klBSS.close()



