import sys
sys.path.append('../..')
import os
os.makedirs('result_iccv_ld', exist_ok=True)
os.makedirs('data_S_cv_ld', exist_ok=True)

import numpy as np
import random
from KLBSS import KLBSS
from utils import *

graph_types = ['ER','SF','Complete'] # 0-2
batches = [range(i*2,(i+1)*2) for i in range(100)] # 0-99

d = 10
s = 3
s0 = 4
err_dist = 'mixed'
betamin = 0.1
betamax = 5
ns = np.arange(1000,8001,1000)
ubs = 4
K = 5
betamins = 10**(np.arange(-2.4,0.8,0.2))

for graph_type in graph_types:
    for b, batch in enumerate(batches):
        string = f'{graph_type}_{b}'
        for i in batch:
            ### fix data
            np.random.seed(1000+i)
            random.seed(1000+i)
            ###
            G = simulate_dag(d, s0*d, graph_type, s)
            X, Y, S = simulate_data(G, max(ns), s, betamin, betamax, betamin, err_dist=err_dist)
            f = open('data_S_cv_ld/S_' + string + '.txt', 'a')
            f.write(str(S) + '\n')
            f.close()
            for j, n in enumerate(ns):
                print(f'd: {d}; s: {s}; s0: {s0}; graph_type: {graph_type}; error_dist: {err_dist}; N: {i}; n: {n}')
                Xt, Yt = X[:n,:], Y[:n]
                
                res_klBSS_vanilla = KLBSS(Xt,Yt,klbss_type='vanilla',ubs=ubs,ic='BIC',cv=True,betams=betamins,K=K)
                res_klBSS_simple = KLBSS(Xt,Yt,klbss_type='simple',ubs=ubs,ic='BIC',cv=True,betams=betamins,K=K)

                f_klBSS_vanilla = open('result_iccv_ld/Shat_klBSS_vanilla_' + string + '.txt', 'a')
                f_klBSS_vanilla.write(str(res_klBSS_vanilla) + '\n')
                f_klBSS_vanilla.close()

                f_klBSS_simple = open('result_iccv_ld/Shat_klBSS_simple_' + string + '.txt', 'a')
                f_klBSS_simple.write(str(res_klBSS_simple) + '\n')
                f_klBSS_simple.close()



