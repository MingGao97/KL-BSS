import sys
sys.path.append('../..')
import os
os.makedirs('data_S_cv_hd', exist_ok=True)

import numpy as np
import random
from utils import *

d = 50
s = 10
s0 = 4
err_dist = 'mixed'
graph_types = ['ER','SF','Complete'] # 0-2
ns = np.arange(1000,8001,1000)
betamin = 0.1

for graph_type in graph_types:
    betamax_SEM = 1 if graph_type == 'Complete' else 2
    betamin_SEM = betamax_SEM/10 if graph_type == 'Complete' else 0.1
    for i in range(200):
        ### fix data
        np.random.seed(1000+i)
        random.seed(1000+i)
        ###
        G = simulate_dag(d, s0*d, graph_type, s)
        X, Y, S = simulate_data(G, max(ns), s, betamin, betamax_SEM, betamin_SEM, err_dist=err_dist)
        f = open(f'data_S_cv_hd/S_{graph_type}_{i}.txt', 'a')
        f.write(str(S) + '\n')
        f.close()