import sys
sys.path.append('../..')
import os
os.makedirs('result_hd_randsupp', exist_ok=True)
os.makedirs('data_S_randsupp', exist_ok=True)

import numpy as np
import random
from utils import *
from KLBSS import KLBSS

def sample_graph_with_example4(d, s, s0):
    # node 0 is k*, nodes 1,2,...,s-1 are T
    max_num_edge = int(d * (d - 1) / 2)
    # Erdos-Renyi
    edge_from, edge_to = np.nonzero(np.triu(np.ones(d), k = 1))
    edges = np.arange(s)
    edges = np.r_[edges, np.random.choice(np.arange(s, len(edge_from)), 
                                        min(s0, max_num_edge)-(s-1), replace = False)]
    edge_from = edge_from[edges]
    edge_to = edge_to[edges]
    B = np.zeros((d, d))
    B[edge_from, edge_to] = 1

    rand_sort = np.arange(d)
    np.random.shuffle(rand_sort)
    B = B[rand_sort, :]
    B = B[:, rand_sort]

    return B

def simulate_data_with_example4(G, n, s, betamin, betamax_SEM, betamin_SEM,
                                sigma=1, sigmamin=0.5, sigmamax=1):
    # generate X
    d = G.shape[0]
    sigmas = np.random.uniform(sigmamin, sigmamax, d)
    X = np.empty((n,d))
    caus_order = compute_caus_order(G)
    for node in caus_order:
        pa_of_node = find_pa(G, node)
        epsilon_node = simulate_error('mixed', n, sigmas[node])
        if len(pa_of_node) == 0:
            X[:,node] = epsilon_node
        else:
            beta = np.random.uniform(betamin_SEM, betamax_SEM, len(pa_of_node))
            beta *= (2 * np.random.binomial(1, 0.5, len(pa_of_node)) - 1)
            fpa = X[:,pa_of_node] @ beta
            X[:,node] = fpa + epsilon_node
    # generate Y
    S = np.sort(np.random.choice(d, s, replace=False))
    beta = betamin * (2 * np.random.binomial(1, 0.5, s) - 1)
    Y = X[:,S] @ beta + simulate_error('mixed', n, sigma)
    return X, Y, S

ss = [10, 15, 20] # 0-2
batches = [range(i*5,(i+1)*5) for i in range(40)] # 0-39
d = 50
s0 = 4*50
ubs = 25
ns = np.arange(1000,8001,1000)
betamin = 0.2
betamax_SEM = 2
betamin_SEM = 0.2

for s in ss:
    for b, batch in enumerate(batches):
        string = f'{s}_{b}'
        for i in batch:
            ### fix data
            np.random.seed(1000+i)
            random.seed(1000+i)
            ###
            G = sample_graph_with_example4(d,s,s0)
            X, Y, S = simulate_data_with_example4(G,max(ns),s,betamin,betamax_SEM,betamin_SEM)
            f = open('data_S_randsupp/S_' + string + '.txt', 'a')
            f.write(str(S) + '\n')
            f.close()
            for j, n in enumerate(ns):
                print(f's: {s}; N: {i}; n: {n}')
                Xt, Yt = X[:n,:], Y[:n]
                cc = np.log(n) / n

                res_BSS = KLBSS(Xt,Yt,ubs=ubs,ic='BIC',bss=True,mip=True)
                res_klBSS = KLBSS(Xt,Yt,ubs=ubs,ic='BIC',betam=betamin,mip=True)

                f_BSS = open(f'result_hd_randsupp/Shat_BSS_' + string + '.txt', 'a')
                f_BSS.write(str(res_BSS) + '\n')
                f_BSS.close()

                f_klBSS = open(f'result_hd_randsupp/Shat_klBSS_' + string + '.txt', 'a')
                f_klBSS.write(str(res_klBSS) + '\n')
                f_klBSS.close()