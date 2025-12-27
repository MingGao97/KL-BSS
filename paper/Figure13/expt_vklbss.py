import sys
sys.path.append('../..')

import numpy as np
import igraph as ig
import networkx as nx
from KLBSS import KLBSS

########## below are utils

def compute_caus_order(G):
    d = G.shape[0]
    remain = list(range(d))
    caus_order = np.empty(d, dtype = int)
    for i in range(d-1):
        root = min(np.where(G.sum(axis=0) == 0)[0])
        caus_order[i] = remain[root]
        del remain[root]
        G = np.delete(G, root, axis = 0)
        G = np.delete(G, root, axis = 1)
    caus_order[d-1] = remain[0]
    return caus_order

def find_pa(G, node):
    return np.where(G[:,node] == 1)[0]


def simulate_dag(d, s0=2, graph_type='ER', q=3, permute=True):
    max_num_edge = int(d * (d - 1) / 2)
    if graph_type == 'ER':
        # Erdos-Renyi
        edge_from, edge_to = np.nonzero(np.triu(np.ones(d), k = 1))
        edges = np.random.choice(len(edge_from), min(s0, max_num_edge), replace = False)
        edge_from = edge_from[edges]
        edge_to = edge_to[edges]
        B = np.zeros((d, d))
        B[edge_from, edge_to] = 1
        if permute:
            rand_sort = np.arange(d)
            np.random.shuffle(rand_sort)
            B = B[rand_sort, :]
            B = B[:, rand_sort]
        
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(min(s0, max_num_edge) / d)), directed=True)
        B = np.array(G.get_adjacency().data)
        rand_sort = np.arange(d)
        np.random.shuffle(rand_sort)
        B = B[rand_sort, :]
        B = B[:, rand_sort]

    elif graph_type == 'Tree':
        # Tree graph
        B = np.tril(nx.to_numpy_matrix(nx.generators.trees.random_tree(d)))

    elif graph_type == 'MC':
        # Markov chain
        B = np.eye(d, k = 1)
        if permute:
            rand_sort = np.arange(d)
            np.random.shuffle(rand_sort)
            B = B[rand_sort, :]
            B = B[:, rand_sort]
    
    elif graph_type == 'Bipartite':
        V = np.arange(d)
        idx = np.random.choice(np.arange(1,d),1)[0]
        V1 = V[:idx]
        V2 = V[idx:]
        qb = min(q,len(V1))
        B = np.zeros((d,d))
        for j in V2:
            npa = np.random.choice(np.arange(1,qb+1),1)[0]
            jpa = np.random.choice(V1,npa)
            B[jpa,j] = 1

        rand_sort = np.arange(d)
        np.random.shuffle(rand_sort)
        B = B[rand_sort, :]
        B = B[:, rand_sort]

    return B


def simulate_data(G, n, q, betamax, betamin, sigma=1, sigmamin=0.5, sigmamax=2):
    # generate X
    d = G.shape[0]
    sigmas = np.random.uniform(sigmamin, sigmamax, d)
    epsilon = np.random.randn(n*d).reshape(n,d) * sigmas
    X = np.empty((n,d))
    caus_order = compute_caus_order(G)
    for node in caus_order:
        pa_of_node = find_pa(G, node)
        if len(pa_of_node) == 0:
            X[:,node] = epsilon[:,node]
        else:
            beta = np.random.uniform(betamin, betamax, len(pa_of_node))
            # beta *= (2 * np.random.binomial(1, 0.5, len(pa_of_node)) - 1)
            fpa = X[:,pa_of_node] @ beta
            X[:,node] = fpa + epsilon[:,node]
    # generate Y
    Y = X[:,:q].sum(axis=1) * betamin + np.random.randn(n) * sigma
    return X, Y

########## above are utils

d=7
q=3
s0=2
graph_type='SF'

N = 200
ns = np.arange(100,3500,400)
betamin = 0.1
betamax = 15

Sstar = [ii for ii in range(q)]
res = np.zeros((2,N,len(ns)))

for i in range(N):
    G = simulate_dag(d=d, s0=d*s0, graph_type='SF')
    X, Y = simulate_data(G, max(ns), q, betamax, betamin)
    for j, n in enumerate(ns):
        print(f'N: {i}; n: {n}')
        Xt, Yt = X[:n,:], Y[:n]
        res[0,i,j] = KLBSS(Xt,Yt,klbss_type='vanilla',betam=betamin,s=q) == Sstar
        res[1,i,j] = KLBSS(Xt,Yt,klbss_type='simple',betam=betamin,s=q) == Sstar

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(ns, res[0,:,:].mean(axis=0), label='Vanilla KLBSS', color='tab:blue', linewidth=4, alpha=0.6)
ax.plot(ns, res[1,:,:].mean(axis=0), label='KLBSS', color='tab:green', linewidth=4, alpha=0.6)
ax.legend()
ax.set_ylabel('P(Support recovery)', fontsize=15)
ax.set_xlabel('n (sample size)', fontsize=15)
ax.set_title('Effect of conditioning', fontsize=18)
fig.savefig(f'Figure13.pdf',bbox_inches='tight')


