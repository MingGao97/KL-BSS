import sys
sys.path.append('../..')

import numpy as np
from utils import *
import itertools

def simulate_cov(G, s, betamax_SEM, betamin_SEM, sigmamin=0.5, sigmamax=1):
    d = G.shape[0]
    sigmas = np.random.uniform(sigmamin, sigmamax, d)
    B = np.zeros((d,d))
    caus_order = compute_caus_order(G)
    for node in caus_order:
        pa_of_node = find_pa(G, node)
        if len(pa_of_node) != 0:
            beta = np.random.uniform(betamin_SEM, betamax_SEM, len(pa_of_node))
            beta *= (2 * np.random.binomial(1, 0.5, len(pa_of_node)) - 1)
            B[pa_of_node, node] = beta
    S = np.sort(np.random.choice(d, s, replace=False))
    return B, sigmas, S.tolist()


def condeig(Sigma, S, T):
    SmT = np.setdiff1d(S,T)
    SigmaS = Sigma[SmT,:][:,SmT]
    SigmaT = Sigma[T,:][:,T]
    SigmaST = Sigma[SmT,:][:,T]
    cond = SigmaS - SigmaST @ np.linalg.inv(SigmaT) @ SigmaST.T
    return np.linalg.eigvals(cond).min()


d = 12
ss = [2,3,4,5]
graph_types = ['ER','SF']
s0s = [2,3,4,5]

N = 5000
res = np.zeros((len(graph_types), len(s0s), len(ss), N))
for j, graph_type in enumerate(graph_types):
    for k, s0 in enumerate(s0s):
        for l, s in enumerate(ss):
            print(graph_type, s0, s)
            for i in range(N):
                G = simulate_dag(d, s0*d, graph_type, s)
                B, D, Star = simulate_cov(G, s, 5, 0.1)
                base = np.linalg.inv(np.eye(G.shape[0]) - B.T)
                Sigma = base @ np.diag(D**2) @ base.T

                SgivenT = []
                TgivenS = []
                for comb in itertools.combinations(np.arange(d), s):
                    T = list(comb)
                    if T != Star:
                        SgivenT.append(condeig(Sigma, Star, T))
                        TgivenS.append(condeig(Sigma, T, Star))
                 
                res[j,k,l,i] = min(SgivenT) < min(TgivenS)


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


alphas = [0.1,0.4,0.7,0.95]
res2 = res.mean(axis=-1)
fig, ax = plt.subplots(1,2, figsize=(10,4))
for j in range(len(graph_types)):
    for k in range(len(s0s)):
        ax[j].plot(ss, res2[j,k], color='tab:blue', alpha=alphas[k], linewidth=2)
    ax[j].set_title(graph_types[j] + ' graph', fontsize=16)
    ax[j].set_xlabel('s', fontsize=14)
    ax[j].set_ylim(0.18,0.65)
ax[-1].get_yaxis().set_ticks([])
ax[0].set_ylabel('Percentage of satisfying ' + r'$\Omega_\Delta$', fontsize=14)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

lines = [Line2D([0], [0], color='tab:blue', linewidth=4, alpha=alp) for alp in alphas]
labels = [r'$s_0$=' + str(s0) for s0 in s0s]
plt.legend(lines, labels, fontsize=12)
fig.savefig('Figure7.pdf',bbox_inches='tight')


