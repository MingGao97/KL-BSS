import sys
sys.path.append('../..')
import os
os.makedirs('res', exist_ok=True)

import numpy as np
from utils import *
from KLBSS import KLBSS

# equal variance function
def EqVarDAG_TD_internal(X):
    n, p = X.shape
    done = []
    S = np.cov(X.T)
    Sinv = np.linalg.inv(S)    
    for i in range(p):
        varmap = np.delete(np.array(range(p)), done)
        v = np.diag(np.linalg.inv(np.delete(np.delete(Sinv,done,axis=0),done,axis=1))).argmin()
        done.append(varmap[v])
    return done

# test if estimated ordering is consistent with DAG structure
def test_order(ordering, G):
    start, end = np.where(G == 1)
    for (ei, ej) in zip(start, end):
        if ordering.index(ei) > ordering.index(ej):
            return False
    return True

# Taken from NOTEARS code
def count_accuracy(B_true, B_est):
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


graph_types = ['ER','SF'] # 0-1
orders = [True, False] # 0-1

s0 = 2
d = 10
err_dist = 'Gaussian'
betamin = 0.1
betamax = 5
betas =  betamin * np.ones(d)
N = 100
ns = np.arange(1000,8001,1000)
res = np.zeros((4,N,len(ns)))

for graph_type in graph_types:
    for order in orders:
        string = f'{graph_type}_{order}'
        print(string)
        for i in range(100):
            G = simulate_dag(d, s0*d, graph_type, None)
            X = simulate_data(G, max(ns), betamin, betamax, 2, 2.01, err_dist=err_dist)
            if order:
                ordering = compute_caus_order(G)
            ubs = min(d, int(max(G.sum(axis=0)))+1)
            for j, n in enumerate(ns):
                print(i,n)
                Xt = X[:n]
                if not order:
                    ordering = EqVarDAG_TD_internal(Xt)
                    res[3,i,j] = test_order(ordering, G)
                    ordering = np.array(ordering)
                Ghat_vklbss = np.zeros((d,d))
                Ghat_sklbss = np.zeros((d,d))
                Ghat_bss = np.zeros((d,d))
                for ll in range(1,d):
                    Shat_vklbss = KLBSS(Xt[:,ordering[:ll]], Xt[:,ordering[ll]], klbss_type='vanilla', ubs=ubs, ic='BIC', betam=betamin)
                    Shat_sklbss = KLBSS(Xt[:,ordering[:ll]], Xt[:,ordering[ll]], klbss_type='simple', ubs=ubs, ic='BIC', betam=betamin)
                    Shat_bss = KLBSS(Xt[:,ordering[:ll]], Xt[:,ordering[ll]], bss=True, ubs=ubs, ic='BIC')
                    Ghat_vklbss[ordering[:ll][Shat_vklbss], ordering[ll]] = 1
                    Ghat_sklbss[ordering[:ll][Shat_sklbss], ordering[ll]] = 1
                    Ghat_bss[ordering[:ll][Shat_bss], ordering[ll]] = 1
                res[0,i,j] = count_accuracy(G,Ghat_vklbss)['shd']
                res[1,i,j] = count_accuracy(G,Ghat_sklbss)['shd']
                res[2,i,j] = count_accuracy(G,Ghat_bss)['shd']

                np.savetxt('res/' + string + '_klBSS_vanilla.csv', res[0])
                np.savetxt('res/' + string + '_klBSS.csv', res[1])
                np.savetxt('res/' + string + '_BSS.csv', res[2])
                if not order:
                    np.savetxt('res/' + string + '_eqvar.csv', res[3])




# plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

graph_types = ['ER','SF'] # 0-1
orders = [True, False] # 0-1
N = 100
ns = np.arange(1000,8001,1000)
methods = ['BSS','klBSS_vanilla','klBSS']
methodslabel = ['BSS','Vanilla KLBSS','KLBSS']
cols = ['tab:red','tab:blue','tab:green']
fig, ax = plt.subplots(1,4,figsize=(12,3))
for i, graph_type in enumerate(graph_types):
    for j, order in enumerate(orders):
        string = f'{graph_type}_{order}'
        for k, metho in enumerate(methods):
            res = np.loadtxt('res/'+string+f'_{metho}.csv')
            ax[i*2+j].plot(ns, rand_jitter(res.mean(axis=0)), color=cols[k], linewidth=4, alpha=0.5)
        ax[i*2+j].set_ylim(0,3)
        temp = 'oracle' if order else 'EqVar'
        ax[i*2+j].set_title(f'{graph_type} w/ {temp} ordering')
        if i!=0 or j!=0:
            ax[i*2+j].get_yaxis().set_ticks([])
ax[0].set_ylabel('SHD', fontsize=12)
ax[1].set_xlabel('                                               n (Sample size)')

lines = [Line2D([0], [0], color=col, linewidth=4, alpha=0.8) for col in cols]
labels = methodslabel[:]
ax[1].legend(lines, labels, loc='upper center', bbox_to_anchor=(1, -0.18), ncol=5, fontsize=10)

# adjust main title
plt.subplots_adjust(top=0.83)
plt.suptitle('Structure learning performance', fontsize=15)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
fig.savefig('Figure6.pdf',bbox_inches='tight')

