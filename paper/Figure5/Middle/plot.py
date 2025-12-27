import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


N = 200
ns = np.arange(1000,8001,1000)
methods = ['BSS','klBSS_vanilla','klBSS_simple']
methodslabel = ['BSS','Vanilla KLBSS','KLBSS']
cols = ['tab:red','tab:blue','tab:green']
metrics = ['rec', 'hd', 'fdr', 'tpr']
metricsdict = {'rec' : 'P(Support recovery)',
                'hd' : 'Hamming distance',
                'fdr' : 'FDR',
                'tpr' : 'TPR'}
ICs = ['BIC', 'EBIC', 'Delta']
ubss = [3,4,5,6,7]
alphas = [0.6*(len(ubss)-i+0.2)/len(ubss)+0.1 for i in range(len(ubss))]


IC = ICs[0]
metric = metrics[0]


fig, ax = plt.subplots(1,1, figsize=(8,5))
for metho in methods:
    res = np.zeros((len(ubss),N,len(ns)))
    if metho != 'BSS':
        res_iccv = np.zeros((len(ubss),N,len(ns)))
    for i, ubs in enumerate(ubss):
        res[i,:,:] = np.loadtxt(f'result/{metric}_{metho}_{ubs}.csv')
        if metho != 'BSS':
            res_iccv[i,:,:] = np.loadtxt(f'result/{metric}_{metho}_cv_{ubs}.csv')
    for i, ubs in enumerate(ubss):
        lw = 4 if i == 0 else 2
        ax.plot(ns, res[i,:,:].mean(axis=0), label=r'$\bar{s}$='+str(ubs),
                color=cols[methods.index(metho)], linewidth=lw, alpha=alphas[i])
        if metho != 'BSS':
            ax.plot(ns, res_iccv[i,:,:].mean(axis=0), label=r'$\bar{s}$='+str(ubs),
                    color=cols[methods.index(metho)], linewidth=lw, alpha=alphas[i],
                    linestyle='--')
ax.legend()
ax.set_ylabel(metricsdict[metric])
ax.set_xlabel('n (sample size)')
ax.set_title(r'Effect of $\bar{s}$' + f' -- {IC}')
lines = [Line2D([0], [0], color=col, linewidth=4, alpha=0.8) for col in cols] +\
      [Line2D([0], [0], color='black', linewidth=4, alpha=alphas[0])] +\
      [Line2D([0], [0], color='black', linewidth=2, alpha=alp) for alp in alphas[1:]]
labels = methodslabel + [r'$\bar{s}$=' + str(ubs) for ubs in ubss]
ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=8)
fig.savefig(f'Figure5middle.pdf',bbox_inches='tight')


