import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


N = 200
ns = np.arange(200,801,100)
methods = ['klBSS','BSS']
methodslabel = ['KLBSS','BSS']
cols = plt.cm.viridis(np.linspace(0,1,5))[::-1]
metrics = ['rec', 'hd', 'fdr', 'tpr']
metricsdict = {'rec' : 'P(Support recovery)',
                'hd' : 'Hamming distance',
                'fdr' : 'FDR',
                'tpr' : 'TPR'}
ds = [50,60,70,80,90]
linestyles = ['-', ':']

metric = metrics[0]

box = np.zeros((len(ds),len(ns)))
fig, ax = plt.subplots(1,1, figsize=(8,5))
for j, d in enumerate(ds):
    ress = np.zeros((2,len(ns)))
    for k, metho in enumerate(methods):
        res = np.loadtxt(f'result/{metric}_{metho}_{d}.csv')
        ax.plot(ns, res.mean(axis=0),
                color=cols[j], linestyle=linestyles[k],
                linewidth=3, alpha=0.6)
        ress[k] = res.mean(axis=0)
    box[j] = ress[0]-ress[1]

bar_width = 15
for j in range(len(ds)):
    ax.bar(ns + j*bar_width, box[j], bar_width, 
              alpha=0.6, color=cols[j])

ax.set_ylabel(metricsdict[metric])
ax.set_xlabel('n (sample size)')
ax.set_title('Recovery performance')
ax.set_ylim(0,1)
lines = [Line2D([0], [0], color='black', linestyle=linestyle, linewidth=4, alpha=0.8) for linestyle in linestyles]
lines += [Line2D([0], [0], color=col, linewidth=4, alpha=0.8) for col in cols]
labels = methodslabel + [f'd={d}' for d in ds]
ax.legend(lines, labels)
fig.savefig(f'Figure8left.pdf',bbox_inches='tight')