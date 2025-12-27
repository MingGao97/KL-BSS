import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

def colorFader(c1,c2,mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


N = 200
ns = np.arange(1000,8001,1000)
betamins = 10**(np.arange(-2.4,0.8,0.2))
methods = ['BSS','klBSS_vanilla','klBSS_simple']
methodslabel = ['BSS','Vanilla KLBSS','KLBSS']
cols = ['tab:red','tab:blue','tab:green']
metrics = ['rec', 'hd', 'fdr', 'tpr']
metricsdict = {'rec' : 'P(Support recovery)',
                'hd' : 'Hamming distance',
                'fdr' : 'FDR',
                'tpr' : 'TPR'}


####
loc = np.where(betamins <=0.1)[0][-1] + 1
cols1 = [colorFader('tab:red','tab:green',(i+1)/loc) for i in range(loc)]
cols2 = [colorFader('tab:green','teal',(i+1)/(len(betamins)-loc)) for i in range(len(betamins)-loc)]
coll = cols1 + cols2
####


metric = metrics[0]


fig, ax = plt.subplots(1,1, figsize=(8,5))
for i, betat in enumerate(betamins):
    res = np.loadtxt(f'result/{metric}_klBSS_simple_{i}.csv')
    ax.plot(ns, res.mean(axis=0),
            color=coll[i], linewidth=2, alpha=0.5)

metho = methods[0]
res = np.loadtxt(f'result/{metric}_{metho}.csv')
ax.plot(ns, res.mean(axis=0), 
                color=cols[methods.index(metho)], linewidth=4, alpha=0.8)

for metho in methods[1:]:
    res = np.loadtxt(f'result/{metric}_{metho}.csv')
    res_cv = np.loadtxt(f'result/{metric}_{metho}_cv.csv')
    ax.plot(ns, res.mean(axis=0),
            color=cols[methods.index(metho)], linewidth=4, alpha=0.8)
    ax.plot(ns, res_cv.mean(axis=0), linestyle='--',
            color=cols[methods.index(metho)], linewidth=4, alpha=0.8)


## colorbar
cmap = mpl.colors.ListedColormap(coll)
bounds = np.arange(-2.4,1,0.2).tolist()
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax, location='right', orientation='vertical', 
             label=r'$\log_{10} \widetilde{\beta}_{\min}$ in KLBSS',
             anchor=(-0.3,0))
##


ax.legend()
ax.set_ylabel(metricsdict[metric])
ax.set_xlabel('n (sample size)')
ax.set_title('Performance of CV')
lines = [Line2D([0], [0], color=col, linewidth=4, alpha=0.8) for col in cols] + \
        [Line2D([0], [0], linestyle='--', color='black', linewidth=4, alpha=0.5)]
labels = methodslabel + ['CV']
ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
          ncol=5, fontsize=8, handlelength=4)

fig.savefig(f'Figure5left.pdf',bbox_inches='tight')
