import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ns = np.arange(1000,8001,1000)
ss = [10,15,20]

fig,ax = plt.subplots(1,3,figsize=(12,5), sharey='row')

for j, s in enumerate(ss):
    rec_BSS = np.loadtxt(f'result_hd_randsupp/rec_BSS_{s}.csv')
    rec_klBSS = np.loadtxt(f'result_hd_randsupp/rec_klBSS_{s}.csv')
    ax[j].plot(ns, rec_BSS.mean(axis=0), color='tab:red')
    ax[j].plot(ns, rec_klBSS.mean(axis=0), color='tab:blue')
    ax[j].set_title(f's={s}', fontsize=14)
    rec_klbss = np.loadtxt(f'result_cv_hd_randsupp/rec_klbss_{s}.csv')
    ax[j].plot(ns, rec_klbss.mean(axis=0), color='tab:blue', linestyle='--')
ax[0].set_ylim(-0.05,1.02)
ax[0].set_ylabel('P(recovery)', fontsize=16)
ax[1].set_xlabel('n (sample size)', fontsize=12)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# legends
lines = [Line2D([0], [0], color='tab:red', linewidth=4, alpha=0.8),
        Line2D([0], [0], color='tab:blue', linewidth=4, alpha=0.8),
        Line2D([0], [0], color='tab:blue', linewidth=4, alpha=0.8, linestyle='--')]
labels = ['BSS', 'Vanilla KLBSS','Vanilla KLBSS (CV)']
ax[1].legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=12, handlelength=5)

fig.savefig('Figure12.pdf', bbox_inches='tight')