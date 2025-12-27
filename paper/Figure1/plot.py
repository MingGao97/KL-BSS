import numpy as np
import matplotlib.pyplot as plt

ns = np.arange(1000,8001,1000)
metrics = ['rec', 'hd', 'fdr', 'tpr']
metric = metrics[0]
err_dist = 'mixed'
IC = 'BIC'

d = 10
s = 3
s0 = 4

fig, ax = plt.subplots(1,1, figsize=(3.5,4))
string1 = f'ER-4_{err_dist}_{d}_{s}'
string2 = f'Complete_{err_dist}_{d}_{s}'
rec11 = np.loadtxt(f'../Figure4/result/{metric}_klBSS_vanilla_' + string1 + '.csv')
rec12 = np.loadtxt(f'../Figure4/result/{metric}_BSS_' + string1 + '.csv')
rec21 = np.loadtxt(f'../Figure4/result/{metric}_klBSS_vanilla_' + string2 + '.csv')
rec22 = np.loadtxt(f'../Figure4/result/{metric}_BSS_' + string2 + '.csv')
toplot1 = np.mean(rec11, axis=0) - np.mean(rec12, axis=0)
toplot2 = np.mean(rec21, axis=0) - np.mean(rec22, axis=0)
bar_width = 300
ax.bar(ns, toplot1, bar_width, color = 'grey', alpha=0.6, label='Sparse graphs')
ax.bar(ns+bar_width, toplot2, bar_width, color = 'black', alpha=0.6, label='Dense graphs')
ax.set_xlabel('n (Sample size)', fontsize=12)
ax.set_ylabel('Increase in P(recovery) over BSS', fontsize=13)
ax.set_ylim(-0.05, 0.6)
ax.set_title('KL-BSS vs. BSS', fontsize=15)
ax.legend()
fig.savefig('Figure1right.pdf',bbox_inches='tight')