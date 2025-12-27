import numpy as np
import matplotlib.pyplot as plt

metricsdict = {'rec' : 'P(Support recovery)',
                'hd' : 'Hamming distance',
                'fdr' : 'FDR',
                'tpr' : 'TPR'}

metric = 'rec'
ds = np.arange(20,110,10).tolist() + [200,500,1000]
res1 = np.loadtxt(f'res/{metric}_low.csv')
res2 = np.loadtxt(f'res/{metric}_high.csv')
res_time1 = np.loadtxt(f'res/res_time_low.csv')
res_time2 = np.loadtxt(f'res/res_time_high.csv')

res = np.r_[res1.mean(axis=0),res2.mean(axis=0)]
res_time = np.r_[np.log10(res_time1).mean(axis=0),np.log10(res_time2).mean(axis=0)]

res1_BSS = np.loadtxt(f'res/{metric}_low_BSS.csv')
res2_BSS = np.loadtxt(f'res/{metric}_high_BSS.csv')
res_time1_BSS = np.loadtxt(f'res/res_time_low_BSS.csv')
res_time2_BSS = np.loadtxt(f'res/res_time_high_BSS.csv') 

res_BSS = np.r_[res1_BSS.mean(axis=0),res2_BSS.mean(axis=0)]
res_time_BSS = np.r_[np.log10(res_time1_BSS).mean(axis=0),np.log10(res_time2_BSS).mean(axis=0)]

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('d (Number of nodes)', fontsize=15)
ax1.set_ylabel(metricsdict[metric], color=color, fontsize=15)

ax1.set_xticks([i for i in range(len(ds))])
ax1.set_xticklabels(ds)

ax1.plot(res, color=color, alpha=0.4, linewidth=2, linestyle='--')
ax1.plot(res_BSS, color='tab:red', alpha=0.4, linewidth=2, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color, labelsize=15)

ax2 = ax1.twinx()

color = 'blue'
ax2.set_ylabel(r'$\log_{10}$ Time (in seconds)', color=color, fontsize=15)

ax2.plot(res_time, color=color, alpha=0.5, linewidth=5)
ax2.plot(res_time_BSS, color='red', alpha=0.5, linewidth=5)
ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
fig.tight_layout() 

fig.savefig(f'Figure5right.pdf', bbox_inches='tight')