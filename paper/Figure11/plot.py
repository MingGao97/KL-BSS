import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


N = 200
ds = [8,9,10,50] 
ss = [2,3,4] 
hss = [10,15,20]
graph_types = ['ER-1','ER-2','ER-4',
               'SF-1','SF-2','SF-4',
               'Bipartite','Complete']
err_dists = ['Gaussian', 't', 'Laplace', 'unif', 'mixed'] # 0-4
ns = np.arange(1000,8001,1000)
methods = ['klBSS_vanilla','klBSS_simple']
hmethods = ['klBSS']
methodslabel = ['Vanilla KLBSS','KLBSS']
cols = ['tab:blue','tab:green']
metrics = ['rec', 'hd', 'fdr', 'tpr']
metricsdict = {'rec' : 'P(Support recovery)',
                'hd' : 'Hamming distance',
                'fdr' : 'FDR',
                'tpr' : 'TPR'}
ICs = ['known','BIC','EBIC','Delta']

ct = np.zeros((len(metrics),len(methods),len(ns)))
ct_success = np.zeros((len(metrics),len(methods),len(ns)))
ct_even = np.zeros((len(metrics),len(methods),len(ns)))
ct_even_recover = np.zeros((len(methods),len(ns)))
for m, metric in enumerate(metrics):
    for graph_type in graph_types:
        for d in ds:
            sst = hss if d == 50 else ss
            meths = hmethods if d == 50 else methods
            for s in sst:
                for err_dist in err_dists:
                    for ic in ICs:
                        if d != 50:
                            if ic == 'known':
                                f_bss = f'../Figure4/result/{metric}_BSS_{graph_type}_{err_dist}_{d}_{s}.csv'
                            else:
                                f_bss = f'../Figure4/result_IC/{ic}/{metric}_BSS_{graph_type}_{err_dist}_{d}_{s}.csv'
                        else:
                            if ic == 'known':
                                f_bss = f'../Figure4/result/{metric}_BSS_{graph_type}_{err_dist}_{d}_{s}.csv'
                            else:
                                f_bss = f'../Figure4/result_IC/{ic}/{metric}_BSS_{graph_type}_{err_dist}_{d}_{s}.csv'
                        res_bss = np.loadtxt(f_bss)
                        for i, metho in enumerate(meths):
                            if d != 50:
                                if ic == 'known':
                                    f_klbss = f'../Figure4/result/{metric}_{metho}_{graph_type}_{err_dist}_{d}_{s}.csv'
                                else:
                                    f_klbss = f'../Figure4/result_IC/{ic}/{metric}_{metho}_{graph_type}_{err_dist}_{d}_{s}.csv'
                            else:
                                if ic == 'known':
                                    f_klbss = f'../Figure4/result/{metric}_{metho}_{graph_type}_{err_dist}_{d}_{s}.csv'
                                else:
                                    f_klbss = f'../Figure4/result_IC/{ic}/{metric}_{metho}_{graph_type}_{err_dist}_{d}_{s}.csv'
                            res_klbss = np.loadtxt(f_klbss)
                            for j in range(len(ns)):
                                for sample in range(N):
                                    ct[m,i,j] += 1
                                    if metric in ['rec','tpr']:
                                        if res_klbss[sample,j] > res_bss[sample,j]:
                                            ct_success[m,i,j] += 1
                                        elif res_klbss[sample,j] == res_bss[sample,j]:
                                            ct_even[m,i,j] += 1
                                        if metric == 'rec' and res_klbss[sample,j] == res_bss[sample,j] == 1:
                                            ct_even_recover[i,j] += 1
                                    else:
                                        if res_klbss[sample,j] < res_bss[sample,j]:
                                            ct_success[m,i,j] += 1
                                        elif res_klbss[sample,j] == res_bss[sample,j]:
                                            ct_even[m,i,j] += 1


metricsdict = {'rec' : r'$\mathbf{1}$(exact recovery)',
                'hd' : 'Hamming distance',
                'fdr' : 'FDR',
                'tpr' : 'TPR'}

categories = [str(n) for n in ns]
# Number of categories
n_categories = len(categories)
# Positions of the category groups
index = np.arange(n_categories)
# Width of each bar
bar_width = 0.4

fig, ax = plt.subplots(1,5,figsize=(12,3))
subcat_values = [(ct_success.sum(axis=0) / ct.sum(axis=0))[i].tolist() for i in range(len(methods))]
subcat_values_even = [(ct_even.sum(axis=0) / ct.sum(axis=0))[i].tolist() for i in range(len(methods))]
for i in range(len(methods)):
    ax[0].bar(index + i*bar_width, subcat_values_even[i], bar_width, color=cols[i], alpha=0.6)
    ax[0].bar(index + i*bar_width, subcat_values[i], bar_width, 
              bottom=subcat_values_even[i], color=cols[i])
ax[0].set_ylim(0,1.02)
ax[0].set_title('Overall', fontsize=13)
ax[0].set_ylabel('Percentage', fontsize=13)
ax[0].set_xticks(index[[1,3,5,7]] + bar_width, ['2000','4000','6000','8000'])
for j in range(4):
    ax[j+1].set_ylim(0,1.02)
    ax[j+1].get_yaxis().set_ticks([])
    ax[j+1].set_title(metricsdict[metrics[j]], fontsize=13)
    subcat_values = [(ct_success[j] / ct[j])[i].tolist() for i in range(len(methods))]
    subcat_values_even = [(ct_even[j] / ct[j])[i].tolist() for i in range(len(methods))]
    if j == 0:
        subcat_values_even_recover = [(ct_even_recover / ct[j])[i].tolist() for i in range(len(methods))]
        subcat_values_even_nonrecover = [((ct_even[j] - ct_even_recover) / ct[j])[i].tolist() for i in range(len(methods))]
    for i in range(len(methods)):
        if j == 0:
            ax[j+1].bar(index + i*bar_width, subcat_values_even_nonrecover[i], bar_width,
                         color=cols[i], alpha=0.1)
            ax[j+1].bar(index + i*bar_width, subcat_values_even_recover[i], bar_width,
                        bottom=subcat_values_even_nonrecover[i], color=cols[i], alpha=0.6)
            ax[j+1].bar(index + i*bar_width, subcat_values[i], bar_width, 
                        bottom=subcat_values_even[i], color=cols[i])
        else:
            ax[j+1].bar(index + i*bar_width, subcat_values_even[i], bar_width, color=cols[i], alpha=0.6)
            ax[j+1].bar(index + i*bar_width, subcat_values[i], bar_width, 
                        bottom=subcat_values_even[i], color=cols[i])
    ax[j+1].set_xticks(index[[1,3,5,7]] + bar_width, ['2000','4000','6000','8000'])
ax[2].set_xlabel('n (Sample size)', fontsize=13)


lines = [Line2D([0], [0], color=col, linewidth=4, alpha=0.8) for col in cols]
lines += [Line2D([0], [0], color='grey', linewidth=4),
          Line2D([0], [0], color='grey', linewidth=4, alpha=0.4)]
labels = methodslabel[:] + ['Strict improvement', 'Tied performance']
ax[2].legend(lines, labels, loc='upper center', bbox_to_anchor=(.5, -0.18), ncol=5, fontsize=10)

# adjust main title
plt.subplots_adjust(top=0.82)
plt.suptitle('Percentage of improvement over BSS', fontsize=16)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
fig.savefig('Figure11.pdf',bbox_inches='tight')