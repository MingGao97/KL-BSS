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
err_dist = 'mixed'
IC = 'BIC'

for index, metric in enumerate(metrics[1:]):
    fig, ax = plt.subplots(2,6, figsize=(16,6))

    # ld
    d = 10
    s = 3
    s0 = 4
    for j, graph_type in enumerate(['ER','SF','Complete']):
        string = f'{graph_type}_{err_dist}_{d}_{s}' if  j==2 else f'{graph_type}-{s0}_{err_dist}_{d}_{s}' 
        for ll in range(len(methods)):
            rec = np.loadtxt(f'../Figure4/result/{metric}_' + methods[ll] + '_' + string + '.csv')
            ax[0][2*j].plot(ns, np.mean(rec, axis=0), color = cols[ll], linewidth=2, alpha=0.6)
            rec = np.loadtxt(f'../Figure4/result_IC/{IC}/{metric}_' + methods[ll] + '_' + string + '.csv')
            ax[0][2*j+1].plot(ns, np.mean(rec, axis=0), color = cols[ll], linewidth=2, alpha=0.6)
        for ll in range(1,len(methods)):
            rec = np.loadtxt(f'../Figure4/result_cv_ld/{metric}_{methods[ll]}_{graph_type}.csv')
            ax[0][2*j].plot(ns, np.mean(rec, axis=0), color = cols[ll], linewidth=2, alpha=0.6, linestyle='--')
            rec = np.loadtxt(f'../Figure4/result_iccv_ld/{metric}_{methods[ll]}_{graph_type}.csv')
            ax[0][2*j+1].plot(ns, np.mean(rec, axis=0), color = cols[ll], linewidth=2, alpha=0.6, linestyle='--')
        
        rec_lasso = np.loadtxt(f'../Figure4/result_lasso/{metric}_Lasso_' + string + '.csv')
        ax[0][2*j].plot(ns, np.mean(rec_lasso, axis=0), color = 'tab:brown', linewidth=2, alpha=0.6)
        ax[0][2*j+1].plot(ns, np.mean(rec_lasso, axis=0), color = 'tab:brown', linewidth=2, alpha=0.6)

    # hd
    methods = ['BSS','klBSS']
    d = 50
    s = 10 
    s0 = 4
    for j, graph_type in enumerate(['ER','SF','Complete']):
        string = f'{graph_type}_{err_dist}_{d}_{s}' if  j==2 else f'{graph_type}-{s0}_{err_dist}_{d}_{s}' 
        for ll in [0,1]:
            rec = np.loadtxt(f'../Figure4/result/{metric}_' + methods[ll] + '_' + string + '.csv')
            ax[1][2*j].plot(ns, np.mean(rec, axis=0), color = cols[ll], linewidth=2, alpha=0.6)
            rec = np.loadtxt(f'../Figure4/result_IC/{IC}/{metric}_' + methods[ll] + '_' + string + '.csv')
            ax[1][2*j+1].plot(ns, np.mean(rec, axis=0), color = cols[ll], linewidth=2, alpha=0.6)
        rec = np.loadtxt(f'../Figure4/result_cv_hd/{metric}_klBSS_{graph_type}.csv')
        ax[1][2*j].plot(ns, np.mean(rec, axis=0), color = cols[1], linewidth=2, alpha=0.6, linestyle='--')
        rec = np.loadtxt(f'../Figure4/result_iccv_hd/{metric}_klBSS_{graph_type}.csv')
        ax[1][2*j+1].plot(ns, np.mean(rec, axis=0), color = cols[1], linewidth=2, alpha=0.6, linestyle='--')
        
        rec_lasso = np.loadtxt(f'../Figure4/result_lasso/{metric}_Lasso_' + string + '.csv')
        ax[1][2*j].plot(ns, np.mean(rec_lasso, axis=0), color = 'tab:brown', linewidth=2, alpha=0.6)
        ax[1][2*j+1].plot(ns, np.mean(rec_lasso, axis=0), color = 'tab:brown', linewidth=2, alpha=0.6)

    #
    for j in range(3):
        ax[0][2*j].set_title('known sparsity', fontsize=16)
        ax[0][2*j+1].set_title('unknown sparsity', fontsize=16)
    ax[0][0].set_ylabel(r'($d,s,\overline{s}$)'+'=(10,3,4)', fontsize=16)
    ax[1][0].set_ylabel(r'($d,s,\overline{s}$)'+'=(50,10,25)', fontsize=16)

    for j in range(6):
        ax[0][j].get_xaxis().set_ticks([])

    for i in range(2):
        for j in range(1,6):
            ax[i][j].get_yaxis().set_ticks([])


    ax[1][2].set_xlabel('                        n (sample size)', fontsize=16)

    if metric != 'hd':
        for i in range(2):
            for j in range(6):
                ax[i][j].set_ylim(-0.05, 1)
    else:
        for j in range(6):
            ax[0][j].set_ylim(-0.05, 3.5)
        for j in range(6):
            ax[1][j].set_ylim(-0.05, 10)

    # legends
    lines = [Line2D([0], [0], color=col, linewidth=4, alpha=0.8) for col in cols]
    lines += [Line2D([0], [0], color=col, linewidth=4, alpha=0.8, linestyle='--') for col in cols[1:]]
    lines += [Line2D([0], [0], color='tab:brown', linewidth=4, alpha=0.8)]
    labels = methodslabel + ['Vanilla KLBSS (CV)','KLBSS (CV)', 'Lasso']
    ax[1][2].legend(lines, labels, loc='upper center', bbox_to_anchor=(1, -0.22), ncol=6, fontsize=14)

    # adjust main title
    plt.subplots_adjust(top=0.88)
    plt.suptitle('      ER-4 graph                           SF-4 graph                           Complete graph', fontsize=20)
    fig.supylabel(metricsdict[metric], fontsize=23, x=0.06)

    # reduce space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # save figures
    fig.savefig(f'Figure{14+index}.pdf',bbox_inches='tight')
