import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np

data_dir   = 'logs'
datasets   = ['cora', 'citeseer', 'pubmed', 'ppi']

exps = [(20, 1, False, True, 'k', 'B=20, preprocess'),
        (1, 1, False, True, 'r', 'B=1, preprocess'),
        (1, 1, False, False, 'r:', 'B=1, nopreprocess'),
        (1, -1, False, True, 'b', 'VR, B=1, preprocess'),
        (1, -1, False, False, 'b:', 'VR, B=1, nopreprocess')]

for data in datasets:
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    cnt = 0
    handles = []
    for deg, a, dropout, pp, style, _ in exps:
        log_file = 'logs/{}_pp{}_dropout{}_deg{}_a{}.log'.format(
                    data, pp, dropout, deg, a)
        print(log_file)
        losses   = []
        amt_data = []
        acc      = []
        with open(log_file) as f:
            lines = f.readlines()
            for line in lines:
                if line.find('Epoch') != -1:
                    line = line.split()
                    losses.append(float(line[7]))
                    acc.append(float(line[9]))
                    amt_data.append(float(line[-1]))

        if dropout:
            xs = np.arange(len(losses)) * 16
        else:
            xs = np.arange(len(losses))
        l = ax[0,0].plot(xs, losses, style, alpha=0.5)
        ax[0,1].plot(xs, acc,    style, alpha=0.5)
        ax[1,0].plot(amt_data, losses, style, alpha=0.5)
        ax[1,1].plot(amt_data, acc,    style, alpha=0.5)
        handles.append(l[0])
        cnt += 1

    ax[0,2].legend(handles, [l[-1] for l in exps])
    ax[0,2].axis('off')
    ax[1,2].axis('off')

    ax[0,0].set_xlabel('Number of iterations')
    ax[0,1].set_xlabel('Number of iterations')
    ax[1,0].set_xlabel('Amount of data seen')
    ax[1,1].set_xlabel('Amount of data seen')
    ax[0,0].set_ylabel('Validation loss')
    ax[0,1].set_ylabel('Validation accuracy')
    ax[1,0].set_ylabel('Validation loss')
    ax[1,1].set_ylabel('Validation accuracy')
    ax[0,0].set_xlim([0, 200])
    ax[0,1].set_xlim([0, 200])
    ylim = (0, 0)
    if data=='cora':
        ylim = (0.7, 0.81)
    elif data=='pubmed':
        ylim = (0.7, 0.81)
    elif data=='citeseer':
        ylim = (0.65, 0.75)
    else:
        ylim = (0.95, 1.0)
    if data=='ppi':
        xlim = (0, 1e7)
    else:
        xlim = (0, 250000)
    if data=='ppi':
        xlim0 = (0, 1000)
    else:
        xlim0 = (0, 200)

    if ylim[0] != 0:
        ax[0,1].set_ylim(ylim)
        ax[1,1].set_ylim(ylim)
    ax[1,0].set_xlim(xlim)
    ax[1,1].set_xlim(xlim)
    ax[0,0].set_xlim(xlim0)
    ax[0,1].set_xlim(xlim0)
    
    fig.savefig('{}.pdf'.format(data))
    
