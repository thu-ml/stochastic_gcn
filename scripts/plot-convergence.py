import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np

data_dir   = 'logs'
datasets   = ['cora', 'citeseer', 'pubmed', 'nell', 'ppi', 'reddit']

exps = [(20, 1, True, True, 'k', 'B=20, preprocess'),
        (1, 1, True, True, 'r', 'B=1, preprocess'),
        (1, 1, True, False, 'r:', 'B=1, nopreprocess'),
        (1, -1, True, True, 'b', 'VR, B=1, preprocess'),
        (1, -1, True, False, 'b:', 'VR, B=1, nopreprocess')]
#exps = [(20, 1, True, True, 'k', 'B=20, preprocess'),
#        (20, 1, False, True, 'k--', 'B=20, preprocess, nodropout'),
#        (1, 1, False, True, 'r--', 'B=1, preprocess, nodropout'),
#        (1, -1, False, True, 'b--', 'VR, B=1, preprocess, nodropout')]

for data in datasets:
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    cnt = 0
    handles = []
    for deg, a, dropout, pp, style, _ in exps:
        dropout_rate = 0
        if dropout:
            if data in set(['cora', 'citeseer', 'pubmed', 'nell']):
                dropout_rate = 0.5
            else:
                dropout_rate = 0.2

        log_file = 'logs/{}_pp{}_dropout{}_deg{}_a{}.log'.format(
                    data, pp, dropout_rate, deg, a)
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
    ylim = (0, 0)
    if data=='cora':
        ylim = (0.6, 0.81)
    elif data=='pubmed':
        ylim = (0.6, 0.81)
    elif data=='citeseer':
        ylim = (0.60, 0.75)
    elif data=='nell':
        ylim = (0.0, 0.7)
    elif data=='ppi':
        ylim = (0.7, 1.0)
    elif data=='reddit':
        ylim = (0.85, 1.0)
    else:
        ylim = (0.55, 0.7)
    if data=='ppi' or data=='reddit':
        xlim = (0, 1e7)
    else:
        xlim = (0, 100000)
    if data=='ppi':
        xlim0 = (0, 400)
    elif data=='reddit':
        xlim0 = (0, 50)
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
    
