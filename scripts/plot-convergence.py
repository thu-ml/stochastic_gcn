import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

#datasets   = ['cora', 'pubmed', 'nell', 'citeseer', 'ppi']
#exps1 = [(20, False, True, True,  'k',  'Batch'),
#         (1,  False, True, False, 'r:', '-CV, -PP'),
#         (1,  False, True, True,  'r',  '-CV, +PP'),
#         (1,  True,  True, False, 'b:', '+CV, -PP'),
#         (1,  True,  True, True,  'b',  '+CV, +PP')]
#exps2 = [(20, False, True,  True,  'k',  'Batch, +dropout'),
#         (20, False, False, True,  'k:', 'Batch'),
#         (1,  False, False, False, 'r:', '-CV, -PP'),
#         (1,  False, False, True,  'r',  '-CV, +PP'),
#         (1,  True,  False, False, 'b:', '+CV, -PP'),
#         (1,  True,  False, True,  'b',  '+CV, +PP')]
#all_exps = [exps1, exps2]
#num_runs = 10
#dir='logs_old'

datasets   = [('citeseer', 10), ('cora', 10), ('pubmed', 10), ('nell', 10), ('ppi', 1), ('reddit', 1)]
#datasets   = ['reddit']
exps1 = [(20, 'False', 'True', True,  '#000000', 'Batch'),               # k
         #(20, False, 'Fast', True,  '#FF0000', 'Batch+Det'),           # r
         (1,  'False', 'True', False, '#777777', 'SGD'),               # 0.5k
         (1,  'False', 'True', True,  '#0000FF',  'SGD+PP'),            # b
         #(1,  False, 'Fast', True,  '#FF00FF',  'SGD+PP+Det'),        # (r, b)
         (1,  'True',  'True', True,  '#00FF00',  'SGD+PP+CV'),         # g
         #(1,  'True',  'Fast', True,  '#FFFF00',  'SGD+PP+CV+Det'),     # (r, g)
         (1,  'TrueD', 'True', True,  '#FF0000',  'SGD+PP+CV2')]   
all_exps = [exps1]
dir='logs'

iafig, iaax = plt.subplots(2, 3, figsize=(16, 8))
ilfig, ilax = plt.subplots(2, 3, figsize=(16, 8))
dafig, daax = plt.subplots(2, 3, figsize=(16, 8))
dlfig, dlax = plt.subplots(2, 3, figsize=(16, 8))
for ndata, data in enumerate(datasets):
    data, num_runs = data
    for nexp, exps in enumerate(all_exps):
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))
        cnt = 0
        handles  = []
        losses   = []
        accs     = []
        train_losses = []
        types    = []
        iters    = []
        amt_data = []
        units    = []
        legends  = []
        colors = {}
    
        cnt = 0
        for deg, cv, dropout, pp, style, text in exps:
            if data == 'nell' and not pp:
                continue
            colors[cnt] = style
            legends.append(text)
            my_amt_data = []
            for run in range(num_runs):
                log_file = '{}/{}_pp{}_dropout{}_deg{}_cv{}_run{}.log'.format(
                            dir, data, pp, dropout, deg, cv, run)
                N = 0
                with open(log_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.find('Epoch') != -1:
                            line = line.replace('=', ' ').split()
                            losses.append(float(line[7]))
                            accs.append(float(line[12]) if data=='ppi' else float(line[9]))
                            train_losses.append(float(line[3]))
                            N += 1
                            if N > len(my_amt_data):
                                my_amt_data.append(float(line[-1]))
                            units.append(run)
    
                amt_data.extend(my_amt_data[:N])
                iters.extend(range(N))
                types.extend([cnt]*N)
    
            cnt += 1
    
        df = pd.DataFrame(data={'loss': losses, 'acc': accs, 'type': types, 'iter': iters, 'data': amt_data, 'run': units, 'train_loss': train_losses})
    
        # iter - trainloss
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="train_loss", ax=ax[0,0], legend=False, color=colors)
        # data - trainloss
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="train_loss", ax=ax[1,0], legend=False, color=colors)
        # iter - loss
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="loss", ax=ax[0,1], legend=False, color=colors)
        # data - loss
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="loss", ax=ax[1,1], legend=False, color=colors)
        # iter - acc
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="acc", ax=ax[0,2], legend=False, color=colors)
        # data - acc
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="acc", ax=ax[1,2], legend=False, color=colors)

        ia_ax = iaax[ndata//3, ndata%3]
        il_ax = ilax[ndata//3, ndata%3]
        da_ax = daax[ndata//3, ndata%3]
        dl_ax = dlax[ndata//3, ndata%3]

        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="acc", ax=ia_ax, legend=False, color=colors)
        ia_ax.legend(ia_ax.lines, legends)
        sns.tsplot(data=df, time="iter", unit="run", condition="type", value="loss", ax=il_ax, legend=False, color=colors)
        il_ax.legend(il_ax.lines, legends)
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="acc", ax=da_ax, legend=False, color=colors)
        da_ax.legend(da_ax.lines, legends)
        sns.tsplot(data=df, time="data", unit="run", condition="type", value="loss", ax=dl_ax, legend=False, color=colors)
        dl_ax.legend(dl_ax.lines, legends)

        ia_ax.set_title(data)
        il_ax.set_title(data)
        da_ax.set_title(data)
        dl_ax.set_title(data)
    
        ax[0,2].legend(ax[0,0].lines, legends)
        #ax[0,2].axis('off')
        #ax[1,2].axis('off')
    
    
        ax[0,0].set_xlabel('Number of iterations')
        ax[0,1].set_xlabel('Number of iterations')
        il_ax.set_xlabel('Number of iterations')
        ax[0,2].set_xlabel('Number of iterations')
        ia_ax.set_xlabel('Number of iterations')
        ax[1,0].set_xlabel('Amount of data seen')
        ax[1,1].set_xlabel('Amount of data seen')
        dl_ax.set_xlabel('Amount of data seen')
        ax[0,0].set_ylabel('Training loss')
        ax[0,1].set_ylabel('Validation loss')
        il_ax.set_ylabel('Validation loss')
        ax[0,2].set_ylabel('Validation accuracy')
        ia_ax.set_ylabel('Validation accuracy')
        ax[1,0].set_ylabel('Training loss')
        ax[1,1].set_ylabel('Validation loss')
        dl_ax.set_ylabel('Validation loss')
        ax[1,2].set_ylabel('Validation accuracy')
        da_ax.set_ylabel('Validation accuracy')
        ylim = (0, 0)
        if data=='cora':
            ylim = (0.725, 0.80)
        elif data=='pubmed':
            ylim = (0.725, 0.81)
        elif data=='citeseer':
            ylim = (0.64, 0.72)
        elif data=='nell':
            ylim = (0.4, 0.7)
        elif data=='ppi':
            ylim = (0.7, 1.0)
        elif data=='reddit':
            ylim = (0.92, 0.97)
        else:
            ylim = (0.55, 0.7)
        if data=='ppi' or data=='reddit':
            xlim = (0, 1e7)
        elif data=='pubmed':
            xlim = (0, 4e4)
        elif data=='nell':
            xlim = (0, 6e4)
        elif data=='citeseer':
            xlim = (0, 4e4)
        else:
            xlim = (0, 8e4)
        if data=='ppi':
            xlim0 = (0, 800)
        elif data=='reddit':
            xlim0 = (0, 50)
        else:
            xlim0 = (0, 200)
    
        print(ylim, xlim, xlim0)
        if ylim[0] != 0:
            ax[0,2].set_ylim(ylim)
            ia_ax.set_ylim(ylim)
            ax[1,2].set_ylim(ylim)
            da_ax.set_ylim(ylim)
        ax[1,0].set_xlim(xlim)
        ax[1,1].set_xlim(xlim)
        dl_ax.set_xlim(xlim)
        ax[1,2].set_xlim(xlim)
        da_ax.set_xlim(xlim)
        ax[0,0].set_xlim(xlim0)
        ax[0,1].set_xlim(xlim0)
        il_ax.set_xlim(xlim0)
        ax[0,2].set_xlim(xlim0)
        ia_ax.set_xlim(xlim0)
        
        fig.savefig('{}_{}.pdf'.format(data, nexp))

iafig.savefig('iter-acc.pdf') 
ilfig.savefig('iter-loss.pdf') 
dafig.savefig('data-acc.pdf')
dlfig.savefig('data-loss.pdf')
