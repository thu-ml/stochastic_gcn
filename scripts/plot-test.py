import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import os, sys
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

datasets   = ['cora', 'pubmed', 'nell', 'citeseer', 'ppi', 'reddit']
exps = [('NS',    'NS'),
        ('NSPP',  'NS+PP'),
        ('NSCV',  'CV'),
        ('Exact', 'Exact')]
colors = ['#777777', '#0000FF', '#00FF00', '#000000']
dir = 'logs'

accs  = []
algos = []
data  = []
for d in datasets:
    for e, etitle in exps:
        if e == 'NS' and d == 'nell':
            continue
        log_file = '{}/test_{}_{}.log'.format(dir, d, e)
        with open(log_file) as f:
            lines = f.readlines()
            lines = [line.replace('=', ' ').split() for line in lines if line.find('Test')!=-1]
            acc = float(lines[-1][9]) if d=='ppi' else float(lines[-1][6])

            accs.append(acc)
            algos.append(etitle)
            data.append(d)

df = pd.DataFrame(data={'Testing accuracy': accs, 'Algorithm': algos, 'Dataset': data})
print(df)
g = sns.factorplot(x='Dataset', y='Testing accuracy', hue='Algorithm', data=df, kind='bar', aspect=2, size=2, palette=colors)
g.savefig('test.pdf')
os.system('pdfcrop test.pdf test.pdf')
