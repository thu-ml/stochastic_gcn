import os, sys

datasets    = ['citeseer', 'ppi']
gcn_datasets = set(['cora', 'citeseer', 'pubmed', 'nell'])
preprocess  = ['True', 'False']
dropout = [True, False]
deg_cv   = [(20, False), (1, False), (1, True)]

f = open('run.sh', 'w')
for data in datasets:
    for pp in preprocess:
        for d in dropout:
            for deg, cv in deg_cv:
                for run in range(10):
                    # Dropout 
                    dropout_str = ""
                    if not d:
                        dropout_str = "--dropout 0"
    
                    log_file = 'logs/{}_pp{}_dropout{}_deg{}_cv{}_run{}.log'.format(data, pp, d, deg, cv, run)
                    if data in set(['cora', 'citeseer', 'pubmed', 'nell']):
                        ndata  = 50000
                        epochs = 400
                    elif data == 'ppi':
                        ndata  = int(1e7)
                        epochs = 200
                    else:
                        ndata  = int(4e7)
                        epochs = 30

                    command = \
'stdbuf -o 0 sh config/{}.config --early_stopping=1000000 --data={} --epochs={} {} --preprocess={} --degree={} --cv={} --seed={} | tee {}'.format(data, ndata, epochs, dropout_str, pp, deg, cv, run, log_file)
                    f.write(command+'\n')
