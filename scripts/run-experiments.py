import os, sys

datasets_runs    = [('citeseer', 10), ('cora', 10), ('pubmed', 10), ('nell', 10), ('ppi', 1), ('reddit', 1)]
gcn_datasets = set(['cora', 'citeseer', 'pubmed', 'nell'])
preprocess  = ['True', 'False']
dropout = [True, False]
#deg_cv_dropout_preprocess   = [(20, 'False', 'True', True), (20, 'False', 'Fast', True), 
#                               (20, 'False', 'True', False), 
#                               (1,  'False', 'True', False), 
#                               (1,  'False', 'True', True), (1, 'False', 'Fast', True), 
#                               (1,  'True', 'True', True), (1, 'True', 'Fast', True),
#                               (1,  'TrueD','True', True)]

deg_cv_dropout_preprocess   = [(1,  'TrueD','True', True)]

f = open('run.sh', 'w')
for data, n_runs in datasets_runs:
    for deg, cv, d, pp in deg_cv_dropout_preprocess:
        if pp==False and data=='nell':
            continue
        for run in range(n_runs):
            dropout_str = ''
            if d=='Fast':
                dropout_str = '--det_dropout'
            if cv=='False':
                cv_str = '--cv=False'
            elif cv=='True':
                cv_str = '--cv=True'
            else:
                cv_str = '--cv --cv2'

            log_file = 'logs/{}_pp{}_dropout{}_deg{}_cv{}_run{}.log'.format(data, pp, d, deg, cv, run)
            if data in set(['cora', 'citeseer', 'pubmed', 'nell']):
                ndata  = 50000
                epochs = 400
            elif data == 'ppi':
                ndata  = int(1e7)
                epochs = 800
            else:
                ndata  = int(4e7)
                epochs = 30
    
            command = \
     'stdbuf -o 0 sh config/{}.config --early_stopping=1000000 --data={} --epochs={} {} --preprocess={} --degree={} {} --seed={} | tee {}'.format(data, ndata, epochs, dropout_str, pp, deg, cv_str, run, log_file)
            f.write(command+'\n')
