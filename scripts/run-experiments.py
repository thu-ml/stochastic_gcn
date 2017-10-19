import os, sys

datasets    = ['citeseer', 'cora', 'pubmed', 'nell', 'ppi', 'reddit']
gcn_datasets = set(['cora', 'citeseer', 'pubmed', 'nell'])
preprocess  = ['True', 'False']
dropout = [True, False]
deg_cv_dropout_preprocess   = [(20, False, 'True', True), (20, False, 'True', False), 
                               (1, False, 'True', False), 
                               (1, False, 'True', True), (1, False, 'Fast', True), 
                               (1, True, 'True', True), (1, True, 'Fast', True)]
test_exps = [('Exact', '--test_degree 10000'),
        ('NS',    '--test_degree 1 --nopreprocess --notest_preprocess'),
        ('NSPP',  '--test_degree 1'),
        ('NSCV',  '--test_degree 1 --cv --test_cv')]

f = open('run.sh', 'w')
ftest = open('test.sh', 'w')
for data in datasets:
    for deg, cv, d, pp in deg_cv_dropout_preprocess:
        for run in range(1):
            dropout_str = ''
            if d=='Fast':
                dropout_str = '--det_dropout'
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
     'stdbuf -o 0 sh config/{}.config --early_stopping=1000000 --data={} --epochs={} {} --preprocess={} --degree={} --cv={} --test_cv={} --seed={} | tee {}'.format(data, ndata, epochs, dropout_str, pp, deg, cv, cv, run, log_file)
            f.write(command+'\n')

    log_file = 'logs/train_{}.log'.format(data)
    command = 'stdbuf -o 0 sh config/{}.config | tee {}'.format(data, log_file)
    ftest.write(command+'\n')
    for name, param in test_exps:
        log_file = 'logs/test_{}_{}.log'.format(data, name)
        command = 'stdbuf -o 0 sh config/{}.config --load {} | tee {}'.format(data, param, log_file)
        ftest.write(command+'\n')

