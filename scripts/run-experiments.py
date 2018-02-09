import os, sys

datasets_runs    = [('citeseer', 10), ('cora', 10), ('pubmed', 10), ('nell', 10), ('ppi', 5), ('reddit', 5)]
gcn_datasets = set(['cora', 'citeseer', 'pubmed', 'nell'])
preprocess  = ['True', 'False']
dropout = [True, False]
deg_cv_dropout_preprocess   = [(20, 'False', 'True', True),  # Exact
                               (1, 'False', 'True', False),  # NS
                               (1, 'False', 'True', True),   # NS+PP
                               (1, 'IS', 'True', True),      # IS+PP
                               (1, 'True', 'True', True),    # CV+PP
                               (1, 'TrueD', 'True', True),   # CVD+PP
                               (20, 'False', 'False', True), # Exact
                               (1, 'False', 'False', False), # NS
                               (1, 'False', 'False', True),  # NS+PP
                               (1, 'IS', 'False', True),     # IS+PP
                               (1, 'True', 'False', True)]   # CV+PP
test_exps = [('Exact', '--test_degree 10000'),
        ('NS',    '--test_degree 1 --nopreprocess --notest_preprocess'),
        ('NSPP',  '--test_degree 1'),
        ('NSCV',  '--test_degree 1 --cv --test_cv')]

var_exps = [('VarTrainCV',  '--test_degree=10000 --dropout 0 --cv --degree=1'),
            ('VarNS',       '--test_degree=10000 --dropout 0 --load --gradvar --nopreprocess --degree=1'),
            ('VarNSPP',     '--test_degree=10000 --dropout 0 --load --gradvar --degree=1'),
            ('VarCV',       '--test_degree=10000 --dropout 0 --load --gradvar --degree=1 --cv'),
            ('DVarTrainCV', '--test_degree=10000 --cv --degree=1'),
            ('DVarNS',      '--test_degree=10000 --load --gradvar --nopreprocess --degree=1'),
            ('DVarNSPP',    '--test_degree=10000 --load --gradvar --degree=1'),
            ('DVarCV',      '--test_degree=10000 --load --gradvar --degree=1 --cv'),
            ('DVarTrainCV', '--test_degree=10000 --cv --cvd --degree=1'),
            ('DVarCVD',     '--test_degree=10000 --load --gradvar --degree=1 --cv --cvd')]

fs = {}
for d, _ in datasets_runs:
    fs[d] = open('run_{}.sh'.format(d), 'w')
ftest = open('test.sh', 'w')
fvar = open('var.sh', 'w')
for data, n_runs in datasets_runs:
    for deg, cv, d, pp in deg_cv_dropout_preprocess:
        if data=='nell' and not pp:
            continue
        for run in range(n_runs):
            dropout_str = ''
            if d=='Fast':
                dropout_str = '--det_dropout'
            elif d=='False':
                dropout_str = '--dropout 0'
            log_file = 'logs/{}_pp{}_dropout{}_deg{}_cv{}_run{}.log'.format(data, pp, d, deg, cv, run)
            if data in set(['cora', 'citeseer', 'pubmed', 'nell']):
                ndata  = 0
                epochs = 400
            elif data == 'ppi':
                ndata  = 0
                epochs = 100
            else:
                ndata  = int(0)
                if pp==False and deg==1 and cv=='False':
                    epochs = 100
                else:
                    epochs = 50

            if cv=='False':
                cv_str = '--cv=False'
            elif cv=='True':
                cv_str = '--cv=True'
            elif cv=='IS':
                cv_str = '--importance'
            else:
                cv_str = '--cv --cvd'
    
            command = \
     'stdbuf -o 0 sh config/{}.config --early_stopping=1000000 --data={} --epochs={} {} --preprocess={} --degree={} {} --seed={} | tee {}'.format(data, ndata, epochs, dropout_str, pp, deg, cv_str, run, log_file)
            fs[data].write(command+'\n')

    log_file = 'logs/train_{}.log'.format(data)
    command = 'stdbuf -o 0 sh config/{}.config | tee {}'.format(data, log_file)
    ftest.write(command+'\n')
    for name, param in test_exps:
        log_file = 'logs/test_{}_{}.log'.format(data, name)
        command = 'stdbuf -o 0 sh config/{}.config --load {} | tee {}'.format(data, param, log_file)
        ftest.write(command+'\n')

    for name, param in var_exps:
        log_file = 'logs/{}_{}.log'.format(data, name)
        command = 'stdbuf -o 0 sh config/{}.config {} | tee {}'.format(data, param, log_file)
        fvar.write(command+'\n')

