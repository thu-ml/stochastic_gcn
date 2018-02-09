from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.plaingcn import PlainGCN
from gcn.vrgcn import VRGCN
from scheduler import PyScheduler
from tensorflow.contrib.opt import ScipyOptimizerInterface
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparsenorm
from stats import Stat

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'graphsage', 'Model string.')  # 'graphsage', 'mlp'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Min number of epochs to train.')
flags.DEFINE_integer('data', 0, 'Max amount of visited data')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_bool('dense_input', False, 'Convert input to dense')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_integer('degree', 20, 'Neighbour subsampling size')
flags.DEFINE_integer('batch_size', 1000, 'Minibatch size for SGD')
flags.DEFINE_bool('cv', False, "Control variate")
flags.DEFINE_bool('preprocess', True,  'Preprocess first aggregation')

flags.DEFINE_integer('test_batch_size', 1000, 'Testing batch size')
flags.DEFINE_integer('test_degree', 20, 'Testing neighbour subsampling size')
flags.DEFINE_bool('test_cv', False, "Testing control variate")
flags.DEFINE_bool('test_preprocess', True,  'Preprocess first aggregation')

flags.DEFINE_integer('num_layers', 2, 'Number of layers')
flags.DEFINE_integer('num_fc_layers', 1, 'Number of FC layers')
flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam')
flags.DEFINE_float('beta2', 0.999, 'Beta2 for Adam')
flags.DEFINE_string('normalization', 'gcn', 'gcn or graphsage')
flags.DEFINE_bool('layer_norm', False, 'Layer normalization')
flags.DEFINE_float('polyak_decay', 0, 'Decay for model averaging')
flags.DEFINE_bool('load', False, 'Load the model')

flags.DEFINE_bool('det_dropout', False, 'Determinstic dropout')
flags.DEFINE_bool('cvd', False, 'CV for Dropout. Only useful when --cv is present.')
flags.DEFINE_bool('test_cvd', False, 'CV for Dropout. Only useful when --cv is present.')
flags.DEFINE_bool('importance', False, 'Importance sampling')
flags.DEFINE_bool('test_importance', False, 'Importance sampling')

flags.DEFINE_integer('seed', 1, 'Random seed')
flags.DEFINE_integer('max_degree', -1, 'Subsample the input. Maximum number of degree. For GraphSAGE.')

flags.DEFINE_bool('gradvar', False, 'Output gradient variance')
flags.DEFINE_bool('reverse', False, 'Original models')

flags.DEFINE_bool('pp_nbr', True, 'Whether pre-process use neighbors')

tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

# Load data
num_data, train_adj, full_adj, features, train_features, test_features, labels, train_d, val_d, test_d = \
        load_data(FLAGS.dataset)

if FLAGS.gradvar:
    print('Analyze mode...')
    full_adj = train_adj.copy()
    test_features = train_features.copy()

print('Features shape = {}, training edges = {}, testing edges = {}'.format(features.shape, train_adj.nnz, full_adj.nnz))
print('{} training data, {} validation data, {} testing data.'.format(
    len(train_d), len(val_d), len(test_d)))

multitask    = True if FLAGS.dataset=='ppi' else False
L            = FLAGS.num_layers-1 if FLAGS.preprocess else FLAGS.num_layers
test_L       = FLAGS.num_layers-1 if FLAGS.test_preprocess else FLAGS.num_layers

# Define placeholders
placeholders = {
    'adj':    [tf.sparse_placeholder(tf.float32, name='adj_%d'%l) for l in range(L)],
    'madj':   [tf.sparse_placeholder(tf.float32, name='madj_%d'%l) for l in range(L)],
    'fadj':   [tf.sparse_placeholder(tf.float32, name='fadj_%d'%l) for l in range(L)],
    'fields': [tf.placeholder(tf.int32, shape=(None),name='field_%d'%l) 
               for l in range(L+1)],
    'ffields': [tf.placeholder(tf.int32, shape=(None),name='ffield_%d'%l) 
               for l in range(L)],
    'scales': [tf.placeholder(tf.float32, shape=(None),name='scale_%d'%l) 
               for l in range(L)],
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1]), 
              name='labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout')
}

t = time()
print('Building model...')
train_model = VRGCN if FLAGS.cv else PlainGCN
test_model  = VRGCN if FLAGS.test_cv else PlainGCN

def model_func(model, nbr_features, adj, preprocess, is_training, cvd):
    return model(FLAGS.num_layers, preprocess, placeholders, 
                 features, nbr_features,
                 adj, cvd, multitask=multitask, is_training=is_training)

create_model = tf.make_template('model', model_func)
train_model  = create_model(train_model, nbr_features=train_features, adj=train_adj, 
                                         preprocess=FLAGS.preprocess, is_training=True, cvd=FLAGS.cvd)
test_model   = create_model(test_model,  nbr_features=test_features, adj=full_adj,
                                         preprocess=FLAGS.test_preprocess, is_training=False, cvd=FLAGS.test_cvd)

print('Finised in {} seconds'.format(time()-t))

train_degrees   = np.array([FLAGS.degree]*L, dtype=np.int32)
test_degrees    = np.array([FLAGS.test_degree]*test_L, dtype=np.int32)
train_sch = PyScheduler(train_adj, labels, L, train_degrees, placeholders, FLAGS.seed, train_d, cv=FLAGS.cv, importance=FLAGS.importance)
eval_sch  = PyScheduler(full_adj,  labels, test_L, test_degrees,  placeholders, FLAGS.seed, cv=FLAGS.test_cv, importance=FLAGS.test_importance)


# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(data):
    total_loss = 0
    total_acc  = 0
    total_pred = []
    total_labs = []

    t_test = time()
    N = len(data)

    for start in range(0, N, FLAGS.test_batch_size):
        end = min(start+FLAGS.test_batch_size, N)
        batch = data[start:end]
        feed_dict = eval_sch.batch(batch)

        los, acc, prd = test_model.run_one_step(sess, feed_dict)
        batch_size = prd.shape[0]
        total_loss += los * batch_size
        total_acc  += acc * batch_size
        total_pred.append(prd)
        total_labs.append(feed_dict[placeholders['labels']])

    total_loss /= N
    total_acc  /= N
    total_pred = np.vstack(total_pred)
    total_labs = np.vstack(total_labs)

    micro, macro = calc_f1(total_pred, total_labs, multitask)
    return total_loss, total_acc, micro, macro, (time()-t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
avg_loss = Averager(1)
avg_acc  = Averager(1)

def SGDTrain():
    if FLAGS.load:
        train_model.load(sess, load_history=FLAGS.gradvar)
        test_model.load(sess)
        sess.run([h2.assign(h1) for h1, h2 in zip(train_model.history_vars, test_model.history_vars)])
        return 

    print('Start training...')

    # Train model
    for epoch in range(100000000):
        train_sch.shuffle()
    
        t = time()
        train_model.init_counts()
        iter = 0
        tsch = 0
        while True:
            iter += 1
            t1 = time()
            feed_dict = train_sch.minibatch(FLAGS.batch_size)
            tsch += time() - t1
            if feed_dict==None:
                break
            #aa = feed_dict[placeholders['adj'][0]]
            #ff = feed_dict[placeholders['fields'][0]]
            #f0 = feed_dict[placeholders['fields'][1]]
            #print('FF shape', ff.shape)
            #for i in range(aa[0].shape[0]):
            #    if aa[0][i,0] == 0:
            #        print(aa[0][i], f0[aa[0][i,0]], ff[aa[0][i,1]])
            #print(aa[0].shape)
            #print(aa[1].sum())
            #exit(0)
            feed_dict[placeholders['dropout']] = FLAGS.dropout

            # Training step
            outs = train_model.run_one_step(sess, feed_dict)
            avg_loss.add(outs[1])
            avg_acc .add(outs[2])

        # Validation 
        cost, acc, micro, macro, duration = evaluate(val_d)
        #cost, acc, micro, macro, duration = 0, 0, 0, 0, 0
        cost_val.append(cost)
    
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), 
              "train_loss=", "{:.5f}".format(avg_loss.mean()),
              "train_acc=", "{:.5f}".format(avg_acc.mean()), 
              "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), 
              "mi F1={:.5f} ma F1={:.5f} ".format(micro, macro),
              "time=", "{:.5f}".format(time() - t),
              "ttime=", "{:.5f}".format(duration),
              "(sch {:.5f} s)".format(tsch),
              "data = {}".format(train_model.amt_data))
        G = float(2**30)
        print('TF time = {}, g time = {}, G GFLOPS = {}, NN GFLOPS = {}, field sizes = {}, adj sizes = {}, fadj sizes = {}'.format(
              train_model.run_t, train_model.g_t, train_model.g_ops/G, train_model.nn_ops/G, train_model.field_sizes, train_model.adj_sizes, train_model.fadj_sizes))
    
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break
        if train_model.amt_data >= FLAGS.data and epoch > FLAGS.epochs:
            break
    
    print("Optimization Finished!")
    train_model.save(sess)
    

def GradientVariance():
    # Testing
    batch = train_d[:FLAGS.batch_size]

    full_times = 1000
    full_preds = Stat()
    full_grads = Stat()
    for i in range(full_times):
        feed_dict = eval_sch.batch(batch)
        feed_dict[placeholders['dropout']] = FLAGS.dropout
        pred, grad = test_model.get_pred_and_grad(sess, feed_dict)
        full_preds.add(pred)
        full_grads.add(grad)

    full_preds_m = np.mean(np.abs(full_preds.mean()))
    full_grads_m = np.mean(np.abs(full_grads.mean()))
    print('Full pred stdev = {}'.format(
        np.mean(full_preds.std())/full_preds_m))
    print('Full grad stdev = {}'.format(
        np.mean(full_grads.std())/full_grads_m))

    part_times = 1000
    part_preds = Stat()
    part_grads = Stat()
    for i in range(part_times):
        feed_dict = train_sch.batch(batch)
        feed_dict[placeholders['dropout']] = FLAGS.dropout
        pred, grad = train_model.get_pred_and_grad(sess, feed_dict)
        part_preds.add(pred)
        part_grads.add(grad)
    print('Part pred bias = {}'.format(
        np.mean(np.abs(part_preds.mean()-full_preds.mean()))/full_preds_m))
    print('Part pred stdev = {}'.format(np.mean(part_preds.std())/full_preds_m))
    print('Part grad bias = {}'.format(np.mean(np.abs(full_grads.mean()-part_grads.mean()))/full_grads_m))
    print('Part grad stdev = {}'.format(np.mean(part_grads.std())/full_grads_m))
    print(full_grads_m, np.mean(part_grads.std()), np.mean(np.abs(part_grads.mean())))
    #print(full_grads.vals[0][:5], part_grads.vals[0][:5])


def Analyze2():
    np.set_printoptions(precision=4, suppress=True)
    # Testing
    batch = train_d[:1]

    num_vars = len(train_model.log_values)

    full_times = 1000
    full_values = [Stat() for _ in range(num_vars)]
    feed_dict = eval_sch.batch(batch)
    feed_dict[placeholders['dropout']] = FLAGS.dropout
    train_model.get_data(feed_dict)

    if not FLAGS.det_dropout:
        for i in range(full_times):
            acts = sess.run(train_model.log_values, feed_dict=feed_dict)
            for j in range(num_vars):
                full_values[j].add(acts[j])

        for i in range(num_vars):
            print(i)
            print(full_values[i].mean()[0,:5], full_values[i].std()[0,:5])
    else:
        for i in range(full_times):
            acts = sess.run(train_model.log_values, feed_dict=feed_dict)
            for j in range(num_vars):
                if len(acts[j]) == 2:
                    full_values[j] = acts[j]
                else:
                    full_values[j].add(acts[j])

        for i in range(num_vars):
            if isinstance(full_values[i], Stat):
                print('Stochastic {}'.format(i))
                print(full_values[i].mean()[0,:5], full_values[i].std()[0,:5])
            else:
                print('Deterministic {}'.format(i))
                print(full_values[i][0][0,:5], np.sqrt(full_values[i][1][0,:5]))


def Test():
    # Testing
    test_cost, test_acc, micro, macro, test_duration = evaluate(test_d)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "mi F1={:.5f} ma F1={:.5f} ".format(micro, macro),
          "time=", "{:.5f}".format(test_duration))
    remaining = np.array(list(set(range(num_data)) - set(test_d)), dtype=np.int32)
    if FLAGS.test_cv:
        evaluate(remaining)


SGDTrain()

if FLAGS.gradvar:
    GradientVariance()
#Analyze()
#Analyze2()

num_runs = FLAGS.num_layers + 1 if FLAGS.test_cv else 1
for i in range(num_runs):
    Test()

#def output_info(sch, verbose=False, val=False):
#    model.backup_model(sess)
#    batch = test_d[:1]
#    feed_dict = sch.batch(batch)
#    feed_dict[placeholders['is_training']] = False
#    feed_dict[placeholders['alpha']] = 1.0
#    model.get_data(feed_dict, False)
#    act = sess.run(model.activations, feed_dict=feed_dict)
#
#    f0 = feed_dict[placeholders['fields'][0]]
#    f1 = feed_dict[placeholders['fields'][1]]
#    perm = np.argsort(f0)
#    f0   = f0[perm]
#    if verbose:
#        for a in act:
#            print(a.shape)
#        a3   = act[3][perm]
#        for i in range(a3.shape[0]):
#            print(f0[i], a3[i][-3:])
#        print('Agg {}'.format(act[4][0, -3:]))
#
#    if val:
#        fadj = model.adj[f1]
#        f0 = fadj.tocoo().col
#        perm = np.argsort(f0)
#        f0   = f0[perm]
#        hist = model.history[0][f0]
#        history      = feed_dict[model.history_ph[0]]
#        history_mean = feed_dict[model.history_mean_ph[0]]
#        print('A3 {}'.format(act[3][:,-3:]))
#        print('History {}'.format(history[:,-3:]))
#        print('History mean {}'.format(history_mean[0,-3:]))
#        print('Agg {}'.format(act[4][0, -3:]))
#
#
#    print('Final act: {}'.format(act[-1][0][:3]))
#    model.restore_model(sess)
#
#output_info(eval_sch, verbose=True)
#for i in range(2):
#    output_info(train_sch, val=True)
