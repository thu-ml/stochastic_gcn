from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.dsgcn import DoublyStochasticGCN
from gcn.vrgcn import VRGCN
from gcn.mlp import NeighbourMLP
from scheduler import PyScheduler
from tensorflow.contrib.opt import ScipyOptimizerInterface
import scipy.sparse as sp

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
flags.DEFINE_bool('det_dropout', False, 'Deterministic dropout')
flags.DEFINE_bool('dense_input', False, 'Convert input to dense')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('batch_size', 1000, 'Minibatch size for SGD')
flags.DEFINE_integer('test_batch_size', 1000, 'Testing batch size')
flags.DEFINE_integer('test_degree', 20, 'Testing neighbour subsampling size')
flags.DEFINE_integer('num_layers', 2, 'Number of layers')
flags.DEFINE_integer('num_fc_layers', 1, 'Number of FC layers')
flags.DEFINE_integer('degree', 10000, 'Neighbour subsampling size')
flags.DEFINE_integer('num_reps', 1, 'Number of replicas')
flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam')
flags.DEFINE_float('beta2', 0.999, 'Beta2 for Adam')
flags.DEFINE_string('normalization', 'gcn', 'gcn or graphsage')
flags.DEFINE_bool('layer_norm', False, 'Layer normalization')
flags.DEFINE_bool('preprocess', True,  'Preprocess first aggregation')
flags.DEFINE_float('polyak_decay', 0, 'Decay for model averaging')
flags.DEFINE_bool('load', False, 'Load the model')

flags.DEFINE_float('alpha', 1.0, 'EMA coefficient')

# Load data
num_data, adj, features, features1, labels, train_d, val_d, test_d = \
        load_data(FLAGS.dataset)
old_num_data = num_data / (FLAGS.num_reps+1)

print('Features shape = {}'.format(features.shape))
print('{} training data, {} validation data, {} testing data.'.format(
    len(train_d), len(val_d), len(test_d)))

multitask    = True if FLAGS.dataset=='ppi' else False
sparse_input = isinstance(features, sp.csr.csr_matrix)
L            = FLAGS.num_layers-1 if FLAGS.preprocess else FLAGS.num_layers
if FLAGS.model == 'mlp':
    L = 0

# Define placeholders
placeholders = {
    'adj':    [tf.sparse_placeholder(tf.float32, name='adj_%d'%l) 
               for l in range(L)],
    'fields': [tf.placeholder(tf.int32, shape=(None),name='field_%d'%l) 
               for l in range(L+1)],
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1]), 
              name='labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'is_training': tf.placeholder(tf.bool, shape=(), name='is_training'),
    'alpha': tf.placeholder(tf.float32, shape=(), name='alpha')
}

t = time()
print('Building model...')
if FLAGS.model == 'graphsage':
    if FLAGS.alpha == -1:
        model = VRGCN
    else:
        model = DoublyStochasticGCN
else:
    model = NeighbourMLP

model = model(old_num_data, FLAGS.num_layers, FLAGS.preprocess,
                            placeholders, features, features1,
                            adj, multitask=multitask)

print('Finised in {} seconds'.format(time()-t))

train_degrees   = np.array([FLAGS.degree]*L, dtype=np.int32)
test_degrees    = np.array([FLAGS.test_degree]*L, dtype=np.int32)
train_sch = PyScheduler(adj, labels, L, train_degrees, placeholders, train_d)
eval_sch  = PyScheduler(adj, labels, L, test_degrees,  placeholders)


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
    model.backup_model(sess)

    for start in range(0, N, FLAGS.test_batch_size):
        end = min(start+FLAGS.test_batch_size, N)
        batch = data[start:end]
        feed_dict = eval_sch.batch(batch)
        feed_dict[placeholders['is_training']] = False
        feed_dict[placeholders['alpha']] = 1.0

        los, acc, prd = model.run_one_step(sess, feed_dict, is_training=False)
        batch_size = prd.shape[0]
        total_loss += los * batch_size
        total_acc  += acc * batch_size
        total_pred.append(prd)
        total_labs.append(feed_dict[placeholders['labels']])
    model.restore_model(sess)

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
    amt_data = 0

    if FLAGS.load:
        model.load(sess)
        return 

    # Train model
    for epoch in range(100000000):
        train_sch.shuffle()
    
        t = time()
        iter = 0
        tsch = 0
        while True:
            iter += 1
            t1 = time()
            feed_dict = train_sch.minibatch(FLAGS.batch_size)
            tsch += time() - t1
            if feed_dict==None:
                break
            feed_dict[placeholders['dropout']] = FLAGS.dropout
            feed_dict[placeholders['is_training']] = True
            feed_dict[placeholders['alpha']] = 1.0 if epoch==0 else FLAGS.alpha
            amt_data += feed_dict[placeholders['fields'][0]].shape[0]

            # Training step
            outs = model.run_one_step(sess, feed_dict, is_training=True)
            avg_loss.add(outs[1])
            avg_acc .add(outs[2])
            if iter % 100 == 0:
                print(avg_loss.mean(), avg_acc.mean(), tsch, 
                      feed_dict[placeholders['adj'][0]][0].shape, features.shape)

        # Validation
        cost, acc, micro, macro, duration = evaluate(val_d)
        cost_val.append(cost)
        test_cost, test_acc, test_micro, test_macro, test_duration = evaluate(test_d)
    
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
              "data = {}".format(amt_data))
    
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break
        if amt_data >= FLAGS.data and epoch > FLAGS.epochs:
            break
    
    print("Optimization Finished!")
    model.save(sess)
    

def Test():
    # Testing
    test_cost, test_acc, micro, macro, test_duration = evaluate(test_d)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), 
          "mi F1={:.5f} ma F1={:.5f} ".format(micro, macro),
          "time=", "{:.5f}".format(test_duration))

def LBFGSTrain():
    print(tf.trainable_variables())
    optimizer = ScipyOptimizerInterface(model.loss, options={'maxiter': 100})
    feed_dict = train_sch.minibatch(FLAGS.batch_size)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict[placeholders['is_training']] = True

    def lcallback(loss, acc):
        print('Training loss = {:.4f} acc = {:.4f}'.format(loss, acc))

    for i in range(3):
        optimizer.minimize(sess,
                feed_dict=feed_dict,
                fetches=[model.loss, model.accuracy],
                loss_callback=lcallback)

        test_feed_dict = eval_sch.batch(val_d)
        test_feed_dict[placeholders['is_training']] = False

        t_test = time()
        los, acc, prd = sess.run([model.loss, model.accuracy, pred], 
                                 feed_dict=test_feed_dict)
        micro, macro  = calc_f1(prd, test_feed_dict[placeholders['labels']], multitask)
        print('Val loss = {:.4f}, acc = {:.4f}, micro = {:.4f}, macro = {:.4f}'
                .format(los, acc, micro, macro))


        test_feed_dict = eval_sch.batch(test_d)
        test_feed_dict[placeholders['is_training']] = False

        t_test = time()
        los, acc, prd = sess.run([model.loss, model.accuracy, pred], 
                                 feed_dict=test_feed_dict)
        micro, macro  = calc_f1(prd, test_feed_dict[placeholders['labels']])
        print('Test loss = {:.4f}, acc = {:.4f}, micro = {:.4f}, macro = {:.4f}'
                .format(los, acc, micro, macro))

# LBFGSTrain()
SGDTrain()
for i in range(10 if FLAGS.load else 1):
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
