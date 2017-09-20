from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.models import GraphSAGE, NeighbourMLP
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
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('batch_size', 1000, 'Minibatch size for SGD')
flags.DEFINE_integer('num_layers', 2, 'Number of layers')
flags.DEFINE_integer('num_hops', 3, 'Number of neighbour hops')
flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam')
flags.DEFINE_float('beta2', 0.999, 'Beta2 for Adam')
flags.DEFINE_string('normalization', 'gcn', 'gcn or graphsage')
flags.DEFINE_bool('layer_norm', False, 'Layer normalization')

# Load data
num_data, train_adj, full_adj, features, labels, train_d, val_d, test_d = \
        load_data(FLAGS.dataset)
print('Features shape = {}'.format(features.shape))
multitask = True if FLAGS.dataset=='ppi' else False
sparse_input = isinstance(features, sp.csr.csr_matrix)

L = FLAGS.num_layers
if L==2:
    train_degrees   = np.array([1, 10000], dtype=np.int32)
    test_degrees    = np.array([1, 10000], dtype=np.int32)
else:
    train_degrees   = np.array([1, 1, 1], dtype=np.int32)
    test_degrees    = np.array([1, 1, 1], dtype=np.int32)

# Define placeholders
placeholders = {
    'adj':    [tf.sparse_placeholder(tf.float32, name='adj_%d'%l) 
               for l in range(L)],
    'fields': [tf.placeholder(tf.int32, shape=(None),name='field_%d'%l) 
               for l in range(L+1)],
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1]), 
              name='labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
    'is_training': tf.placeholder(tf.bool, shape=(), name='is_training')
}

if FLAGS.model == 'graphsage':
    model     = GraphSAGE(L, placeholders, features, train_adj, full_adj, multitask=multitask)
else:
    model     = NeighbourMLP(L, placeholders, features, train_adj, full_adj, multitask=multitask)

pred      = model.predict()
train_sch = PyScheduler(train_adj, labels, L, train_degrees, placeholders, train_d)
eval_sch  = PyScheduler(full_adj,  labels, L, test_degrees,  placeholders)


# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(data):
    feed_dict_val = eval_sch.batch(data)
    feed_dict_val[placeholders['is_training']] = False
    t_test = time()
    los, acc, prd = sess.run([model.loss, model.accuracy, pred], 
                             feed_dict=feed_dict_val)
    micro, macro  = calc_f1(prd, feed_dict_val[placeholders['labels']], multitask)
    return los, acc, micro, macro, (time()-t_test)


# Init variables
print('Loading data to GPU...')
t = time()
sess.run(tf.global_variables_initializer())
sess.run(model.pre_processing_ops, feed_dict=model.pre_processing_dict)
print('Finished in {} seconds'.format(time()-t))

cost_val = []
avg_loss = Averager(1)
avg_acc  = Averager(1)

def SGDTrain():
    # Train model
    for epoch in range(FLAGS.epochs):
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
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            feed_dict[placeholders['is_training']] = True

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            avg_loss.add(outs[1])
            avg_acc .add(outs[2])
            if iter % 100 == 0:
                print(avg_loss.mean(), avg_acc.mean(), tsch, 
                      feed_dict[placeholders['adj'][0]][0].shape, features.shape)
    
        # Validation
        cost, acc, micro, macro, duration = evaluate(val_d)
        cost_val.append(cost)
    
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), 
              "train_loss=", "{:.5f}".format(avg_loss.mean()),
              "train_acc=", "{:.5f}".format(avg_acc.mean()), 
              "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), 
              "mi F1={:.5f} ma F1={:.5f} ".format(micro, macro),
              "time=", "{:.5f}".format(time() - t),
              "(sch {:.5f} s)".format(tsch))
    
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break
    
    print("Optimization Finished!")
    
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
