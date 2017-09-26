from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.models import GraphSAGE, NeighbourMLP, DoublyStochasticGCN, VRGCN
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
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('batch_size', 1000, 'Minibatch size for SGD')
flags.DEFINE_integer('test_batch_size', 1000, 'Testing batch size')
flags.DEFINE_integer('test_degree', 20, 'Testing neighbour subsampling size')
flags.DEFINE_integer('num_layers', 2, 'Number of layers')
flags.DEFINE_integer('num_hops', 3, 'Number of neighbour hops')
flags.DEFINE_integer('degree', 10000, 'Neighbour subsampling size')
flags.DEFINE_integer('num_reps', 1, 'Number of replicas')
flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam')
flags.DEFINE_float('beta2', 0.999, 'Beta2 for Adam')
flags.DEFINE_string('normalization', 'gcn', 'gcn or graphsage')
flags.DEFINE_bool('layer_norm', False, 'Layer normalization')
flags.DEFINE_bool('preprocess', True,  'Preprocess first aggregation')

flags.DEFINE_float('alpha', 1.0, 'EMA coefficient')

# Load data
old_num_data, train_adj, full_adj, features, labels, train_d, val_d, test_d = \
        load_data(FLAGS.dataset)

num_data, adj, features, labels, train_d, val_d, test_d = \
        data_augmentation(old_num_data, train_adj, full_adj, features, labels, train_d, val_d, test_d, FLAGS.num_reps)

print('Features shape = {}'.format(features.shape))
print('{} training data, {} validation data, {} testing data.'.format(
    len(train_d), len(val_d), len(test_d)))

multitask    = True if FLAGS.dataset=='ppi' else False
sparse_input = isinstance(features, sp.csr.csr_matrix)
L            = FLAGS.num_layers-1 if FLAGS.preprocess else FLAGS.num_layers

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

if FLAGS.model == 'graphsage':
    if FLAGS.alpha == -1:
        model = VRGCN(FLAGS.num_layers, FLAGS.preprocess,
                                    placeholders, features, 
                                    train_adj, full_adj, multitask=multitask)
    else:
        model = DoublyStochasticGCN(old_num_data, FLAGS.num_layers, FLAGS.preprocess,
                                    placeholders, features, 
                                    adj, multitask=multitask)
else:
    model = NeighbourMLP(FLAGS.num_layers, placeholders, features, 
                         train_adj, full_adj, multitask=multitask)
pred      = model.predict()

train_degrees   = np.array([FLAGS.degree]*L, dtype=np.int32)
test_degrees    = np.array([10000]*L, dtype=np.int32)
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
    for start in range(0, N, FLAGS.test_batch_size):
        end = min(start+FLAGS.test_batch_size, N)
        batch = data[start:end]
        feed_dict = eval_sch.batch(batch)
        feed_dict[placeholders['is_training']] = False
        feed_dict[placeholders['alpha']] = 1.0
        model.get_data(feed_dict, False)
        los, acc, prd = sess.run([model.loss, model.accuracy, pred], 
                                 feed_dict=feed_dict)
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
    amt_data = 0
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
            # model.get_data(feed_dict, True)
            feed_dict[placeholders['dropout']] = FLAGS.dropout
            feed_dict[placeholders['is_training']] = True
            feed_dict[placeholders['alpha']] = 1.0 if epoch==0 else FLAGS.alpha
            amt_data += feed_dict[placeholders['fields'][0]].shape[0]

            # Training step
            # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            outs = model.train_one_step(sess, feed_dict, True)
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
              "(sch {:.5f} s)".format(tsch),
              "data = {}".format(amt_data))
    
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break
        if amt_data >= FLAGS.data and epoch > FLAGS.epochs:
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
