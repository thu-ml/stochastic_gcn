from __future__ import division
from __future__ import print_function

from time import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.models import GraphSAGE
from scheduler import PyScheduler
from sklearn.metrics import f1_score

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'reddit', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'fastgcn', 'Model string.')  # 'fastgcn', 'vrgcn'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batch_size', 512, 'Minibatch size for SGD')
flags.DEFINE_integer('num_layers', 2, 'Number of layers')
flags.DEFINE_bool('vr', True, 'Variance reduction for vrgcn')

# Load data
num_data, train_adj, full_adj, features, labels, train_d, val_d, test_d = \
        load_graphsage_data('reddit/reddit')
print('Features shape = {}'.format(features.shape))

L = FLAGS.num_layers

# Some preprocessing
# Ax = train_adj.dot(features)

# Define placeholders
placeholders = {
    'adj':    [tf.sparse_placeholder(tf.float32, name='adj_%d'%l) 
               for l in range(L)],
    'fields': [tf.placeholder(tf.int32, shape=(None),name='field_%d'%l) 
               for l in range(L+1)],
    'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1]), 
              name='labels'),
    'dropout': tf.placeholder_with_default(0., shape=(), name='dropout')
}

degrees   = np.array([10, 25], dtype=np.int32)
model     = GraphSAGE(placeholders, features)
pred      = model.predict()
train_sch = PyScheduler(train_adj, labels, L, degrees, placeholders, train_d)
eval_sch  = PyScheduler(full_adj, labels, L, degrees, placeholders)


# Initialize session
sess = tf.Session()


def calc_f1(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")


# Define model evaluation function
def evaluate(data):
    feed_dict_val = eval_sch.batch(data)
    t_test = time()
    los, acc, prd = sess.run([model.loss, model.accuracy, pred], 
                             feed_dict=feed_dict_val)
    micro, macro  = calc_f1(prd, feed_dict_val[placeholders['labels']])
    return los, acc, micro, macro, (time()-t_test)


# Init variables
print('Loading data to GPU...')
t = time()
sess.run(tf.global_variables_initializer())
print('Finished in {} seconds'.format(time()-t))

cost_val = []
avg_loss = Averager(100)
avg_acc  = Averager(100)

# Train model
for epoch in range(FLAGS.epochs):
    train_sch.shuffle()

    t = time()
    iter = 0
    while True:
        iter += 1
        t1 = time()
        feed_dict = train_sch.minibatch(FLAGS.batch_size)
        tsch = time() - t1
        if feed_dict==None:
            break
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

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
          "time=", "{:.5f}".format(time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(test_d)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
