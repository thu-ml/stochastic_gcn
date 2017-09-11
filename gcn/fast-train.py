from __future__ import division
from __future__ import print_function

import time
import sys
import tensorflow as tf

from gcn.utils import *
from gcn.models import FastGCN
from scheduler import schedule

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('dropconnect', 0.2, 'Dropconnect rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('unlabeled_weight', 0, 'Weight of unlabeled data.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batch_size', 200, 'Minibatch size for SGD')

# Load data
if FLAGS.dataset != 'nell':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_nell_data(FLAGS.dataset)
y = y_train + y_val + y_test
print('Features shape = {}'.format(features.shape))

# Some preprocessing
features = tuple_to_coo(preprocess_features(features)).tocsr()
support = tuple_to_coo(preprocess_adj(adj)).astype(np.float32)
num_supports = 1
model_func = FastGCN
train_d = np.nonzero(train_mask)[0].astype(np.int32)
test_d = np.nonzero(test_mask)[0].astype(np.int32)
val_d = np.nonzero(val_mask)[0].astype(np.int32)

# Define placeholders
placeholders = {
    'support_1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support_2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features.shape, dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features.shape[1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, data, placeholders):
    batches = schedule(support, data, L=2, batch_size=len(data), dropconnect=0.0)
    t_test = time.time()
    feed_dict_val = fast_construct_feed_dict(features, batches[0], labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    batches = schedule(support, train_d, L=2, batch_size=FLAGS.batch_size, dropconnect=FLAGS.dropconnect)

    t = time.time()
    for batch in batches:
        print(batch.fields[0].shape)
        # Construct feed dictionary
        feed_dict = fast_construct_feed_dict(features, batch, y, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y, val_d, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_d, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
