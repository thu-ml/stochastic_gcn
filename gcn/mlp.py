from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, dropout
from gcn.models import Model
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

class NeighbourMLP(Model):
    def __init__(self, data_per_fold, L, preprocess, placeholders, 
                 features, features1, adj,
                 **kwargs):
        super(NeighbourMLP, self).__init__(**kwargs)

        self.data_per_fold = data_per_fold
        self.preprocess    = preprocess
        self.placeholders  = placeholders
        self.features      = features
        self.features1     = features1
        self.adj           = adj
        self.L             = FLAGS.num_fc_layers

        self.build()

    def _preprocess(self):
        # Preprocess aggregation
        print('Preprocessing aggregations')
        start_t = time()

        # Create all the features
        def _create_features(X, A):
            features = [X]
            print('Hops = {}'.format(FLAGS.num_layers))
            for i in range(FLAGS.num_layers):
                features.append(A.dot(features[-1]))
            return np.hstack(features)

        self.features = _create_features(self.features, self.adj)
        self.input_dim      = self.features.shape[1]
        print('Finished in {} seconds.'.format(time() - start_t))


    def get_data(self, feed_dict, is_training):
        ids = feed_dict[self.placeholders['fields'][-1]]
        if self.sparse_input:
            feed_dict[self.inputs_ph] = sparse_to_tuple(self.features[ids])
        else:
            feed_dict[self.inputs_ph] = self.features[ids]


    def run_one_step(self, sess, feed_dict, is_training):
        self.get_data(feed_dict, is_training)

        # Run
        if is_training:
            outs, hist, values = sess.run([[self.opt_op, self.loss, self.accuracy], 
                                           self.history_ops, self.average_get_ops],
                                  feed_dict=feed_dict)
        else:
            outs, hist, values = sess.run([[self.loss, self.accuracy, self.pred],
                                           self.history_ops, self.average_get_ops],
                                     feed_dict=feed_dict)
        self.average_model(values)

        return outs


    def _build(self):
        for l in range(0, self.L-1):
            input_dim = self.input_dim if l==0 else FLAGS.hidden1
            self.layers.append(Dropout(1-self.placeholders['dropout'],
                                       self.placeholders['is_training']))
            self.layers.append(Dense(input_dim=input_dim,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     logging=self.logging,
                                     sparse_inputs=self.sparse_input if l==0 else False,
                                     norm=FLAGS.layer_norm,
                                     name='dense%d'%l))

        input_dim     = self.input_dim    if self.L==1 else FLAGS.hidden1
        sparse_inputs = self.sparse_input if self.L==1 else False
        self.layers.append(Dropout(1-self.placeholders['dropout'],
                                   self.placeholders['is_training']))
        self.layers.append(Dense(input_dim=input_dim,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 logging=self.logging,
                                 sparse_inputs=sparse_inputs,
                                 norm=False,
                                 name='dense%d'%(self.L-1)))
