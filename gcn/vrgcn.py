from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, dropout
from gcn.models import GCN
import numpy as np
from history import slice

flags = tf.app.flags
FLAGS = flags.FLAGS


class VRGCN(GCN):
    def __init__(self, L, preprocess, placeholders, 
                 features, nbr_features, adj, 
                 **kwargs):
        super(VRGCN, self).__init__(L, preprocess, placeholders, 
                                    features, nbr_features,
                                    adj, **kwargs)

    def _build_history(self):
        # Create history after each aggregation
        self.history         = []
        for i in range(self.L):
            dims = self.agg0_dim if i==0 else FLAGS.hidden1
            history = tf.Variable(tf.zeros((self.num_data, dims), dtype=np.float32), 
                                  trainable=False, name='history_{}'.format(i))
            self.history.append(history)
            print('History size = {} GB'.format(self.num_data*dims*4/1024.0/1024.0/1024.0))
            

    def get_data(self, feed_dict):
        input = self.features
        f0    = feed_dict[self.placeholders['fields'][0]]
        if self.sparse_input:
            input = slice(input, f0)
        else:                           # TODO dense slicing is slow?
            input = input[f0,:]
        feed_dict[self.inputs_ph] = input

        # Read history
        for l in range(self.L):
            ofield = feed_dict[self.placeholders['fields'][l+1]]
            fadj = slice(self.adj, ofield)
            adj = feed_dict[self.placeholders['adj'][l]][0]
            feed_dict[self.placeholders['fadj'][l]] = fadj

            dim = self.agg0_dim if l==0 else FLAGS.hidden1
            self.g_ops += fadj[0].shape[0] * dim * 2
            self.g_ops += adj.shape[0] * dim * 2
            self.adj_sizes[l] += adj.shape[0]
            self.fadj_sizes[l] += fadj[0].shape[0]
        for l in range(self.L+1):
            self.field_sizes[l] += feed_dict[self.placeholders['fields'][l]].size
        for c, l in self.layer_comp:
            self.nn_ops += c * feed_dict[self.placeholders['fields'][l]].size * 4

    def run_one_step(self, sess, feed_dict):
        t = time()
        self.get_data(feed_dict)
        self.g_t += time() - t

        # Run
        t = time()
        if self.is_training:
            outs = sess.run([self.train_op, self.loss, self.accuracy], feed_dict=feed_dict)
        else:
            outs, _ = sess.run([[self.loss, self.accuracy, self.pred], self.test_op], feed_dict=feed_dict)
        self.run_t += time() - t

        return outs

    def get_pred_and_grad(self, sess, feed_dict):
        self.get_data(feed_dict)

        # Run
        pred, grads = sess.run([self.pred, self.grads], 
                               feed_dict=feed_dict)

        return pred, grads


    def _build_aggregators(self):
        adjs   = self.placeholders['adj']
        fadjs  = self.placeholders['fadj']
        for l in range(self.L):
            ifield  = self.placeholders['fields'][l]
            history = self.history[l]
            agg = VRAggregator(adjs[l], fadjs[l],
                               ifield,
                               history, name='agg%d'%l)
            self.aggregators.append(agg)

