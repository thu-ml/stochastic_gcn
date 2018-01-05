from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, np_dropout, np_sparse_dropout
from gcn.models import GCN
import numpy as np
from history import slice, dense_slice

flags = tf.app.flags
FLAGS = flags.FLAGS


class VRGCN(GCN):
    def __init__(self, L, preprocess, placeholders, 
                 features, nbr_features, adj, cvd,
                 **kwargs):
        super(VRGCN, self).__init__(L, preprocess, placeholders, 
                                    features, nbr_features, 
                                    adj, cvd, **kwargs)

    def _build_history(self):
        # Create history after each aggregation
        self.history         = []
        for i in range(self.L):
            dims = self.agg0_dim if i==0 else FLAGS.hidden1
            n_history = 2 if FLAGS.det_dropout else 1
            histories = []
            for h in range(n_history):
                history = tf.Variable(tf.zeros((self.num_data, dims), dtype=np.float32), 
                                      trainable=False, name='history_{}_{}'.format(i, h))
                histories.append(history)

            self.history.append(histories)
            print('History size = {} GB'.format(self.num_data*dims*4*n_history/1024.0/1024.0/1024.0))
            

    def get_data(self, feed_dict):
        input = self.features
        f0    = feed_dict[self.placeholders['fields'][0]]
        if self.sparse_input:
            input = slice(input, f0)
        else:
            input = dense_slice(input, f0)
            #input = input[f0,:]
        feed_dict[self.inputs_ph] = input

        # Read history
        for l in range(self.L):
            ofield = feed_dict[self.placeholders['fields'][l+1]]
            adj = feed_dict[self.placeholders['adj'][l]][0]
            fadj = feed_dict[self.placeholders['fadj'][l]][0]

            dim = self.agg0_dim if l==0 else FLAGS.hidden1
            g_ops = (fadj.shape[0] + adj.shape[0]) * dim * 4
            if self.cvd:
                g_ops *= 2
            self.g_ops += g_ops
            self.adj_sizes[l] += adj.shape[0]
            self.fadj_sizes[l] += fadj.shape[0]
            self.amt_data += adj.shape[0]
        for l in range(self.L+1):
            self.field_sizes[l] += feed_dict[self.placeholders['fields'][l]].size
        for c, l in self.layer_comp:
            nn_ops = c * feed_dict[self.placeholders['fields'][l]].size * 4
            if self.cvd:
                nn_ops *= 2
            self.nn_ops += nn_ops

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
        madjs  = self.placeholders['madj']
        for l in range(self.L):
            ifield  = self.placeholders['fields'][l]
            ffield  = self.placeholders['ffields'][l]
            scale   = self.placeholders['scales'][l]
            history = self.history[l]
            agg = VRAggregator(adjs[l], fadjs[l], madjs[l],
                               ifield, ffield,
                               history, scale, self.cvd, name='agg%d'%l)
            self.aggregators.append(agg)

