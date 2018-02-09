from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, np_dropout, np_sparse_dropout
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# History -> History_mean -> Loss, gradient -> History

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'multitask', 'is_training'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            #name = self.__class__.__name__.lower()
            name = 'model'
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = []
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.multitask = kwargs.get('multitask', False)

        self.history_ops = []
        self.aggregators = []

        self.log_values = []

        self.is_training = kwargs.get('is_training', True)
        if self.is_training:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, 
                                                    beta1=FLAGS.beta1, beta2=FLAGS.beta2)

    def _preprocess(self):
        raise NotImplementedError

    def _build(self):
        raise NotImplementedError

    def _build_history(self):
        pass

    def _build_aggregators(self):
        pass

    def _history(self):
        pass

    def _loss(self):
        # Weight decay loss on the first layer
        l = 0
        while len(self.layers[l].vars.values()) == 0:
            l += 1
        for var in self.layers[l].vars.values():
            print('Var')
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        if self.multitask:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.outputs, labels=self.placeholders['labels']))
        else:
            # Cross entropy error
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        if self.multitask:
            preds = self.outputs > 0
            labs  = self.placeholders['labels'] > 0.5
            correct_prediction = tf.equal(preds, labs)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            correct_prediction = tf.equal(tf.argmax(self.outputs, 1), 
                                          tf.argmax(self.placeholders['labels'], 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def get_ph(self, name):
        if self.sparse_input:
            return tf.sparse_placeholder(tf.float32, name=name)
        else:
            return tf.placeholder(tf.float32, name=name)


    def average_model(self, values):
        if FLAGS.polyak_decay > 0:
            for i in range(len(values)):
                self.average_values[i] = self.average_values[i]*FLAGS.polyak_decay + \
                                         values[i]*(1-FLAGS.polyak_decay)


    def backup_model(self, sess):
        if FLAGS.polyak_decay > 0:
            self.bak_values = sess.run(self.average_get_ops)
            sess.run(self.average_update_ops,
                     feed_dict={ph: v for ph, v in zip(self.average_phs, self.average_values)})


    def restore_model(self, sess):
        if FLAGS.polyak_decay > 0:
            sess.run(self.average_update_ops, 
                     feed_dict={ph: v for ph, v in zip(self.average_phs, self.bak_values)})
            

    def build(self):
        self.sparse_mm     = self.sparse_input
        self.inputs_ph     = self.get_ph('input')
        self.inputs        = tf.sparse_reorder(self.inputs_ph) if self.sparse_input else self.inputs_ph
        if self.sparse_input and not self.preprocess:
            print('Warning: we do not support sparse input without pre-processing. Converting to dense...')
            self.inputs    = tf.sparse_to_dense(self.inputs.indices, 
                                                 self.inputs.dense_shape,
                                                 self.inputs.values)
            self.sparse_mm = False

        self.num_data      = self.adj.shape[0]

        self.output_dim    = self.placeholders['labels'].get_shape().as_list()[1]
        self.placeholders  = self.placeholders

        self._preprocess()

        """ Wrapper for _build() """
        self._build_history()
        self._build_aggregators()
        self._build()

        self.activations.append(self.inputs)
        self.log_values.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            if isinstance(hidden, tuple):
                print('{} shape = {}'.format(layer.name, hidden[0].get_shape()))
            else:
                print('{} shape = {}'.format(layer.name, hidden.get_shape()))
            self.activations.append(hidden)
            if hasattr(layer, 'log_values'):
                self.log_values.extend(layer.log_values)

        self.outputs = self.activations[-1]
        self.update_history = []
        for l in range(self.L):
            ifield = self.placeholders['fields'][l]
            if hasattr(self.aggregators[l], 'new_history'):
                new_history = self.aggregators[l].new_history
                self.update_history.extend([tf.scatter_update(h, ifield, nh).op
                         for h, nh in zip(self.history[l], new_history)])

        self._predict()

        # Store model variables for easy access
        # Trainable variables + layer norm variables
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.vars = variables
        self.history_vars = [var[0] for var in self.history]
        print('Model variables')
        for k in self.vars:
            print(k.name, k.get_shape())
        print('History variables')
        for k in self.history_vars:
            print(k.name, k.get_shape())

        # Build metrics
        self._loss()
        self._accuracy()

        if self.is_training:
            self.opt_op = [self.optimizer.minimize(self.loss)]
            self.train_op = []
            with tf.control_dependencies(self.opt_op):
                self.train_op = tf.group(*self.update_history)
            self.test_op = tf.group(*self.update_history)
        else:
            self.train_op = tf.group(*self.update_history)
            self.test_op  = self.train_op

        self.grads  = tf.gradients(self.loss, self.vars[0])

    def _predict(self):
        if self.multitask:
            self.pred = tf.nn.sigmoid(self.outputs)
        else:
            self.pred = tf.nn.softmax(self.outputs)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars + self.history_vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None, load_history=False):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        if not load_history:
            saver = tf.train.Saver(self.vars)
        else:
            saver = tf.train.Saver(self.vars + self.history_vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, L, preprocess, placeholders, 
                 features, nbr_features, adj, cvd,
                 **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.L             = L
        self.preprocess    = preprocess
        self.placeholders  = placeholders
        self.sparse_input  = not isinstance(features, np.ndarray)

        self.input_dim = features.shape[1]
        self_dim       = 0 if FLAGS.normalization=='gcn' else self.input_dim
        if preprocess and FLAGS.pp_nbr:
            self_features  = features[:,:self_dim]
            stacker        = (lambda x: sp.hstack(x).tocsr()) if self.sparse_input else np.hstack
            self.features = stacker((self_features, nbr_features))
        else:
            self.features = features

        self.adj = adj
        self.cvd = cvd
        self.build()
        self.init_counts()

    def init(self, sess):
        pass

    def _preprocess(self):
        if self.preprocess:
            self.L       -= 1

        self.agg0_dim       = FLAGS.hidden1 if self.preprocess else self.input_dim


    def _build(self):
        # Aggregate
        fields = self.placeholders['fields']
        adjs   = self.placeholders['adj']
        dim_s  = 1 if FLAGS.normalization=='gcn' else 2
        cnt    = 0

        self.layer_comp = []

        if self.preprocess:
            for l in range(FLAGS.num_fc_layers):
                input_dim = self.input_dim*dim_s if l==0 else FLAGS.hidden1
                sparse_inputs = self.sparse_mm if l==0 else False
                last_layer = self.L==0 and l+1==FLAGS.num_fc_layers
                output_dim = self.output_dim if last_layer else FLAGS.hidden1
                act        = (lambda x: x)   if last_layer else tf.nn.relu
                layer_norm = False           if last_layer else FLAGS.layer_norm
                if FLAGS.det_dropout:
                    self.layers.append(DetDropoutFC(keep_prob=1-self.placeholders['dropout'],
                                             input_dim=input_dim,
                                             output_dim=FLAGS.hidden1,
                                             placeholders=self.placeholders,
                                             logging=self.logging,
                                             sparse_inputs=sparse_inputs,
                                             name='dense%d'%cnt, norm=FLAGS.layer_norm))
                elif self.cvd:
                    self.layers.append(AugmentedDropoutDense(keep_prob=1-self.placeholders['dropout'],
                                             input_dim=input_dim,
                                             output_dim=FLAGS.hidden1,
                                             logging=self.logging,
                                             sparse_inputs=sparse_inputs,
                                             name='dense%d'%cnt, norm=FLAGS.layer_norm))
                else:
                    self.layers.append(Dropout(1-self.placeholders['dropout'], self.cvd))
                    self.layers.append(Dense(input_dim=input_dim,
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=act,
                                             logging=self.logging,
                                             sparse_inputs=sparse_inputs,
                                             name='dense%d'%cnt, norm=layer_norm))
                self.layer_comp.append((input_dim*FLAGS.hidden1, 0))
                cnt += 1

        for l in range(self.L):
            self.layers.append(self.aggregators[l])
            for l2 in range(FLAGS.num_fc_layers):
                dim        = self.agg0_dim if l==0 else FLAGS.hidden1
                input_dim  = dim*dim_s if l2==0 else FLAGS.hidden1
                last_layer = l2+1==FLAGS.num_fc_layers and l+1==self.L
                output_dim = self.output_dim if last_layer else FLAGS.hidden1
                act        = (lambda x: x)   if last_layer else tf.nn.relu 
                layer_norm = False           if last_layer else FLAGS.layer_norm

                if FLAGS.det_dropout and l+1 != self.L:
                    self.layers.append(DetDropoutFC(keep_prob=1-self.placeholders['dropout'],
                                             input_dim=input_dim,
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             logging=self.logging,
                                             name='dense%d'%cnt, norm=layer_norm))
                elif self.cvd and l+1 != self.L:
                    self.layers.append(AugmentedDropoutDense(keep_prob=1-self.placeholders['dropout'],
                                             input_dim=input_dim,
                                             output_dim=output_dim,
                                             logging=self.logging,
                                             name='dense%d'%cnt, norm=layer_norm))
                else:
                    if not FLAGS.reverse:
                        self.layers.append(Dropout(1-self.placeholders['dropout'], self.cvd))
                    self.layers.append(Dense(input_dim=input_dim,
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=act,
                                             logging=self.logging,
                                             name='dense%d'%cnt, norm=layer_norm))
                    if FLAGS.reverse and not last_layer:
                        self.layers.append(Dropout(1-self.placeholders['dropout'], self.cvd))
                self.layer_comp.append((input_dim*output_dim, l+1))
                cnt += 1

    def init_counts(self):
        self.run_t = 0
        self.g_t   = 0
        self.g_ops = 0
        self.nn_ops = 0
        self.field_sizes = np.zeros(self.L+1)
        self.adj_sizes   = np.zeros(self.L)
        self.fadj_sizes  = np.zeros(self.L)
        self.amt_data = 0
