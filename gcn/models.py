from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time
import scipy.sparse as sp
from gcn.utils import sparse_to_tuple, dropout, sparse_dropout
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'multitask'}
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
        self.sparse_input   = not isinstance(self.features, np.ndarray)
        self.sparse_mm      = self.sparse_input
        self.inputs_ph      = self.get_ph('input')

        self.inputs         = self.inputs_ph
        if self.sparse_input:
            self.inputs     = tf.sparse_reorder(self.inputs)
        if not self.preprocess and self.sparse_input:
            print('Warning: we do not support sparse input without pre-processing. Converting to dense...')
            self.inputs     = tf.sparse_to_dense(self.inputs.indices, 
                                                 self.inputs.dense_shape,
                                                 self.inputs.values)
            self.sparse_mm  = False

        self.num_data       = self.features.shape[0]

        self.output_dim     = self.placeholders['labels'].get_shape().as_list()[1]
        self.placeholders   = self.placeholders

        self._preprocess()

        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build_history()
            self._build_aggregators()
            self._build()

        # Build sequential layer model
        self.average_get_ops    = []
        self.average_phs        = []
        self.average_update_ops = []
        self.average_values     = []

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            print('{} shape = {}'.format(layer.name, hidden.get_shape()))
            self.activations.append(hidden)

            # Polyak-ops
            if FLAGS.polyak_decay > 0:
                for var in layer.vars:
                    print(var.name, var.get_shape())
                    self.average_get_ops.append(var)
                    self.average_phs.append(tf.placeholder(tf.float32))
                    self.average_update_ops.append(tf.assign(var, self.average_phs[-1]))
                    self.average_values.append(np.zeros(var.get_shape(), np.float32))

        self.outputs = self.activations[-1]
        self._predict()

        with tf.variable_scope(self.name):
            #self.activations.append(self.inputs)  TODO: bug
            for layer in self.layers:
                if hasattr(layer, 'new_history'):
                    self.history_ops.append(layer.new_history)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = variables
        print('Model variables')
        for k in self.vars:
            print(k.name, k.get_shape())

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = [self.optimizer.minimize(self.loss)]
        self.grads  = tf.gradients(self.loss, self.vars)

    def _predict(self):
        if self.multitask:
            self.pred = tf.nn.sigmoid(self.outputs)
        else:
            self.pred = tf.nn.softmax(self.outputs)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, data_per_fold, L, preprocess, placeholders, 
                 features, features1, adj, 
                 **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.data_per_fold = data_per_fold
        self.L             = L
        self.preprocess    = preprocess
        self.placeholders  = placeholders
        self.features      = features
        self.features1     = features1
        self.adj           = adj

        self.build()

    def _preprocess(self):
        self.input_dim = self.features.shape[1]
        self_dim       = 0 if FLAGS.normalization=='gcn' else self.input_dim

        if self.preprocess:
            self_features = self.features[:,:self_dim]
            #nbr_features  = self.adj.dot(self.features)
            nbr_features  = self.features1
            self.L       -= 1
        else:
            self_features = self.features
            nbr_features  = np.zeros((self.num_data, 0), dtype=np.float32)

        num_training_data = self.data_per_fold*FLAGS.num_reps
        if self.sparse_input:
            self.features = sp.hstack((self_features, nbr_features)).tocsr()
            if FLAGS.det_dropout:
                self.features[:num_training_data] = sparse_dropout(self.features[:num_training_data], 
                                                                   1-FLAGS.dropout)
        else:
            self.features = np.hstack((self_features, nbr_features))
            if FLAGS.det_dropout:
                self.features[:num_training_data] = dropout(self.features[:num_training_data], 1-FLAGS.dropout)

        self.agg0_dim       = FLAGS.hidden1 if self.preprocess else self.input_dim


    def _build(self):
        # Aggregate
        fields = self.placeholders['fields']
        adjs   = self.placeholders['adj']
        dim_s  = 1 if FLAGS.normalization=='gcn' else 2
        alpha  = self.placeholders['alpha']
        cnt    = 0

        if self.preprocess:
            if not FLAGS.det_dropout:
                self.layers.append(Dropout(1-self.placeholders['dropout'],
                                           self.placeholders['is_training']))
            for l in range(FLAGS.num_fc_layers):
                input_dim = self.input_dim*dim_s if l==0 else FLAGS.hidden1
                sparse_inputs = self.sparse_mm if l==0 else False
                self.layers.append(Dense(input_dim=input_dim,
                                         output_dim=FLAGS.hidden1,
                                         placeholders=self.placeholders,
                                         act=tf.nn.relu,
                                         logging=self.logging,
                                         sparse_inputs=sparse_inputs,
                                         name='dense%d'%cnt, norm=FLAGS.layer_norm))
                cnt += 1

        for l in range(self.L):
            self.layers.append(self.aggregators[l])
            if not FLAGS.det_dropout:
                self.layers.append(Dropout(1-self.placeholders['dropout'],
                                           self.placeholders['is_training']))
            for l2 in range(FLAGS.num_fc_layers):
                dim        = self.agg0_dim if l==0 else FLAGS.hidden1
                input_dim  = dim*dim_s if l2==0 else FLAGS.hidden1
                last_layer = l2+1==FLAGS.num_fc_layers and l+1==self.L
                output_dim = self.output_dim if last_layer else FLAGS.hidden1
                act        = (lambda x: x)   if last_layer else tf.nn.relu 
                layer_norm = False           if last_layer else FLAGS.layer_norm

                self.layers.append(Dense(input_dim=input_dim,
                                         output_dim=output_dim,
                                         placeholders=self.placeholders,
                                         act=act,
                                         logging=self.logging,
                                         name='dense%d'%cnt, norm=layer_norm))
                cnt += 1


