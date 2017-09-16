from gcn.layers import *
from gcn.metrics import *
from gcn.inits import *
from time import time

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            print('{} shape = {}'.format(layer.name, hidden.get_shape()))
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = [self.optimizer.minimize(self.loss)]
        with tf.control_dependencies(self.opt_op):
            for layer in self.layers:
                for ref, indices, updates in layer.post_updates:
                    self.opt_op.append(tf.scatter_update(ref, indices, updates))

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

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


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

        # Entropy error
        self.loss += FLAGS.unlabeled_weight * masked_entropy(self.outputs, 1-self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


# ----------------------------------------------------------
class FastGCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(FastGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), 
                                      tf.argmax(self.placeholders['labels'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu, 
                                            dropout=True, 
                                            sparse_inputs=True,
                                            logging=self.logging,
                                            support=self.placeholders['support_1']))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging,
                                            support=self.placeholders['support_2']))

    def predict(self):
        return tf.nn.softmax(self.outputs)
    

class VRGCN(Model):
    def __init__(self, placeholders, input_dim, num_data, vr, **kwargs):
        super(VRGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.num_data = num_data
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.vr = vr

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), 
                                      tf.argmax(self.placeholders['labels'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        if self.vr:
            self.layers.append(VarianceReductedAggregator(
                                     num_data=self.num_data,
                                     input_dim=FLAGS.hidden1,
                                     input_fields=self.placeholders['hidden_fields'],
                                     subsampled_support=self.placeholders['subsampled_support'],
                                     support=self.placeholders['support']))
        else:
            self.layers.append(PlainAggregator(
                                     subsampled_support=self.placeholders['subsampled_support']))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

# -------------------------------------------------------------------------------------------------------------
class MACGCN(Model):
    def __init__(self, placeholders, input_dim, input_n, **kwargs):
        super(MACGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.acs = [self.inputs]
        self.decoupled_layers = []
        self.mac_loss = 0
        self.input_n = input_n

        self.build()

        params = tf.trainable_variables()
        for i in params:
            print(i.name, i.get_shape())

    def _mac_loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.mac_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Reconstruction loss
        for i in range(1, len(self.acs)):
            self.mac_loss += self.placeholders['lambda'] * tf.nn.l2_loss(self.acs[i] - self.mac_acts[i-1])

        # Cross entropy error
        self.mac_loss += masked_softmax_cross_entropy(self.mac_outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])


    def _prediction_difference(self):
        diff = tf.nn.softmax(self.outputs) - tf.nn.softmax(self.mac_outputs)
        self.pred_diff = tf.reduce_mean(tf.reduce_sum(tf.abs(diff), -1))


    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        # Create variables for hidden layers
        self.acs.append(zeros(shape=(self.input_n, FLAGS.hidden1)))
        
        # Directed prediction
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.mac_acts = []
        for i, layer in enumerate(self.layers):
            hidden = layer(self.acs[i])
            self.mac_acts.append(hidden)
        self.mac_outputs = self.mac_acts[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._mac_loss()
        self._prediction_difference()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.mac_loss)


    def predict(self):
        raise NotImplementedError
        # return tf.nn.softmax(self.outputs)

# -----------------------------------------------
class GraphSAGE(Model):
    def __init__(self, placeholders, features, train_adj=None, test_adj=None, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)

        if train_adj is not None:
            # Preprocess first aggregation
            print('Preprocessing first aggregation')
            start_t = time()

            train_features = train_adj.dot(features)
            test_features  = test_adj.dot(features)

            self.train_inputs = tf.Variable(train_features, trainable=False)
            self.test_inputs  = tf.Variable(test_features,  trainable=False)
            self.self_inputs  = tf.Variable(features,       trainable=False)
            self.nbr_inputs   = tf.cond(placeholders['is_training'], 
                                        lambda: self.train_inputs, 
                                        lambda: self.test_inputs)
            self.inputs       = tf.concat((self.self_inputs, self.nbr_inputs), -1)
            self.input_dim    = features.shape[1]
            self.preprocess   = True

            print('Finished in {} seconds.'.format(time() - start_t))
        else:
            self.inputs     = tf.Variable(features, trainable=False)
            self.input_dim  = features.shape[1]
            self.preprocess = False

        self.num_data = features.shape[0]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.outputs, labels=self.placeholders['labels']))

    def _accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), 
                                      tf.argmax(self.placeholders['labels'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _build(self):
        # Aggregate
        fields = self.placeholders['fields']
        adjs   = self.placeholders['adj']

        if not self.preprocess:
            self.layers.append(GatherAggregator(fields[0], name='gather'))
            self.layers.append(PlainAggregator(adjs[0], fields[0], fields[1],
                                               name='agg1'))
        else:
            self.layers.append(GatherAggregator(fields[1], name='gather'))

        self.layers.append(Dense(input_dim=self.input_dim*2,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging,
                                 name='dense1'))

        self.layers.append(PlainAggregator(adjs[1], fields[1], fields[2], 
                                           name='agg2'))
        self.layers.append(Dense(input_dim=FLAGS.hidden1*2,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging,
                                 name='dense2'))

    def predict(self):
        return tf.nn.softmax(self.outputs)
