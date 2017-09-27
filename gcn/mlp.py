
class NeighbourMLP(Model):
    def __init__(self, L, placeholders, features, train_adj, test_adj, **kwargs):
        super(NeighbourMLP, self).__init__(**kwargs)

        self.L = L
        self.sparse_input = not isinstance(features, np.ndarray)
        self.inputs_ph    = self.get_ph('input')
        self.inputs       = self.inputs_ph

        # Preprocess aggregation
        print('Preprocessing aggregations')
        start_t = time()

        # Create all the features
        def _create_features(X, A):
            features = [X]
            print('Hops = {}'.format(FLAGS.num_hops))
            for i in range(FLAGS.num_hops):
                features.append(A.dot(features[-1]))
            return np.hstack(features)

        self.train_features = _create_features(features, train_adj)
        self.test_features  = _create_features(features, test_adj)
        self.input_dim      = self.train_features.shape[1]
        print('Finished in {} seconds.'.format(time() - start_t))

        self.num_data = features.shape[0]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, 
                                                beta1=FLAGS.beta1, beta2=FLAGS.beta2)

        self.build()


    def get_data(self, feed_dict, is_training):
        ids = feed_dict[self.placeholders['fields'][-1]]
        inputs = self.train_features[ids] if is_training else self.test_features[ids]
        if self.sparse_input:
            inputs = sparse_to_tuple(inputs)

        feed_dict[self.inputs_ph] = inputs


    def train_one_step(self, sess, feed_dict, is_training):
        self.get_data(feed_dict, is_training)

        # Run
        outs = sess.run([self.opt_op, self.loss, self.accuracy], 
                              feed_dict=feed_dict)

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
