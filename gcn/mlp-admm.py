import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
from optimizers import *
from utils import hinge_loss
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist.train mnist.test mnist.validation

N = mnist.train.images.shape[0]
D = mnist.train.images.shape[1]
C = mnist.train.labels.shape[1]
batch_size = 100
hidden_size = 100
L = 2
learning_rate = 0.01
epoches = 500
admm_epoches = 500
decay = 1e-3
init_scale = 1e-4
update_multiplier = 50

z_penalty = 2
a_penalty = 5

# Plain MLP
x = tf.placeholder(shape=(None, D), dtype=tf.float32, name='x')
y = tf.placeholder(shape=(None, C), dtype=tf.float32, name='y')
y_int = tf.argmax(y, 1)
h = x
with tf.variable_scope('NN'):
    for i in range(L-1):
        h = layers.fully_connected(h, hidden_size, biases_initializer=None)
    pred = layers.fully_connected(h, C, activation_fn=None, biases_initializer=None)

# weight decay
weight_decay = 0
weights = tf.trainable_variables()
for i in weights:
    print(i.name, i.get_shape())
    if i.name.find('weight') != -1:
        weight_decay += tf.nn.l2_loss(i) * decay

# one vs rest hinge loss
bp_loss = tf.reduce_sum(hinge_loss(y, pred)) + weight_decay
accuracy = tf.nn.in_top_k(predictions=pred, targets=y_int, k=1)
accuracy = tf.reduce_mean(tf.cast(accuracy, dtype=tf.float32))

# ---- ADMM graph ----
As = [x]
Zs = []
admm_penalty = 0.0

optimizer_ops = []

with tf.variable_scope('NN', reuse=True):
    for i in range(L-1):
        # A[i] -> Z[i] (W[i])
        z_val = layers.fully_connected(As[-1], hidden_size, biases_initializer=None)
        Zs.append(tf.Variable(init_scale*tf.random_normal(shape=(N, hidden_size)), trainable=False, name='Z_{}'.format(i)))
        admm_penalty += z_penalty * tf.nn.l2_loss(z_val - Zs[-1])

        # Z[i] -> A[i+1]
        a_val = tf.nn.relu(Zs[-1])
        As.append(tf.Variable(init_scale*tf.random_normal(shape=(N, hidden_size)), trainable=False, name='A_{}'.format(i)))
        admm_penalty += a_penalty * tf.nn.l2_loss(a_val - As[-1])

    z_val = layers.fully_connected(As[-1], C, activation_fn=None, biases_initializer=False)
    Zs.append(tf.Variable(init_scale*tf.random_normal(shape=(N, C))))
    admm_penalty += z_penalty * tf.nn.l2_loss(z_val - Zs[-1])
    multiplier = tf.Variable(tf.zeros(shape=(N, C)), trainable=False, name='multiplier')

    for i in range(L):
        # Ops to optimize W
        optimizer_ops.append(('optimize_w_{}'.format(i), 
                             optimize_w(weights[i], Zs[i], As[i], decay)))

        if i+1 < L:
            # Ops to optimize Z
            z_val = tf.matmul(As[i], weights[i])
            optimizer_ops.append(('optimize_z_{}'.format(i),
                                 optimize_z(Zs[i], As[i+1], z_val, 
                                            a_penalty, z_penalty)))

            # Ops to optimize A
            a_val = tf.nn.relu(Zs[i])
            optimizer_ops.append(('optimize_a_{}'.format(i+1),
                                 optimize_a(As[i+1], a_val, 
                                            Zs[i+1], weights[i+1], a_penalty, z_penalty)))
        else:
            # Op to optimize last Z
            z_val = tf.matmul(As[i], weights[i])
            optimizer_ops.append(('optimize_last_z',
                optimize_last_z(Zs[i], y, multiplier, z_val, z_penalty)))

            # Op to update multiplier
            update_multiplier_op = optimize_multiplier(multiplier, Zs[-1], z_val, z_penalty)
    

admm_clf_loss = tf.reduce_sum(hinge_loss(y, Zs[-1]))
admm_mul_loss = tf.reduce_sum(multiplier * Zs[-1])
admm_loss = admm_penalty + admm_clf_loss + admm_mul_loss

# print('Optimizers:')
# for name, _ in optimizer_ops:
#     print(name)

def train_bp():
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.95)
    bp_train = optimizer.minimize(bp_loss)
    
    # Train with BackPropgation
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches+1):
            _, loss, acc, decay = sess.run([bp_train, bp_loss, accuracy, weight_decay],
                    feed_dict={x: mnist.train.images, y: mnist.train.labels})
            test_acc = sess.run(accuracy, 
                    feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print('Epoch {}, loss = {:.4f}, decay = {:.4f}, training accuracy = {:.4f}, testing accuracy = {:.4f}'
                    .format(epoch, loss, decay, acc, test_acc))

def train_admm():
    def evaluate(epoch):
        # Evaluate bp loss and admm loss, training accuracy
        bl, acl, aml, al, train_acc = sess.run([bp_loss, admm_clf_loss, admm_mul_loss, admm_loss, accuracy],
                feed_dict={x: mnist.train.images, y: mnist.train.labels})
        test_bl, test_acc = sess.run([bp_loss, accuracy],
                feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Epoch {}, BP loss = {:.4f}, {:.4f} ADMM CLF loss = {:.4f} Mul loss = {:.4f} Loss = {:.4f} Train acc = {:.4f} Test acc = {:.4f}'
                .format(epoch, bl, test_bl, acl, aml, al, train_acc, test_acc))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        evaluate(0)
        for epoch in range(1, admm_epoches+1):
            # Run the optimizers
            for name, op in optimizer_ops:
                sess.run(op, feed_dict={x: mnist.train.images, y: mnist.train.labels})
                if epoch < 5:
                    print(name)
                    evaluate(epoch)
            if epoch % update_multiplier == 0:
                sess.run(update_multiplier_op)
            if epoch >= 5:
                evaluate(epoch)

# train_bp()
train_admm()
