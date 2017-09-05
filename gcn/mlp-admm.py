import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
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
decay = 1e-4

# Plain MLP
x = tf.placeholder(shape=(None, D), dtype=tf.float32)
y = tf.placeholder(shape=(None, C), dtype=tf.int32)
y_int = tf.argmax(y, 1)
h = x
with tf.variable_scope('NN'):
    for i in range(L-1):
        h = layers.fully_connected(h, hidden_size)
    pred = layers.fully_connected(h, C, activation_fn=None)

# weight decay
weight_decay = 0
params = tf.trainable_variables()
for i in params:
    print(i.name, i.get_shape())
    if i.name.find('weight') != -1:
        weight_decay += tf.nn.l2_loss(i) * decay

# one vs rest hinge loss
bp_loss = tf.losses.hinge_loss(labels=y, logits=pred, reduction=tf.losses.Reduction.NONE)
bp_loss = tf.reduce_mean(bp_loss) + weight_decay
accuracy = tf.nn.in_top_k(predictions=pred, targets=y_int, k=1)
accuracy = tf.reduce_mean(tf.cast(accuracy, dtype=tf.float32))

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
