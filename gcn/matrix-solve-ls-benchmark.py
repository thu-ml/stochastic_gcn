import tensorflow as tf
import time

M = 60000
N = 784

tf.set_random_seed(1)

flops = M*N*N * 2
chol_flops = M*N*N*4 + N*N*N*2
reg = 1
A = tf.Variable(tf.random_uniform((M, N)))
B = tf.Variable(tf.random_uniform((M, N)))
true_X = tf.Variable(tf.random_uniform((N, N)))
assign_B = tf.assign(B, tf.matmul(A, true_X) + 10 * tf.random_normal((M, N)))

def least_squares(A, B, reg):
    D = int(A.get_shape()[1])
    ATA = tf.matmul(A, A, transpose_a=True) + tf.eye(D) * reg
    ATB = tf.matmul(A, B, transpose_a=True)
    return tf.cholesky_solve(tf.cholesky(ATA), ATB)

C = tf.reduce_sum(tf.matmul(A, B, transpose_a=True))
X1 = tf.matrix_solve_ls(A, B, reg)
X1_reg = tf.nn.l2_loss(X1) * reg
X1_loss = tf.nn.l2_loss(B-tf.matmul(A, X1))
X1_cost = X1_reg + X1_loss
X2 = least_squares(A, B, reg)
X2_reg = tf.nn.l2_loss(X2) * reg
X2_loss = tf.nn.l2_loss(B-tf.matmul(A, X2))
X2_cost = X2_reg + X2_loss


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(assign_B)

    for i in range(5):
        t = -time.time()
        print(sess.run(C))
        t += time.time()
        # print(flops)
        print('{} s, {} GFlops'.format(t, flops/t/1024/1024/1024))

    for i in range(5):
        t = -time.time()
        print(sess.run([X1_cost, X1_loss, X1_reg]))
        t += time.time()
        # print(chol_flops)
        print('{} s, {} GFlops'.format(t, chol_flops/t/1024/1024/1024))

    for i in range(5):
        t = -time.time()
        print(sess.run([X2_cost, X2_loss, X2_reg]))
        t += time.time()
        # print(chol_flops)
        print('{} s, {} GFlops'.format(t, chol_flops/t/1024/1024/1024))
