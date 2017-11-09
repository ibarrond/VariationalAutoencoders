'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Generate Training Data

# number samples
N = 100
# dimension of the samples
P = 1
rs = np.random.RandomState(0)
train_X = rs.randn(N,P)
# draw N times one sample (n samples) from a binomial distribution
# logistic probability (i.e. we draw n points for the LR model)
train_Y = rs.binomial(1,sigmoid(np.dot(train_X,z_real)))

n_samples = train_X.shape[0]






## Log-density of a univariate Gaussian distribution
def log_norm_pdf(x, m=0.0, log_v=0.0):
    return - 0.5 * tf.log(2 * np.pi) - 0.5 * log_v - 0.5 * tf.square(x - m) / tf.exp(log_v)

## Kullback-Leibler divergence between multivariate Gaussian distributions q and p with diagonal covariance matrices
def DKL_gaussian(mq, log_vq, mp, log_vp):
    """
    KL[q || p]
    :param mq: vector of means for q
    :param log_vq: vector of log-variances for q
    :param mp: vector of means for p
    :param log_vp: vector of log-variances for p
    :return: KL divergence between q and p
    """
    log_vp = tf.reshape(log_vp, (-1, 1))
    return 0.5 * tf.reduce_sum(log_vp - log_vq + (tf.pow(mq - mp, 2) / tf.exp(log_vp)) + tf.exp(log_vq - log_vp) - 1)

## Log-Likelihood of the data with sampled weights
def log_likelihood(z_sample, mu, sigma):
    log_p = tf.reduce_sum(
        train_Y * tf.log(tf.sigmoid(tf.matmul(train_X,z_sample))) + \
        (1-train_Y) * tf.log(1-tf.sigmoid(tf.matmul(train_X,z_sample)))
    )
    return log_p

## Draw a tensor of standard normals
def get_normal_samples(ns, dim):
    """"
    :param ns: Number of samples
    :param dim: Dimension
    :return:
    """
    return tf.random_normal(shape=[ns, dim])

## Log-sum operation
def logsumexp(vals, dim=None):
    m = tf.reduce_max(vals, dim)
    if dim is None:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - m), dim))
    else:
        return m + tf.log(tf.reduce_sum(tf.exp(vals - tf.expand_dims(m, dim)), dim))

def get_prior_W(n_W):
    prior_mean_W = tf.zeros(n_W)
    log_prior_var_W = tf.zeros(n_W)
    return prior_mean_W, log_prior_var_W

def init_posterior_W(dim):
    mean_W = tf.Variable(tf.zeros(n_W), name="q_W")
    log_var_W = tf.Variable(tf.zeros(n_W), name="q_W")

    return mean_W, log_var_W

def sample_from_W(n_mc_samples, dim):
    W_from_q = []
    z = get_normal_samples(n_mc_samples, dim)
    W_from_q = tf.add(tf.multiply(z, tf.exp(self.log_var_W[i] / 2)), self.mean_W[i]))
    return W_from_q

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(P), name="weight")
#b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
#pred = tf.add(tf.multiply(X, W), b)
pred = tf.multiply(X, W)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
