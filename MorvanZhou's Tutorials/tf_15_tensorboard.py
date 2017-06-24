# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:38:18 2017

@author: Opeth
"""

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    '''
    Add one more layer and return the output of this layer
    '''
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# make up some data
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise  = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')


# add hidden layers
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between real data and prediction
with tf.name_scope('loss'):
    loss = tf.reduce_mean(ys - prediction)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)
'''
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
'''