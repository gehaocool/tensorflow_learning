#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:04:55 2017

@author: gehao
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# add layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #print(Weights)
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)
    #print(biases.shape)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    #print(Wx_plus_b.shape)
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    #print (output.shape)
    return output


# 1.Make up some Data for training
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 2.Define palceholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.Add nueral layers: hidden layers and prediction layer
# add hidden layer, input: xs, 10 neurons
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer, input: l1, output 1 result in prediction layer
prediction = add_layer(l1, 10, 1, activation_function=None)


# 4.Define the error between prediction and real data
loss = tf.reduce_mean(tf.square(ys-prediction))


# 5.Chose a optimizer to minimize loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# Initialize all the variables
init = tf.global_variables_initializer()
sess = tf.Session()
# No calculation till sess.run
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# Iteration 1000 to train
for i in range(1000):
    # train train_step and loss
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        
        plt.pause(0.5)

plt.ioff()