# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:26:21 2017

@author: Opeth
"""

import tensorflow as tf

# 在 Tensor flow中需要定义placeholder的type，一般为float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))