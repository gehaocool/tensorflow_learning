#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:05:46 2017

@author: gehao
"""

import tensorflow as tf

state = tf.Variable(0, name='conter')

add_value = tf.add(state, 1)
update = tf.assign(state, add_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:  
    sess.run(init)
    for i in range(4):
        # sess.run(add_value)
        sess.run(update)
        print( sess.run(state))