# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.01,
                 learning_rate_decay_factor=0.99):
        self.x_ = tf.placeholder(tf.float32, [None, (35 + 18) * 24 * 3 * 6])
        #self.x2_ = tf.placeholder(tf.float32, [None, 840 * 5, 6])
        self.y_ = tf.placeholder(tf.float32, [None, 318])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = learning_rate
        self.W_1 = tf.Variable(tf.random_normal([22896, 500]), trainable=is_train)
        self.W_2 = tf.Variable(tf.random_normal([500, 318]), trainable=is_train)
        self.b_1 = tf.Variable(tf.random_normal([500]),trainable=is_train)
        self.b_2 = tf.Variable(tf.random_normal([318]),trainable=is_train)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        fc1 = tf.add(tf.matmul(self.x_, self.W_1), self.b_1)
        fc1 = batch_normalization_layer(fc1, 500, is_train)
        fc1 = tf.nn.relu(fc1)
        self.y_predict = tf.add(tf.matmul(fc1, self.W_2), self.b_2)
        
        self.loss = tf.reduce_mean(tf.square(self.y_predict - self.y_))  
        
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.loss)  # Use Adam Optimizer

        # self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
        #                             max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1., shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, shape, isTrain=True):

    input_mean, input_var = tf.nn.moments(inputs, axes = [0])

    beta = tf.Variable(tf.zeros([shape]), name='beta', trainable=isTrain)
    gamma = tf.Variable(tf.ones([shape]), name='gamma', trainable=isTrain)
    gb_mean = tf.Variable(tf.zeros_like(input_mean), name='gb_mean', trainable=False)
    gb_var = tf.Variable(tf.ones_like(input_mean), name='gb_var', trainable=False)

    if isTrain:
        
        update_mean = tf.assign(gb_mean, gb_mean * 0.999 + input_mean * 0.001)
        update_var = tf.assign(gb_var, gb_var * 0.999 + input_var * 0.001)
        with tf.control_dependencies([update_mean, update_var]):
            return tf.nn.batch_normalization(inputs, input_mean, input_var, beta, gamma, 1e-5)
    
    else:
       return tf.nn.batch_normalization(inputs, gb_mean, gb_var, beta, gamma, 1e-5) 
    return inputs



