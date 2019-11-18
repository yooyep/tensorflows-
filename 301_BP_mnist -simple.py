# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:18:23 2019

@author: 13115
"""
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

#%%
#--------------- 导入数据 ---------------
mnist = input_data.read_data_sets("mnist/", one_hot=True)

#%%
#--------------- 定义网络结构 ---------------
xs = tf.placeholder(dtype=tf.float32,shape=[None,784])
ys = tf.placeholder(tf.float32,[None,10])

l1 = tf.layers.dense(xs,784,activation=tf.nn.relu)
predict= tf.layers.dense(l1,10,activation=tf.nn.softmax)

##损失函数
loss = -tf.reduce_sum(ys*tf.log(predict))

##训练步骤
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#--------------- 初始化变量 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#%%
#--------------- 训练 ---------------
list_loss = []
list_accu = []
#argmax(y,1) 依次比较各行，取各行最大元素 所在的index
def compute_accuracy(v_xs , v_y_correct):
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_y_correct})
    return result


for i in range(300):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    _,loss_ = sess.run([train_step,loss],feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        accuracy_ = compute_accuracy(mnist.test.images,mnist.test.labels)
        list_loss.append(loss_)
        list_accu.append(accuracy_)
        print('正确率: %s' %accuracy_)
        print('i:%d' %i)

        