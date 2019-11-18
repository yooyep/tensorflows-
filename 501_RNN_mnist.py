# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:57:53 2019

@author: 13115
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)
np.random.seed(1)
tf.reset_default_graph() #重置图表，避免变量冲突
#%%
#--------------- 导入数据 ---------------
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#--------------- 定义网络结构 ---------------
batch_size=50
input_size =28 #输入图片的一行
time_step = 28 #序列的长度
n_class = 10 #分类个数
n_hidden_layers = 64

xs = tf.placeholder(tf.float32,[None,input_size*time_step])
ys = tf.placeholder(tf.float32,[None,n_class])
images = tf.reshape(xs,[-1,time_step,input_size])

rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_layers) #初始化一个lstm_cell
outputs,(h_c,h_n) = tf.nn.dynamic_rnn(
            rnn_cell, #选择的lstm_cell
            images, #输入
            initial_state=None, #输出hidden的state
            dtype = tf.float32,
            time_major = False, # False: (batch, time step, input); True: (time step, batch, input)
            )
output = tf.layers.dense(outputs[:, -1, :], 10)  # output based on the last output step

##损失函数
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=output)

##训练步骤
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1),)[1]

#--------------- 初始化变量 ---------------
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

#%%
#--------------- 训练 ---------------
list_loss = []
list_accu = []
#argmax(y,1) 依次比较各行，取各行最大元素 所在的index
def compute_accuracy(v_xs , v_y_correct):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(ys,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy1, feed_dict={xs: v_xs, ys: v_y_correct})
    return result

for i in range(300):
    batch_xs , batch_ys = mnist.train.next_batch(batch_size)
    _,loss_ = sess.run([train_step,loss],feed_dict={xs:batch_xs , ys:batch_ys})
    if i%50 == 0:
        accuracy_ = compute_accuracy(mnist.test.images,mnist.test.labels)
        list_loss.append(loss_)
        list_accu.append(accuracy_)
        print('loss: %s' %loss_ , '| accuracy: %s' %accuracy_)






