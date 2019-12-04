# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:34:03 2019

@author: 13115
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import time

from pylab import mpl
# matplotlib没有中文字体，动态解决
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题



mnist = input_data.read_data_sets('mnist', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#%% 权重初始化 与卷积池化层
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#%% 结构建立
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  

x_image = tf.reshape(x, [-1,28,28,1])  
  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  
  

#第二层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#%% 训练
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

lr=1e-4
#train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(lr,0.9).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(lr,0.9,0.999,epsilon=1e-08).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

list_loss = []
list_accu = []
start = time.time()
for i in range(20000):
  batch = mnist.train.next_batch(50)
  _, loss_ = sess.run([train_step, cross_entropy], {x: batch[0], y_: batch[1], keep_prob: 0.5})
#  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  
  if i%50 == 0:
    batch = mnist.test.next_batch(500)
    test_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    
    list_loss.append(loss_)
    list_accu.append(test_accuracy)
    print("step %d, test accuracy %g"%(i, test_accuracy))
  
  
end = time.time()
print('time: {}'.format(end-start))

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#设置坐标轴刻度
my_x_ticks = np.arange(0, 20, 1)

plt.figure()
plt.plot(list_loss,'*-')
plt.xticks(my_x_ticks)
plt.xlabel(r'迭代次数')
plt.ylabel(r'损失函数') #横纵坐标轴
plt.title('Adam参数优化--损失函数值') #标题
plt.show()

plt.figure()
plt.plot(list_accu,'*-')
plt.xticks(my_x_ticks)
plt.xlabel(r'迭代次数')
plt.ylabel(r'正确率') #横纵坐标轴
plt.title('Adam参数优化--正确率') #标题
plt.show()

