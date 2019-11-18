# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:18:23 2019

@author: 13115
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

#%%
#--------------- 导入数据 ---------------
x_data = np.linspace(-1,1,100)[:,np.newaxis]
noise = np.random.normal(0,0.1,x_data.shape).astype(np.float32)
y_data = x_data**2+noise

xs = tf.placeholder(dtype=tf.float32,shape=[None,1])
ys = tf.placeholder(tf.float32,[None,1])
#%%
#--------------- 定义网络结构 ---------------
l1 = tf.layers.dense(xs,10,activation=tf.nn.relu)
predict= tf.layers.dense(l1,1,activation=None)

##损失函数
loss = tf.reduce_mean(tf.square(predict-ys))

##训练步骤
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#--------------- 初始化变量 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#%%
#--------------- 训练 ---------------
loss_num=[]
plt.ion()
for i in range(300):
    _,loss_,pred = sess.run([train_step,loss,predict],feed_dict={xs:x_data,ys:y_data})
    loss_num.append(loss_)
    if i%5 == 0:
        plt.cla()
        plt.scatter(x_data,y_data)
        plt.plot(x_data,pred,'r-',lw=3)
        plt.text(0,1,'loss:%.4f' %loss_,fontdict={'size':20,'color':'red'})
        plt.pause(0.01)
plt.ioff()
plt.show()

