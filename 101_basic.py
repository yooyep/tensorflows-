# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:24:18 2019

@author: 13115
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#创建constant、session ================================
'''
import tensorflow as tf
#1x2和2x1的矩阵
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2] , [1]])
product = tf.matmul(matrix1,matrix2)
#创建会话，执行op 
#方法1,显式调用close
sess= tf.Session()
result = sess.run(product)
print(result)
sess.close()

#方法2
with tf.Session() as sess:
    result1 = sess.run(product)
    print(result1)
      
'''

#variable 变量激活================================
'''
state = tf.Variable(0,name='count')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

#变量需要激活!!! --版本更新语句
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) #打印state，更新run(state)会执行state+1
'''


#placeholder暂存变量 ================================
'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)  

with tf.Session() as sess:
    #暂存的变量，采用feed_dict{}对应一个个placeholder
    print(sess.run(output,feed_dict={input1:[7],
                               input2:[2.0]}))
'''