#%% 版本二 固定十个点，进行logistic回归

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#--------------- 导入数据 ---------------
x_train = [[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 5.0], [1.0, 3.0], [4.0, 2.0], [7.0, 3.0], [4.0, 5.0], [11.0, 3.0],
           [8.0, 7.0]]
y_train = [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_train = np.array(y_train)

x = np.array(x_train)
plt.scatter(x[:,0] , x[:,1] , marker='o' , c=y_train, edgecolor='k')

#--------------- 模型 训练 ---------------
theta = tf.Variable(tf.zeros([2, 1]))
theta0 = tf.Variable(tf.zeros([1, 1]))
y = 1 / (1 + tf.exp(-tf.matmul(x_train, theta) + theta0))

loss = tf.reduce_mean(- y_train.reshape(-1, 1) * tf.log(y) - (1 - y_train.reshape(-1, 1)) * tf.log(1 - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(1000):
    _ , loss1 = sess.run([train,loss])
    if step%50 == 0:
        print("i:%d loss:%.2f" %(step,loss1))
print(step, sess.run(theta).flatten(), sess.run(theta0).flatten())

w,b=sess.run([theta,theta0])
xx=np.linspace(0,10,100)
yy=[]
for i in xx:
    yy.append(float((i* -w[0]-b) / w[1]))
    #yy.append(float((i* -w[1]-b) / w[0]))
plt.plot(xx,yy)



