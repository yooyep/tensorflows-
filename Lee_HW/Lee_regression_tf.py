# -*- coding: utf-8 -*-
#李宏毅老师讲完regression的demo，展示w b参数的回归问题
#https://www.bilibili.com/video/av10590361?p=4

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# matplotlib没有中文字体，动态解决
from pylab import mpl
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#--------------- 数据 及展示 ---------------
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
x_data = np.asarray(x_data)[:,np.newaxis]
y_data = np.asarray(y_data)[:,np.newaxis]

xs = tf.placeholder(tf.float32,shape=[None,1])
ys = tf.placeholder(tf.float32,shape=[None,1])

weight = tf.Variable(-4.0)
#weight = tf.Variable(tf.truncated_normal([1]))
#biaes = tf.Variable(tf.zeros([1]))
biaes = tf.Variable(-140.0)
y = weight*x_data+biaes
loss = tf.reduce_sum(tf.square(y-y_data))

lr=1e-6
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

#--------------- 训练 ---------------
w_history = []
b_history = []
import time
start_time = time.time()
for i in range(100000):
    sess.run(train_op)
    if i%1000 ==0:
        w_,b_,loss_ = sess.run([weight,biaes,loss])
        w_history.append(w_)
        b_history.append(b_)
        print('i:%d w:%.2f b:%.2f loss:%.2f' %(i,w_,b_,loss_))
end_time = time.time()
print("大约需要时间：",end_time-start_time)
        
#%%
##显示数据
#plt.plot(x_data,y_data,'*')
##最佳曲线
#xx=np.linspace(0,600,600,dtype=np.float32)
#w_star=2.67
#b_star=-188.4
#yy=w_star*xx+b_star
#plt.plot(xx,yy,'-r',label='最佳')
#plt.legend()
#plt.show()

#--------------- 画loss function的等高线图 ---------------
x = np.arange(-200,-100,1) #b的参数
y = np.arange(-5,5,0.1) #w的参数
Z = np.zeros([len(x),len(y)])
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b=x[i]
        w=y[j]
        # meshgrid吐出结果：行为w即y，列为x即b。对于同等来说，b变化对loss的影响小
        Z[j][i]=np.mean(np.square(y_data - (w*x_data+b)))

#--------------- 画图 ---------------
contour = plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))  # 填充等高线
plt.clabel(contour,fontsize=10,colors='k')

plt.plot([-188.4], [2.67], 'x', ms=12, mew=3, color="orange")
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$')
plt.ylabel(r'$w$')
plt.title("线性回归 学习率=%s" %lr)
plt.show()


