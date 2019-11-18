import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#简单回归拟合 y=x^2-0.5

# 建造第一个神经网络 tf.layers.dense方式================================
#文章链接：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/3-2-create-NN/

## 导入数据
x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
#生成mu=0，sigma=0.05的正态分布 噪声
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = x_data**2-0.5+noise

## 使用placeholder占位符，代替数据的输入
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

## 定义网络结构
#激活函数采用tf.nn.relu,小于0的部分置为0，大于0不变
#l1 = add_layer(xs,1,10,activation_funciton=tf.nn.relu)
l1 = tf.layers.dense(xs,10,activation=tf.nn.relu)
#输出层 输入为隐含层输出，输出为预测值
#prediction = add_layer(l1,10,1,activation_funciton=None)
prediction = tf.layers.dense(l1,1,activation=None)
#loss的 sum暂时存疑
#loss = tf.reduce_mean(tf.reduce_sum(tf.square( ys-prediction ) , reduction_indices=[1]))
loss = tf.reduce_mean(tf.square(ys-prediction))

#以0.1的步长 最小化loss，进行梯度下降训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion() #用于连续显示，单次运行注释，全局不要注释
for i in range(200):
    _,l,pred = sess.run([train_step,loss,prediction],feed_dict={xs:x_data,ys:y_data})
    if i%5 == 0:
    
        plt.cla() #这一句话很重要，不然有很多根红线
        plt.scatter(x_data,y_data)
        plt.plot(x_data,pred,'r-',lw=5)
        plt.text(0,0.5,'loss:%.4f' %l,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()


# 建造第一个神经网络 add_layer方式 ================================
# 重点在于执行add_layer要传入参数   prediction_data = sess.run(prediction,feed_dict={xs:x_data})
'''
def add_layer(inputs,in_size,out_size,activation_funciton=None):
    Weights = tf.Variable(tf.random.normal([in_size,out_size]))
    #biase不推荐为0，因此+0.1
    biase = tf.Variable(tf.zeros([1,out_size])+0.1)
    
    Wx_plus_b = tf.matmul(inputs , Weights) + biase
    
    if activation_funciton==None:
        outputs = Wx_plus_b
    else:
        outputs = activation_funciton(Wx_plus_b)
        
    return outputs

## 导入数据  
x_data = np.linspace(-1,1,100)[:,np.newaxis]
noise = np.random.normal(0,0.1,size=x_data.shape)
y_data = np.power(x_data,2)+noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

## 定义网络结构
l1 = add_layer(xs,1,10,activation_funciton=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_funciton=None)
loss = tf.reduce_mean(tf.square(y_data-prediction))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#plt.ion()
for i in range(200):
     #这里只填充了xs，因为loss在写的时候没用到ys。下面同理
    sess.run(train_step,feed_dict={xs:x_data})
    if i%20 == 0:
        prediction_data = sess.run(prediction,feed_dict={xs:x_data})
        loss_data = sess.run(loss,feed_dict={xs:x_data})
        plt.cla() 
        plt.scatter(x_data,y_data)
        plt.plot(x_data , prediction_data , 'r-' , lw=5)
        plt.text(0,1,'loss:%.4f' %loss_data,fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
#plt.ioff()
plt.show()
'''   


