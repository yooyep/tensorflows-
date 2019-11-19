#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#简单回归拟合 y=x^2================================
#建立tensorboard可视化神经网络
#注意：该代码不要在spyder运行，使用sublime运行，不然有很多层。

#cmd的操作
#1.cd进入该文件夹 
#2.运行tensorboard --logdir log 
#3.打开浏览器输入http://localhost:6006

tf.reset_default_graph() #重置图表，避免变量冲突
#生成数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.1,size=x_data.shape)
y_data = np.power(x_data,2)+noise

#--------------- 定义网络结构  --------------- 
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')  
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

with tf.name_scope('Net222'):
    l1 = tf.layers.dense(xs,10,tf.nn.relu,name='hidden_layer')
    predict = tf.layers.dense(l1,1,name='output_layer')
    #添加变化图表
    tf.summary.histogram('h_out',l1)
    tf.summary.histogram('pred',predict)
    
with tf.name_scope('loss'):
    loss = tf.losses.mean_squared_error(ys,predict)
    #画出loss的图
    tf.summary.scalar('loss', loss) 
    
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#--------------- 训练  --------------- 
#创建会话，初始化
sess= tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./log', sess.graph)
merge_op = tf.summary.merge_all()

for i in range(500):
     _,l,pred = sess.run([train_step,loss,predict] , feed_dict={xs:x_data,ys:y_data})
     if i%20==0:
         result=sess.run(merge_op,feed_dict={xs:x_data,ys:y_data})
         writer.add_summary(result,i) #20步一个点
#    _,l,pred,result = sess.run([train_step,loss,predict,merge_op] , feed_dict={xs:x_data,ys:y_data})
#    writer.add_summary(result,i)


#%%
#--------------- add_layer方式  ---------------
def add_layer(inputs,in_size,out_size,activation_funciton=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random.normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            #biase不推荐为0，因此+0.1
            biase = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs , Weights) + biase
        
        if activation_funciton==None:
            outputs = Wx_plus_b
        else:
            outputs = activation_funciton(Wx_plus_b)
            
        return outputs

#l1 = add_layer(xs,1,10,activation_funciton=tf.nn.relu)
#predict = add_layer(l1,10,1,activation_funciton=None)




