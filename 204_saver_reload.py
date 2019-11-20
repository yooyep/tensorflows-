#%%
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

tf.reset_default_graph()
#--------------- 导入数据 ---------------
x_data = np.linspace(-2,2,100)[:,np.newaxis]
noise = np.random.normal(0,0.1,size=x_data.shape).astype(np.float32)
y_data = x_data**2+noise

with open('data.pickle','wb') as f:
    pickle.dump([x_data,y_data],f)
f.close()

with open('data.pickle','rb') as f:
    x_data,y_data = pickle.load(f)

#--------------- 建立网络结构 ---------------
xs = tf.placeholder(tf.float32,shape=[None,1])
ys = tf.placeholder(tf.float32,shape=[None,1])
l1 = tf.layers.dense(xs,10,activation=tf.nn.relu,name='l1')
predict = tf.layers.dense(l1,1,activation=None)

loss = tf.reduce_sum(tf.square(predict-ys))

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#--------------- 训练并保存模型 ---------------
saver = tf.train.Saver()

is_train=False
if is_train:
    #训练
    for i in range(300):
        _,loss_,pred_ = sess.run([train_step,loss,predict],feed_dict={xs:x_data,ys:y_data})
        if i%50 == 0:
            print('step:%s' %i , ' | loss:%s' %loss_)
            
    #保存模型
    saver.save(sess,'params/model')
    #画图，展示最后的loss
    loss_,pred_ = sess.run([loss,predict],feed_dict={xs:x_data,ys:y_data})
    plt.subplot(121)
    plt.scatter(x_data,y_data)
    plt.plot(x_data,pred_,'r-')
    plt.text(-1,3.5,'save loss:%.4f' %loss_ , fontdict={'size':10,'color':'red'})
       
else:
    #--------------- 加载模型与参数 ---------------
    #model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,'params/model')
    
    #通过加载的参数 进行x_data的预测，并画图
    pred_,loss2= sess.run([predict,loss],{xs:x_data,ys:y_data})
    plt.subplot(122)
    plt.scatter(x_data,y_data)
    plt.plot(x_data,pred_,'r-')
    plt.text(-1,3.5,'reload loss:%.4f' %loss2 , fontdict={'size':10,'color':'red'})
    plt.show()

