#%%
# -*- coding: utf-8 -*-
#logistics回归模型：https://blog.csdn.net/williamyi96/article/details/52737509
#tensorflow实现二分类：https://blog.csdn.net/he_wen_jie/article/details/80868864
#模型写的很好！ https://blog.csdn.net/wangyangzhizhou/article/details/70474391

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

#%%
#--------------- 导入数据 ---------------
from sklearn import datasets
from sklearn.model_selection import train_test_split

x,y = datasets.make_classification(n_samples=500,n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)  
train_y = train_y[:,np.newaxis]
test_y = test_y[:,np.newaxis] #增加一维，避免tf报错
              
plt.scatter(train_x[:,0],train_x[:,1], marker='o', c=train_y[:,0],
            s=25, edgecolor='k')

#%%
#--------------- 定义网络结构 ---------------
#输入2维，输出1维
xs = tf.placeholder(dtype=tf.float32,shape=[None,2],name='xs')
ys = tf.placeholder(tf.float32,[None,1],name='ys') #labels如果为float 损失函数那里会报错

w = tf.Variable(tf.truncated_normal(shape=[2,1]),name='w')
b = tf.Variable(tf.zeros([1]),name='b')


logits = tf.sigmoid(tf.matmul(xs,w)+b) #logistics函数
predict = tf.round(logits,name='predict') #四舍五入，如果是y0,y1，采用tf.arg_max(logits,1)

##损失函数
#loss = tf.losses.sparse_softmax_cross_entropy(labels=ys,logits=logits)
#loss = tf.reduce_mean(loss)
loss = -tf.reduce_sum(ys*tf.log(logits)+(1-ys)*tf.log(1-logits))
loss = tf.reduce_mean(loss)
tf.summary.scalar('loss', loss)

##正确率
_,acc_op = tf.metrics.accuracy(labels=ys,predictions=predict)
tf.summary.scalar('acc', acc_op)

##训练步骤
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

#--------------- 初始化变量 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
print(sess.run([w,b]))

#--------------- 保存模型 ---------------
saver = tf.train.Saver(max_to_keep=4)

writer = tf.summary.FileWriter('./log', sess.graph)
merge_op = tf.summary.merge_all()
#%%
#--------------- 训练 ---------------
for epoch in range(30000):
    _,loss1,acc1 = sess.run([train_op,loss,acc_op],feed_dict={xs:train_x,ys:train_y})
    if epoch%100 == 0:
        print('i:%d loss:%.2f acc:%.2f' %(epoch,loss1,acc1))
        result = sess.run(merge_op,feed_dict={xs:train_x,ys:train_y})
        writer.add_summary(result,epoch)
        
acc_test = sess.run(acc_op,feed_dict={xs:train_x,ys:train_y})
print('test accuracy:%.2f' %acc_test)
#保存模型
saver.save(sess, 'Model/model')

#%%
#--------------- 画图 ---------------
plt.scatter(train_x[:,0],train_x[:,1], marker='o', c=train_y[:,0],
            s=25, edgecolor='k')

w1,b1 = sess.run([w,b])
xx=np.linspace(-4,4,100)
yy=[]
for i in xx:
    yy.append(float((i* -w1[0]-b1) / w1[1]))
    #yy.append(float((i* -w[1]-b) / w[0]))
plt.plot(xx,yy)
plt.xlim(-4,4)
plt.title('train data')

#测试集图形
plt.figure()           
plt.scatter(test_x[:,0],test_x[:,1], marker='o', c=test_y[:,0],
            s=25, edgecolor='k')
plt.plot(xx,yy)
plt.xlim(-4,4)
plt.title('test data')

#%% 
#--------------- 等高线 可视化 ---------------
#https://blog.csdn.net/he_wen_jie/article/details/80868864
h = 0.02
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.scatter(train_x[:,0],train_x[:,1], marker='o', c=train_y[:,0],
            s=25, edgecolor='k')

#合并xx，yy为[?,2]的点 
#np.c_[xx.ravel(), yy.ravel()]
prob = sess.run(logits,feed_dict={xs:np.c_[xx.ravel(), yy.ravel()]})
prob  = prob.reshape(xx.shape) #平铺成二维
cm = plt.cm.RdBu
plt.contourf(xx, yy, prob,cmap=cm, alpha=.3)


