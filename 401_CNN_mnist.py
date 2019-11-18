# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:18:23 2019

@author: 13115
"""
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#%%
#--------------- 导入数据 ---------------
mnist = input_data.read_data_sets("mnist/", one_hot=True)

#%%
#--------------- 定义网络结构 ---------------
batch_size=100
xs = tf.placeholder(dtype=tf.float32,shape=[None,784])
ys = tf.placeholder(tf.float32,[None,10])
images = tf.reshape(xs , [-1,28,28,1]) #修改为batch*图片大小

#CNN结构
conv1 = tf.layers.conv2d(
            inputs = images, #输入
            filters = 16, #输出的维数
            kernel_size = 5, #卷积核的size
            strides = 1,
            padding = 'same', #same filter中心点对准卷积，卷积后输入输出尺寸一样
                              #valid 不填充，直接卷积。
            activation=tf.nn.relu,
            ) # ->(28,28,16)

pool1 = tf.layers.max_pooling2d(
            inputs = conv1,
            pool_size = 2,
            strides=2,
            ) # ->(14,14,16)
conv2 = tf.layers.conv2d(pool1,32,5,1,'same',
                         activation=tf.nn.relu) # ->(14,14,32)
pool2 = tf.layers.max_pooling2d(conv2,2,2,) # ->(7,7,32)

flat = tf.reshape(pool2,[-1,7*7*32]) #第一个为batch
predict = tf.layers.dense(flat,10) ##这里不需要softmax，下面loss有

##损失函数
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys,logits=predict)

##训练步骤
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#--------------- 初始化变量 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#%%
#--------------- 训练 ---------------
list_loss = []
list_accu = []

#argmax(y,1) 依次比较各行，取各行最大元素 所在的index
def compute_accuracy(v_xs , v_y_correct):
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_y_correct})
    return result

start = time.time()
for i in range(500):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    _,loss_ = sess.run([train_step,loss],feed_dict={xs:batch_xs,ys:batch_ys})
    if i%100 == 0:
        accuracy_ = compute_accuracy(mnist.test.images,mnist.test.labels)
        list_loss.append(loss_)
        list_accu.append(accuracy_)
        print('Step:', i, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
end = time.time()
print('time: {}'.format(end-start))

#%%
#--------------- 可视化结果 ---------------
import matplotlib.pyplot as plt 
from pylab import mpl
# matplotlib没有中文字体，动态解决
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def view_result():
    img_cont = np.random.randint(mnist.test.images.shape[0])
    img_data = np.atleast_2d(mnist.test.images[img_cont])
    pred = sess.run(predict , feed_dict={xs:img_data})
    
    print('真实值：{} 预测值：{}'.format(mnist.test.labels[img_cont].argmax() , pred.argmax()))
    plt.figure()
    plt.imshow(img_data.reshape(28,28),cmap='gray')
    plt.title('真实值：{} 预测值：{}'.format(mnist.test.labels[img_cont].argmax() , pred.argmax()))
    plt.show()

view_result()

#%%
#--------------- 可视化结果 九宫图效果 ---------------
#images为9张图片,[9,28*28] ；cls_true真实label ；cls_pred预测label
def view_results(images, cls_true, cls_pred=None):
    fig, axes = plt.subplots(3, 3) #生成3*3的图片
    fig.subplots_adjust(hspace=0.3, wspace=0.3)    
    
    name = np.arange(10)
    for i, ax in enumerate(axes.flat): #对3*3进行迭代
        # Get the i'th image and reshape the array.
        #image = images[i].reshape(img_shape)
        
        # Ensure the noisy pixel-values are between 0 and 1.
        #image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(images[i].reshape(28,28),
                  cmap = 'gray',
                  interpolation='nearest')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(name[cls_true[i]])
        else:
            xlabel = "True: {0}, Pred: {1}".format(name[cls_true[i]], name[cls_pred[i]])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

img_nums = np.random.randint(0,10000,9) #生成10个0-test_num的随机数，随机选10个数
img_data = np.atleast_2d(mnist.test.images[img_nums]) #9*784,9张图片
img_label = mnist.test.labels[img_nums].argmax(1) #10张图片的onehot按行进行argmax()

pred_label=[] #保存预测值label
for row in img_data:
    pred = sess.run(predict,feed_dict={xs:row[np.newaxis,:]})
    pred_label.append(pred.argmax())    
    
view_results(img_data , img_label , pred_label) #图片、真实label、预测label

##设置坐标轴刻度
#my_x_ticks = np.arange(0, 20, 1)
#plt.figure()
#plt.plot(list_accu,'*-')
#plt.xticks(my_x_ticks)
#plt.xlabel(r'迭代次数')
#plt.ylabel(r'正确率')
#plt.title('Adam参数优化--系统建模')
#plt.show()

