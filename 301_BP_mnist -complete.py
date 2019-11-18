#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import time

from pylab import mpl
# matplotlib没有中文字体，动态解决
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


#分类问题，识别minist手写字体===============================

#版本1：tensorflow官网的http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
'''
#--------------- 计算正确率 ---------------
#argmax(y,1) 依次比较各行，取各行最大元素 所在的index
def compute_accuracy(v_xs , v_y_correct):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_correct,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={xs: v_xs, y_correct: v_y_correct})
    return result

#导入数据
#tensorflow官方的读取程序 C:\Anaconda3\Lib\site-packages\tensorflow\contrib\learn\python\learn\datasets\mnist.py

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#--------------- 定义网络结构 ---------------
xs = tf.placeholder(tf.float32,[None,784]) #28*28
y_correct = tf.placeholder(tf.float32,[None,10]) #输出0-9即可
#中间只有一层，w b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(xs,W)+b) #softmax激活,预测值

#交叉熵 描述预测的准确性
cross_entropy = -tf.reduce_sum(y_correct*tf.log(y))
#交叉熵最小，建立训练步骤，学习率0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#--------------- 训练 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #初始化所有值

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,y_correct:batch_ys})
    if i%50 == 0:
        print('正确率: %s' %compute_accuracy(mnist.test.images,mnist.test.labels))


'''
#版本2：通过tf.layer.dense()，建立网络
import struct
import collections
#--------------- 读取数据 ---------------
data_dir = r'C:\Users\13115\Desktop\python程序\识别手写字体\MNIST'
train_num=50000
valid_num=10000
test_num=10000

#--------------- 计算正确率 ---------------
#argmax(y,1) 依次比较各行，取各行最大元素 所在的index
def compute_accuracy(v_xs , v_y_correct):
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y_correct,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={xs: v_xs, y_correct: v_y_correct})
    return result

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

#onehot=np.eye(10) #根据labels的值 访问作为
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    #labels_dense.ravel() 变成一维数据,在np.flat迭代所有要变为1的位置。
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

#定义数据集，将image label打包
class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, images,labels):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._index_in_epoch = 0
        self._epochs_completed = 0
    @property
    def images(self):
    	return self._images
    @property
    def labels(self):
    	return self._labels
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        #第一次epoch 打乱数组
        if self._epochs_completed==0 and start==0:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
        #如果超出数组大小
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            #获取剩下不够batch_size的数据，先获取剩下，再打乱，拼接上不够的
            #1.获取剩下的
            rest_num = self._num_examples-start
            image_rest = self._images[start:self._num_examples]
            label_rest = self._labels[start:self._num_examples]
            #2.再打乱
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            #3.拼接上不够的，凑出size
            start=0
            self._index_in_epoch = batch_size-rest_num
            end = self._index_in_epoch
            #补充的部分
            image_bu = self._images[start:end]
            label_bu = self._labels[start:end]
            return np.concatenate((image_rest,image_bu),axis=0) , np.concatenate((label_rest,label_bu),axis=0) 
        
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

#读取数据，data_path数据集位置，返回一个namedtuple
def read_data_sets(data_path,onehot=False):
    train_img_path = data_path + '/train-images.idx3-ubyte'
    test_img_path = data_path + '/t10k-images.idx3-ubyte'
    train_label_path = data_path + '/train-labels.idx1-ubyte'
    test_label_path = data_path + '/t10k-labels.idx1-ubyte'
    with open(train_img_path,'rb') as f:
        print(struct.unpack('>4i',f.read(16))) #magic_num 数据个数 行列等
        img = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255
        train_img = img[:train_num]
        valid_img = img[train_num:]
    with open(test_img_path,'rb') as f:
         print(struct.unpack('>4i',f.read(16)))
         test_img = np.fromfile(f,dtype=np.uint8).reshape(-1,28*28)/255
     
    with open(train_label_path,'rb') as f:
         print(struct.unpack('>2i',f.read(8)))
         label = np.fromfile(f,dtype=np.uint8)
         if onehot:
             train_label = dense_to_one_hot(label[:train_num],10)
             valid_label = dense_to_one_hot(label[train_num:],10)
    with open(test_label_path,'rb') as f:
         print(struct.unpack('>2i',f.read(8)))
         if onehot:
             test_label = np.fromfile(f,dtype=np.uint8)
             test_label = dense_to_one_hot(test_label,10)
    
    train = Dataset(train_img,train_label)
    valid = Dataset(valid_img,valid_label)
    test = Dataset(test_img,test_label)
    #建立一个namedtuple类似结构体，将dataset集合为Datasets
    #创建数据集，包含train valid test
    Datasets = collections.namedtuple('Datasets',['train','valid','test'])
    return Datasets(train=train,valid=valid,test=test)
        
mnist = read_data_sets(data_dir,onehot=True)
##mnist.train.images  mnist.test.labels来访问

#--------------- 定义网络结构 ---------------
#占位符 定义网络的输入
xs = tf.placeholder(tf.float32,[None,28*28])
y_correct = tf.placeholder(tf.float32,[None,10])

##1.输入784，中间一层784，激活函数tf.nn.relu效果好，输出10 softmax
##正确率97 96左右
#add hidden layer 隐含层
l1 = tf.layers.dense(xs,784,activation=tf.nn.relu)
#add output layer 输出层
predict = tf.layers.dense(l1,10,activation=tf.nn.softmax)

#tf.summary.histogram('pred', predict)

##2.中间不添加层  输入直接连接输出--计算速度快，但是正确率87%
#predict = add_layer(xs,784,10,activation_funciton=tf.nn.softmax)
##3.中间不添加层--dense实现，但是正确率91%
#predict = tf.layers.dense(xs,10,activation=tf.nn.softmax)


#the error 计算error
#交叉熵 描述预测的准确性
cross_entropy = -tf.reduce_sum(y_correct*tf.log(predict))
#均方根误差
#cross_entropy = tf.reduce_mean(tf.square(predict-y_correct))
tf.summary.scalar('loss',cross_entropy)
#交叉熵最小，建立训练步骤，学习率0.01
lr=0.1
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(lr,0.9).minimize(cross_entropy)
#train_step = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(lr,0.9,0.999,epsilon=1e-08).minimize(cross_entropy)


#important step 初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train_wirter = tf.summary.FileWriter('log/train',sess.graph)
#merged = tf.summary.merge_all()


list_loss = []
list_accu = []

start=time.time()
#--------------- 训练 ---------------
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    _ , loss_ = sess.run([train_step,cross_entropy],feed_dict={xs:batch_xs , y_correct:batch_ys})
#    list_loss.append(loss_)
#    result = sess.run(merged,feed_dict={xs:batch_xs , y_correct:batch_ys})
    if i%50 == 0:
        accuracy_ = compute_accuracy(mnist.test.images,mnist.test.labels)
        list_loss.append(loss_)
        list_accu.append(accuracy_)
        print('正确率: %s' %accuracy_)
        print('i:%d' %i)
#        train_wirter.add_summary(result, i) #记录i次的result

#for i in range(1000):
#    sess.run(train_step,feed_dict={xs:mnist.train.images , y_correct:mnist.train.labels})
end=time.time()     

#--------------- 可视化结果 ---------------
def view_result():
    img_cont = np.random.randint(test_num)
    img_data = np.atleast_2d(mnist.test.images[img_cont])
    pred = sess.run(predict,feed_dict={xs:img_data})
    plt.figure()
    print('真实值：{} 预测值：{}'.format(mnist.test.labels[img_cont].argmax() , pred.argmax()))
    plt.imshow(img_data.reshape(28,28),cmap='gray')
    plt.title('真实值：{} 预测值：{}'.format(mnist.test.labels[img_cont].argmax() , pred.argmax()))


     
#cls_true真实值label cls_pred预测值标签
def view_results(images, cls_true, cls_pred=None):
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)    
    
    name = np.arange(10)
    for i, ax in enumerate(axes.flat):
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


img_nums = np.random.randint(0,test_num,9) #生成10个0-test_num的随机数，随机选10个数
img_data = np.atleast_2d(mnist.test.images[img_nums]) #10*784,10张图片
#plt.imshow(img_data[1].reshape(28,28),cmap='gray')
img_label = mnist.test.labels[img_nums].argmax(1) #10张图片的onehot按行进行argmax()
pred_label=[]
for row in img_data:
    pred = sess.run(predict,feed_dict={xs:row[np.newaxis,:]})
    pred_label.append(pred.argmax())    
view_results(img_data , img_label , pred_label)   
#view_result()


##设置坐标轴刻度
#my_x_ticks = np.arange(0, 20, 1)
#
#plt.figure()
#plt.plot(list_loss)
#plt.xticks(my_x_ticks)
#plt.xlabel(r'迭代次数')
#plt.ylabel(r'损失函数')
#plt.title('Adam参数优化--系统建模')
#plt.show()
#
#plt.figure()
#plt.plot(list_accu,'*-')
#plt.xticks(my_x_ticks)
#plt.xlabel(r'迭代次数')
#plt.ylabel(r'正确率')
#plt.title('Adam参数优化--系统建模')
#plt.show()



