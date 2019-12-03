#%%
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

classes = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

#--------------- 清洗数据 class  ---------------
class clean_data(object):
    _train = True
    def __init__(self, file_name, train=True):
        self._train = train
        self._train_df = pd.read_csv(file_name)
        #将csv中的feature(str)转为1*n的矩阵
        self._train_df['feature'] = self._train_df['feature'].map(lambda x : np.array(list(map(float , x.split()))))
        self._image_size = (48,48)
        #(n,48,48)
        self._feature = np.array(self._train_df['feature'].map(lambda x : x.reshape(self._image_size)).values.tolist())
        if self._train: #只有训练集有label
            self._label = self._train_df.label.values
            self._onehot = pd.get_dummies(self._train_df.label).values
        
    @property
    def feature(self):
        return self._feature 
    
    if _train:
        @property
        def label(self):
            return self._label
        @property
        def onehot(self):
            return self._onehot
    
#--------------- 画图  ---------------   
def plot_training_history(r):
    # plot some data
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()


def plot_images(images, cls_true, cls_pred=None):
    name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.

        image = images[i].reshape(48, 48)

        # Ensure the noisy pixel-values are between 0 and 1.
        # image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='gray',
                  interpolation='nearest')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True:{0}".format(name[cls_true[i]])
        else:
            xlabel = "True:{0}, Pred:{1}".format(name[cls_true[i]], name[cls_pred[i]])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
      
#%%
import tensorflow as tf

train_data = clean_data('data/train.csv')
test_data = clean_data('data/test.csv', False)

train = train_data.feature.reshape((-1, 48, 48, 1))/255
train_x = train[:-2000]
train_label = train_data.label[:-2000]
train_onehot = train_data.onehot[:-2000]
test_x = train[-2000:]
test_label = train_data.label[-2000:]
test_onehot = train_data.onehot[-2000:]

#%%
#--------------- 定义网络结构 ---------------
xs = tf.placeholder(dtype=tf.float32,shape=[None,48,48,1])    
ys = tf.placeholder(dtype=tf.float32,shape=[None,7])     
keep_prob = tf.placeholder(tf.float32)
tf_is_train = tf.placeholder(tf.bool, None) 
 
images = tf.reshape(xs,[-1,48,48,1]) #修改图片大小

#CNN结构
conv1 = tf.layers.conv2d(inputs=images,#输入
                        filters=64, #输出维数
                        kernel_size=5, #卷积核size
                        strides=1,
                        padding='same', #输入输出一样
                        activation=tf.nn.relu
                        ) # ->(48,48,64)
pool1 = tf.layers.max_pooling2d(conv1,2,strides=2) # ->(24,24,64)
pool1 = tf.layers.batch_normalization(pool1, training=tf_is_train)
pool1 = tf.layers.dropout(pool1, rate=keep_prob, training=tf_is_train) # drop out 50% of inputs

conv2 = tf.layers.conv2d(pool1,128,5,1,'same',activation=tf.nn.relu) # ->(24,24,128)
pool2 = tf.layers.max_pooling2d(conv2,2,strides=2) # ->(12,12,128)
pool2 = tf.layers.batch_normalization(pool2, training=tf_is_train)
pool2 = tf.layers.dropout(pool2, rate=keep_prob, training=tf_is_train) # drop out 50% of inputs

flat = tf.reshape(pool2,[-1,12*12*128])  
fc1 = tf.layers.dense(flat,128,activation=tf.nn.relu)
output = tf.layers.dense(fc1,7,activation=tf.nn.softmax)

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)           # compute cost
train_op = tf.train.AdagradOptimizer(1e-4).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1))[1]


##数据分批次
#def get_Batch(data, label, batch_size):
#    input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32 ) 
#    x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
#    return x_batch, y_batch
#x_batch, y_batch = get_Batch(train_x, train_onehot, 1000)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # 初始化全局、局部变量

def get_batch(data,label,batch_size):
    sample_idxs = np.random.choice(range(len(data)), size=batch_size)
    batch_xs = []
    batch_ys = []
    for j in range(batch_size):
        train_id = sample_idxs[j]
        batch_xs.append(data[train_id])
        batch_ys.append(label[train_id])
    batch_xs = np.array(batch_xs)
    batch_ys = np.array(batch_ys)
    return batch_xs,batch_ys

start = time.time()
for epoch in range(22):
    data,label = get_batch(train_x,train_onehot,100)
    _,train_accuracy = sess.run([train_op,accuracy], feed_dict={xs:data, ys:label, tf_is_train:True, keep_prob:0.5})
    test_accuracy = sess.run(accuracy, feed_dict={xs:test_x, ys:test_onehot, tf_is_train:False, keep_prob:1})
#    test_accuracy=1.0
    print("Epoch %d, train acc %.4f, test acc %.4f" % (epoch, train_accuracy, test_accuracy))
end = time.time()

print('time %.2fs' %(end-start))

#coord = tf.train.Coordinator() # 开启协调器
#threads = tf.train.start_queue_runners(sess,coord) # 使用start_queue_runners 启动队列填充
#
#epoch = 0
#try:
#    while not coord.should_stop():
#        # 获取训练用的每一个batch中batch_size个样本和标签
#        data, label = sess.run([x_batch, y_batch])
#        _,train_accuracy = sess.run([train_op,accuracy], feed_dict={xs: data, ys: label})
#        test_accuracy = sess.run([accuracy], feed_dict={xs: test_x, ys: test_onehot})
#        print("Epoch %d, train acc %g, test acc %g" % (epoch, train_accuracy, test_accuracy))
#        epoch = epoch + 1
#except tf.errors.OutOfRangeError:
#    print('Done training')
#finally:
#    coord.request_stop()
#coord.join(threads)
