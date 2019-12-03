#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

classes = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# ### 测试读取数据

# In[ ]:


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
    


# In[ ]:


filename='data/train1.csv'
train = clean_data(filename,True)


# ## utlity.py 读取数据 画图

# In[ ]:


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


# In[ ]:


class clean_data(object):
    _train = True

    def __init__(self, filename, train=True):
        self._train = train
        self._train_df = pd.read_csv(filename)
        self._train_df['feature'] = self._train_df['feature'].map(lambda x: np.array(list(map(float, x.split()))))
        self._image_size = self._train_df.feature[0].size
        self._image_shape = (int(np.sqrt(self._image_size)), int(np.sqrt(self._image_size)))
        self._dataNum = self._train_df.size
        self._feature = np.array(self._train_df.feature.map(lambda x: x.reshape(self._image_shape)).values.tolist())
        if self._train:
            self._label = self._train_df.label.values
            self._labelNum = self._train_df['label'].unique().size
            self._onehot = pd.get_dummies(self._train_df.label).values

    @property
    def distribution(self):
        return self._distribution

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def dataNum(self):
        return self._dataNum

    @property
    def feature(self):
        return self._feature

    if _train:
        @property
        def label(self):
            return self._label

        @property
        def labelNum(self):
            return self._labelNum

        @property
        def onehot(self):
            return self._onehot


# ## 数据处理

# In[ ]:


# import tensorflow as tf
# from tensorflow import keras

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import InputLayer, Input
# from keras.layers import Reshape, MaxPooling2D
# from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import TensorBoard
# from keras.models import Model

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model

from sklearn.utils.class_weight import compute_class_weight


# In[ ]:


train_data = clean_data('data/train.csv')
test_data = clean_data('data/test.csv', False)

train = train_data.feature.reshape((-1, 48, 48, 1))/255
train_x = train[:-2000]
train_label = train_data.label[:-2000]
train_onehot = train_data.onehot[:-2000]
test_x = train[-2000:]
test_label = train_data.label[-2000:]
test_onehot = train_data.onehot[-2000:]


class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(train_data.label),
                                    y=train_data.label)


# In[ ]:





# In[ ]:





# ## CNN模型

# In[ ]:


#CNN model

inputs = Input(shape=(48,48,1))

# First convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
             activation='relu', name='layer_conv1')(inputs)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = BatchNormalization(axis = -1)(net)
net = Dropout(0.25)(net)

# Second convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=128, padding='same',
             activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = BatchNormalization(axis = -1)(net)
net = Dropout(0.25)(net)

# Third convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=256, padding='same',
             activation='relu', name='layer_conv3')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = BatchNormalization(axis = -1)(net)
net = Dropout(0.5)(net)

# Flatten the output of the conv-layer from 4-dim to 2-dim.
net = Flatten()(net)

# First fully-connected / dense layer with ReLU-activation.
net = Dense(128)(net)
net = BatchNormalization(axis = -1)(net)
net = Activation('relu')(net)

# Last fully-connected / dense layer with softmax-activation
# so it can be used for classification.
net = Dense(7)(net)
net = BatchNormalization(axis = -1)(net)
net = Activation('softmax')(net)
# Output of the Neural Network.
outputs = net

model = Model(inputs=inputs, outputs=outputs)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.compile(optimizer='Adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])
y = model.fit(x=train_x[:2000],
           y=train_onehot[:2000],
           validation_data=(test_x, test_onehot),
           class_weight=class_weight,
           epochs=100, batch_size=64,
#           callbacks=[tensorboard]
             )

# plot_training_history(y)
#
model.save('cnn1.h5')


# In[ ]:


# #%% 
# #DNN model

# inputs = Input(shape=(48,48,1))

# dnn = Flatten()(inputs)

# dnn = Dense(512)(dnn)
# dnn = BatchNormalization(axis = -1)(dnn)
# dnn = Activation('relu')(dnn)
# dnn = Dropout(0.25)(dnn)

# dnn = Dense(1024)(dnn)
# dnn = BatchNormalization(axis = -1)(dnn)
# dnn = Activation('relu')(dnn)
# dnn = Dropout(0.5)(dnn)

# dnn = Dense(512)(dnn)
# dnn = BatchNormalization(axis = -1)(dnn)
# dnn = Activation('relu')(dnn)
# dnn = Dropout(0.5)(dnn)

# dnn = Dense(7)(dnn)
# dnn = BatchNormalization(axis = -1)(dnn)
# dnn = Activation('softmax')(dnn)

# outputs = dnn

# model2 = Model(inputs=inputs, outputs=outputs)
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# model2.compile(optimizer='Adam',
#                loss='categorical_crossentropy',
#                metrics=['accuracy'])

# d = model2.fit(x=train_x,
#            y=train_onehot,
#            validation_data=(test_x, test_onehot),
#            class_weight=class_weight,
#            epochs=100, batch_size=64,
#            callbacks=[tensorboard]
#              )

# plot_training_history(d)
# #model2.save('dnn.h5')


# ### cnn模型 展示效果

# In[ ]:


from tensorflow.python.keras.models import load_model
cnn = load_model('cnn2.h5')
# dnn = load_model('dnn2.h5')
cnn_predict = cnn.predict(test_x) #(2000, 7)
cnn_cls = np.argmax(cnn_predict, axis=1) #预测标签值


# In[ ]:


#取出测试集中的真实的label
a=pd.Series(test_label) 
sample_images = [list(a[a == x].index)[0:9] for x in range(7)]
sample_images


# In[ ]:


for x in sample_images:
    plot_images(test_x[x], test_label[x], cnn_cls)


# ### 训练集查看正确率

# In[ ]:


from tensorflow.python.keras.models import load_model
cnn = load_model('cnn2.h5')
# dnn = load_model('dnn2.h5')
cnn_predict = cnn.predict(train_x) #(2000, 7)
cnn_cls = np.argmax(cnn_predict, axis=1) #预测标签值


# In[ ]:


#取出测试集中的真实的label
a=pd.Series(train_label) 
sample_images = [list(a[a == x].index)[0:9] for x in range(7)]
sample_images


# In[ ]:


for x in sample_images:
    plot_images(train_x[x], train_label[x], cnn_cls)


# In[ ]:





# In[ ]:




