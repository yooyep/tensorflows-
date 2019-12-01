#%% 使用tf对hw2 进行logist回归
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
lr=LogisticRegression()

#--------------- 导入数据 ---------------
path='./data'
train_data = pd.read_csv(path+'/train.csv')
test_data = pd.read_csv(path+'/test.csv')
data = pd.concat([train_data,test_data],axis=0)
data = data.drop(columns=data.columns[0])

#频率统计，其中?也被统计在内
mp=data['native-country'].value_counts()/data.shape[0]
data['native-country']=data['native-country'].map(mp)

##对数值型指标 进行最大最小值归一
#data[['age','fnlwgt' ,'education-num' ,'capital-gain' ,'capital-loss' ,'hours-per-week']] \
#=min_max_scaler.fit_transform(data[['age','fnlwgt' ,'education-num' ,'capital-gain' ,'capital-loss' ,'hours-per-week']])

##通过 np.sum(data=="?",axis=0) 查看数据缺失值在哪列
#data.loc[data["workclass"]=="?" , "workclass"] = "Private"
#data.loc[data["occupation"]=="?" , "occupation"] = "other"

cols=['workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'sex']
#名词性属性 onehot矩阵
def p_data(data,col):
    tmp=pd.get_dummies(data[col],prefix=col) #生成onehot矩阵
    # data=data.join(tmp) 这样不光列拼接进来，行也拼接进来
    data=pd.concat([data,tmp],axis=1) 
    data=data.drop(col,axis=1)
    return data

for col in cols:
    data=p_data(data,col)

label_d = {'<=50K':0,
           '<=50K.':0,
           '>50K':1,
           '>50K.':1}
data.label = data.label.map(label_d)
data.label = data.label.astype(int)

#分离train test
train_data=data[0:train_data.shape[0]]
test_data=data[train_data.shape[0]:]

train_label = train_data['label'][:,np.newaxis]
test_label = test_data['label'][:,np.newaxis]
train = train_data.drop('label',axis=1)
test = test_data.drop('label',axis=1)

#均值化归一
train = (train - train.mean())/train.std() #32561*66
test = (test - test.mean())/test.std() #16281*66


#%%
#--------------- 手写logistic回归  ---------------
train = train.values
test = test.values

#增加全为1的一列，作为参数b
train = np.concatenate((np.ones(shape = (train.shape[0],1)) , train) , axis=1)
test = np.concatenate((np.ones(shape = (test.shape[0],1)) , test) , axis=1)

def sigmoid(z):
    res = 1/(1+np.exp(-z))
    return np.clip(res,1e-8,(1-(1e-8)))

def _shuffle(X,Y):
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    return [X[randomize],Y[randomize]]


def train_lr(X_train,Y_train):
    w = np.random.rand(len(X_train[0]))
    l_rate = 0.001
    batch_size = 32
    list_cost = []
    
    epoch_num = 10
    step_num = int(np.floor(len(X_train)/batch_size))
    
    for epoch in range(1,epoch_num):
        total_loss = 0
        for idx in range(1,step_num):
            X = X_train[idx*batch_size : (idx+1)*batch_size]
            Y = Y_train[idx*batch_size : (idx+1)*batch_size]
        
            z = np.dot(X,w) #n*1
            y = sigmoid(z) 
            loss = y-np.squeeze(Y)
            coss1 = np.dot(np.squeeze(Y.T), np.log(y))
            coss2 = np.dot((1-np.squeeze(Y.T)), np.log(1-y))
            cross_entropy = -1 * (coss1+coss2)/len(Y)
            total_loss+=cross_entropy
            
            grad = np.sum(-1*X*(np.squeeze(Y)-y).reshape((batch_size, 1)), axis=0)
            w = w - l_rate*grad
            
        list_cost.append(total_loss)
        
        result = valid(X_train , Y_train, w)
        train_res = (np.squeeze(Y_train) == result)
        print('epoch:%d acc:%f' % (epoch , float(train_res.sum()) / train_res.shape[0]))

    return w
    
def valid(X,Y,w):
    z = sigmoid(np.dot(X,w))
    y = np.around(z)
    return y

w = train_lr(train,train_label)
y_ = valid(test, test_label, w)
result = (np.squeeze(test_label) == y_)
print('Valid accuracy = %f' % (float(result.sum()) / result.shape[0]))


