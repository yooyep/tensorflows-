#%% 使用tf对hw2 进行logist回归
import numpy as np
import pandas as pd
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

#通过 np.sum(data=="?",axis=0) 查看数据缺失值在哪列
data.loc[data["workclass"]=="?" , "workclass"] = "Private"
data.loc[data["occupation"]=="?" , "occupation"] = "other"

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


#%% logistic回归模型
#--------------- 定义网络结构 ---------------
#输入2维，输出1维
xs = tf.placeholder(dtype=tf.float32,shape=[None,train.shape[1]],name='xs')
ys = tf.placeholder(tf.float32,[None,1],name='ys') #0-1分类问题

##第一种 dense方式
#y = tf.layers.dense(xs,1,activation=tf.nn.sigmoid)

##第二种 变量法师
w = tf.Variable(tf.truncated_normal(shape=[train.shape[1],1]),name='w')
b = tf.Variable(tf.zeros([1]),name='b')
y = tf.nn.sigmoid(tf.matmul(xs,w)+b) #logistics函数

##损失函数
loss = tf.reduce_mean(tf.square(ys-y)) #采用均方差函数
##loss = -tf.reduce_mean(ys*tf.log(predict)) #交叉熵训练不起来
tf.summary.scalar('loss', loss)

##正确率
predict = y>=0.5  #四舍五入，如果是y0,y1，采用tf.arg_max(logits,1)
_,acc_op = tf.metrics.accuracy(labels=ys,predictions=predict)
tf.summary.scalar('acc', acc_op)

##训练步骤
train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

#--------------- 初始化变量 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#--------------- 保存模型 ---------------
#saver = tf.train.Saver(max_to_keep=4)

#%%
#--------------- 训练 ---------------
#a1 = sess.run([logits],feed_dict={xs:train,ys:train_label})

for epoch in range(100):
    _,loss1,acc1 = sess.run([train_op,loss,acc_op],feed_dict={xs:train,ys:train_label})
    if epoch%10 == 0:
        print('i:%d loss:%.2f acc:%.2f' %(epoch,loss1,acc1))
        #result = sess.run(merge_op,feed_dict={xs:train_x,ys:train_y})
        #writer.add_summary(result,epoch)
        
acc_test = sess.run(acc_op,feed_dict={xs:train,ys:train_label})
print('test accuracy:%.2f' %acc_test)
#保存模型
#saver.save(sess, 'Model/model')
