#%%
import numpy as np
import pandas as pd
data_train = pd.read_csv('train.tsv',sep='\t') #tsv的读取采用分隔符为\t
data_test = pd.read_csv('test.tsv',sep='\t')


train_sentences = data_train['Phrase']
test_sentences = data_test['Phrase']
sentences = pd.concat([train_sentences,test_sentences])

label = data_train['Sentiment']
stop_words = open('stop_words.txt',encoding='utf-8').read().splitlines()

# 用sklearn库中的CountVectorizer构建词袋模型
# 词袋模型的详细介绍请看子豪兄的视频
# analyzer='word'指的是以词为单位进行分析，对于拉丁语系语言，有时需要以字母'character'为单位进行分析
# ngram指分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
# max_features指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词

from sklearn.feature_extraction.text import CountVectorizer
co = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4), #1个词-4个词 都提取出来
    stop_words=stop_words,
    max_features=150000
)
# 使用语料库，构建词袋模型
co.fit(sentences)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_sentences, label, test_size=0.3, random_state=1234)

# 用上面构建的词袋模型，把训练集和验证集中的每一个词都进行特征工程，变成向量
x_train = co.transform(x_train)
x_test = co.transform(x_test)


#%%
import tensorflow as tf

train = x_train.toarray()
train = np.array(x_train)
test = x_test
train_label = y_train.values
train_label = train_label[:,np.newaxis]

#--------------- 定义网络结构 ---------------
output_dim=1
xs = tf.placeholder(dtype=tf.float32,shape=[None,train.shape[1]],name='xs')
ys = tf.placeholder(tf.float32,[None,output_dim],name='ys') #0-1分类问题
y = tf.layers.dense(xs,output_dim,activation=None)

loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=ys)
loss = tf.reduce_mean(loss)
##正确率
_,acc_op = tf.metrics.accuracy(labels=ys , predictions=y)
##训练步骤
train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)
#--------------- 初始化变量 ---------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for epoch in range(100):
    _,loss1,acc1 = sess.run([train_op,loss,acc_op],feed_dict={xs:train, ys:train_label})
    if epoch%10 == 0:
        print('i:%d loss:%.2f acc:%.2f' %(epoch,loss1,acc1))
        #result = sess.run(merge_op,feed_dict={xs:train_x,ys:train_y})
        #writer.add_summary(result,epoch)
        
acc_test = sess.run(acc_op,feed_dict={xs:train,ys:train_label})
print('test accuracy:%.2f' %acc_test)



