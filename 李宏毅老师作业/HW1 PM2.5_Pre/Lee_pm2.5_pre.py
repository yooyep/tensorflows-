#%%  选取9个pm2.5的特征进行预测第10个pm2.5值
#导入模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
path = r'.\Dataset'

#--------------- 数据处理 ---------------
#读取csv
#18个feature，12月的前20天的24个小时数据
train_data = pd.read_csv(path+r'\train.csv')
#train = train[train['observation']=='PM2.5']

#扔掉无关的列，inplace=True直接在原基础上修改
train_data.drop(['Date','stations'],axis=1,inplace=True) 

column = train_data['observation'].unique()
new_train_data = pd.DataFrame(np.zeros((12*20*24 , 18)),columns=column)
#12个月20天24个小时，18个特征

for i in column:
    #依次取出不同指标的数据 12*20天，24小时。
    train_data1 = train_data[train_data['observation']==i]
    train_data1.drop('observation',axis=1,inplace=True) #240*24
    train_data1 = np.array(train_data1)
    train_data1[train_data1=='NR'] = '0' #去除异常
    train_data1 = train_data1.astype('float')
    train_data1 = train_data1.reshape(1,12*20*24)
    train_data1 = train_data1.T
    
    new_train_data[i] = train_data1 #保存处理后的值
    
#第10个小时的PM
label = np.array(new_train_data['PM2.5'][9:],dtype='float32')  
  

# 探索性数据分析 EDA
# 最简单粗暴的方式就是根据 HeatMap 热力图分析各个指标之间的关联性
#f, ax = plt.subplots(figsize=(9, 6))    
#sns.heatmap(new_train_data.corr(), fmt="d", linewidths=0.5, ax=ax)
#plt.show()

#--------------- 模型建立---线性回归 ---------------
#1.归一化
#2.利用前9小时的PM2.5 预测第十个小时的PM2.5，是
pm = new_train_data['PM2.5']
pm_mean = int(pm.mean())
pm_theta = int(pm.var()**0.5)
pm = (pm-pm_mean)/pm_theta

w = np.random.rand(1,10)
lr = 0.1
m = len(label)


import time 
start=time.time()
for i in range(50): #训练所有数据100次
    loss=0
    i+=1
    gradient=0
    gra_2 = 0
    #5751个数据 更新以此w
    
    for j in range(m):
        x = np.array(pm[j : j+9])
        x = np.insert(x,0,1) #第0个位置插入1，表示b
        y_pre = np.matmul(w,x)
        error = label[j] - y_pre
        loss += error**2
        gradient += -2*error*x #对w一次的更新10*1
        gra_2 += (2*error*x)**2
    loss = loss/(2*m) #平均下
    gra_2 = gra_2/m
    print('iter:%d | loss:%s' %(i,loss))
    w = w - lr*gradient/m #普通的梯度下降
#    w = w - lr*gradient/m/np.sqrt(gra_2) #Adagrad
    
end=time.time()
print('花费时间：{}'.format(end-start))
    
with open('w.pickle','wb') as f:
    pickle.dump(w,f) #保存参数



#%%
#--------------------------  test_data 计算准确性 --------------------------
#计算测试集 的误差  
test_data = pd.read_csv(path+r'\test.csv',header=None) #第一行不取数据
result_data = pd.read_csv(path+r'\result.csv')
y_real = result_data['value']

test_data.drop(0,axis=1,inplace=True)
column = test_data[1].unique()
new_test_data = pd.DataFrame(np.zeros((240*9 , 18)),columns=column)

for i in column:
    test_data1 = test_data[test_data[1]==i]
    test_data1.drop(1,axis=1,inplace=True) #240*9
    test_data1 = np.array(test_data1)
    test_data1[test_data1=='NR'] = '0' #去除异常
    test_data1 = test_data1.astype('float')
    test_data1 = test_data1.reshape(1,240*9)
    test_data1 = test_data1.T
    
    new_test_data[i] = test_data1 #保存处理后的值

#归一化 
pm = new_test_data['PM2.5']
pm_mean = int(pm.mean())
pm_theta = int(pm.var()**0.5)
pm2 = (pm-pm_mean)/pm_theta

with open('w.pickle','rb') as f:
    w_ = pickle.load(f) #读取参数

loss=0
y_pre=[]
for i in range(240):
    x = np.array(pm2[i*9:i*9+9])
    x = np.insert(x,0,1)
    y_pre1 = np.matmul(w_,x)
    
    error = y_real[i] - y_pre1
    y_pre.append(y_pre1)
    loss += error**2
print('loss:',loss)

plt.title('Result Analysis')
plt.plot(range(240), y_real, color='green', label='y_real')
plt.plot(range(240), y_pre, color='red', label='y_pre')
plt.legend() # 显示图例
plt.xlabel('iteration times')
plt.ylabel('pm2.5')
plt.show()



