#%% iris 鸢尾花 knn
#https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-2-general-pattern/

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#--------------- 导入数据 ---------------
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

#划分数据集，同时打乱
x_train,x_test,y_train,y_test = train_test_split(
                                 iris_x,iris_y,test_size=0.3)

#--------------- 训练 预测 ---------------
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pre = knn.predict(x_test)
print(y_pre)
print(y_test)

#%% 波士顿房价
#数据集 https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data,target = load_boston(return_X_y=True)
#x = load_boston() #返回对象

model = LinearRegression()
model.fit(data,target)
y_pre = model.predict(data)

plt.plot(range(0,100),target[:100],y_pre[:100])

#%% 创建数据集
from sklearn import datasets
#--------------- 回归数据 ---------------
x,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(x,y)

#--------------- 分类数据 ---------------
x,y = datasets.make_classification(n_samples=200,n_features=2,
                                   n_redundant=0, #冗余特征数
                                   n_classes=3, #类别数
                                   n_clusters_per_class=1, #每类的簇
                                   )
plt.scatter(x[:,0], x[:,1], marker='o',
            c=y, #color颜色
            edgecolors='k'#边缘颜色为黑色
            ) 

#--------------- 聚类数据 ---------------
x,y = datasets.make_blobs(n_samples=100,n_features=2,
                          centers=[[-1,-1],[1,1],[2,2]], #簇中心
                          cluster_std = [0.4,0.5,0.2]
                          )
plt.scatter(x[:,0],x[:,1],marker='o',c=y)

#--------------- 分组正态分布 ---------------
x,y = datasets.make_gaussian_quantiles(n_samples=100,
                                       n_features=2, #正态分布维数
                                       n_classes=3,
                                       mean=[1,2], #特征均值
                                       cov=2, #样本协方差系数
                                      )
plt.scatter(x[:,0],x[:,1],marker='o',c=y)

#%% model的常用属性、功能
from sklearn import datasets
from sklearn.linear_model import LinearRegression

load_data = datasets.load_boston()
x = load_data.data
y = load_data.target
model = LinearRegression()
model.fit(x,y)
print(model.coef_) #斜率
print(model.intercept_) #截距
print(model.get_params()) #取回定义的参数
model.score(x,y) #R^2评分

#%% 标准化normalization
from sklearn import preprocessing
import numpy as np

a = np.array([[10,2.7,3.6],
              [-100,5,-2],
              [120,20,40]],dtype=np.float64
             )
print(preprocessing.scale(a)) #标准化函数
preprocessing.minmax_scale(a,feature_range=(0,1)) #最大最小值
print((a - a.mean(axis=0))/a.std(axis=0)) #手动标准化，列为各指标，行为数据

#--------------- 聚类数据 ---------------
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x,y = make_classification(n_samples=300,n_features=2,
                          n_redundant=0,#冗余特征
                          n_informative=2, #2个比较相关
                          n_clusters_per_class=1, #簇
                          random_state=22, #每个都一样
                          scale=100)
plt.scatter(x[:,0],x[:,1],c=y)


x = preprocessing.scale(x) #归一化，注释后正确率很差
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
df = SVC()
df.fit(x_train,y_train)
df.score(x_test,y_test)

#%% 交叉验证
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x = iris.data
y = iris.target

# x_train,x_test,y_train,y_test = \
#         train_test_split(x,y,random_state=3) #随机数种子，用于复现结果       
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train,y_train)
# knn.score(x_test,y_test)

from sklearn.model_selection import cross_val_score
k_scores=[]
for k in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,x,y,cv=5,scoring='accuracy')#分类问题
    #loss = -cross_val_score(knn,x,y,cv=5,scoring='neg_mean_squared_error')#回归模型
    k_scores.append(scores.mean())

#%% 过拟合，learning_curve是随着训练集的增多，train和test的变化
from sklearn.model_selection import learning_curve #学习曲线模块
from sklearn.datasets import load_digits #digits数据集
from sklearn.svm import SVC #Support Vector Classifier
import matplotlib.pyplot as plt #可视化模块
import numpy as np

digits = load_digits()
x = digits.data
y = digits.target
train_sizes, train_loss, test_loss = learning_curve(
    SVC(gamma=0.001), x, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1) #一行一行求
test_loss_mean = -np.mean(test_loss, axis=1)

#training一直很低，但是交叉验证时，testing loss会比较大。
plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

#%% 过拟合，validation_curve是看param的变化，train和test的变化
from sklearn.model_selection import validation_curve #学习曲线模块
from sklearn.datasets import load_digits #digits数据集
from sklearn.svm import SVC #Support Vector Classifier
import matplotlib.pyplot as plt #可视化模块
import numpy as np

digits = load_digits()
x = digits.data
y = digits.target
param_range = np.logspace(-6,-2,5)
train_loss, test_loss = validation_curve(
    SVC(), x, y, param_name='gamma',param_range=param_range,
    cv=10, scoring='neg_mean_squared_error')

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1) #一行一行求
test_loss_mean = -np.mean(test_loss, axis=1)

#training一直很低，但是交叉验证时，testing loss会比较大。
plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()

#%% 保存重载模型
import pickle
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
x, y = iris.data, iris.target
clf.fit(x,y)

#1.pickle
with open('clf.pickle','wb') as f: #存储
    pickle.dump(clf,f)

with open('clf.pickle','rb') as f: #读取并预测
    clf1 = pickle.load(f)
    y_pre = clf1.predict(x[:3])

#2.joblib
from sklearn.externals import joblib
joblib.dump(clf,'clf.pkl') #保存
clf2 = joblib.load('clf.pkl') #加载
y_pre = clf2.predict(x[:3])


