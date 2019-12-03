#%% 模型及训练
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
lr=LogisticRegression()

#--------------- 导入数据 ---------------
path='./data'
train_data = pd.read_csv(path+'/train.csv')
test_data = pd.read_csv(path+'/test.csv')
data = pd.concat([train_data,test_data],axis=0)

#频率统计，其中?也被统计在内
mp=data['native-country'].value_counts()/data.shape[0]
data['native-country']=data['native-country'].map(mp)

#对数值型指标 进行最大最小值归一
data[['age','fnlwgt' ,'education-num' ,'capital-gain' ,'capital-loss' ,'hours-per-week']] \
=min_max_scaler.fit_transform(data[['age','fnlwgt' ,'education-num' ,'capital-gain' ,'capital-loss' ,'hours-per-week']])

#通过 np.sum(data=="?",axis=0) 查看数据缺失值在哪列
data.loc[data["workclass"]=="?" , "workclass"] = "Private"
data.loc[data["occupation"]=="?" , "occupation"] = "other"

cols=['workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'sex']
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

train_label = train_data['label']
test_label = test_data['label']
train = train_data.drop('label',axis=1)
test = test_data.drop('label',axis=1)

#--------------- logistic回归模型  ---------------
lr.fit(train,train_label)
lr.score(train,train_label)
lr.score(test,test_label)


#%% 将原始数据保存至csv文件中
path='./data'
#np.genfromtxt 读取不了英文等
#data = np.genfromtxt(path+'/adult.data',delimiter=',',encoding='utf-8')

train_data = []
#这里'r' 不能'rb'。rb读取出来都是字节，之后replace啥的也要b""
with open(path+'/adult.data','r') as f:
    lines = f.readlines()
    for line in lines:
        #line.strip() 返回新字符串
        line = line.strip().replace(" ","") #去掉空格
        train_data.append(line.split(","))

test_data = []
with open(path+'/adult.test','r') as f:
    lines = f.readlines()
    for line in lines:
        #type(line) <class 'bytes'>
        #line.strip() bytes
        line = line.strip().replace(" ","") #去掉空格
        test_data.append(line.split(","))

#列名
cols=['age',
'workclass',
'fnlwgt',
'education',
'education-num',
'marital-status',
'occupation',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'native-country','label']


train_df = pd.DataFrame(train_data,columns=cols)
test_df = pd.DataFrame(test_data,columns=cols)

train_df.to_csv(path+'/train.csv')
test_df.to_csv(path+'/test.csv')



#%% 另一种读取方式
def dataProcess_X(rawData):

    #sex 只有两个属性 先drop之后处理
    if "income" in rawData.columns:
        Data = rawData.drop(["sex", 'income'], axis=1)
    else:
        Data = rawData.drop(["sex"], axis=1)
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn] #数字的column

    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjedtColumn]
    #insert set into nonobject data with male = 0 and female = 1
    NonObjectData.insert(0 ,"sex", (rawData["sex"] == " Female").astype(np.int))
    #set every element in object rows as an attribute
    ObjectData = pd.get_dummies(ObjectData)

    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data_x = Data.astype("int64")
    # Data_y = (rawData["income"] == " <=50K").astype(np.int)

    #normalize
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()

    return Data_x

def dataProcess_Y(rawData):
    df_y = rawData['income']
    Data_y = pd.DataFrame((df_y==' >50K').astype("int64"), columns=["income"])
    return Data_y

