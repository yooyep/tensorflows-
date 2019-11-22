## 链接
> 问题描述(PPT)：https://docs.google.com/presentation/d/1r2u-xVytctdRSbaCAHwWlHIBkmJ50Stnpj1hqi9pFXs/edit#slide=id.p3
>
> 代码连接： https://github.com/maplezzz/NTU_ML2017_Hung-yi-Lee_HW/tree/master/HW1

## 数据集
### 训练集 ./Dataset/train.csv
18个feature，每个月前20天 24个小时的数据，总共行数=12*20*18，列数24。
### 测试集
已有连续9小时的数据(18个feature)，预测第10小时，共预测240笔数据。
### 数据处理
去除不需要的数值后，进行标准归一化/最大最小值归一化。
列为指标变量名(x1 x2)，行为每小时的数据(12*20*24)。



## **选取9个pm2.5的特征进行预测**

9小时的pm2.5作为x1-x9，预测的第10小时

总数据12*20*24=5760，去除一开始的9个，训练5751次。

普通gradient decent方法，一开始很快，慢慢的变慢。

Adagrad方法，下降幅度没gradient大，但也能稳定在较低的loss上。

**Adagrad方法注意：学习率可以设置为1**（自己code的话），因为要除以一个数，变化幅度会小。

**结果**：(两种方法相差不大)

```python
iter:50 | loss:[21.77885746]
花费时间：69.24343585968018
loss:% [1844.9286088]
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g96u3foa7bj20hk0d0766.jpg" width=400/ >

## **选取162个特征，进行预测---效果没9特征好、、**

即选取9小时18个feature作为x1-x162，预测第10小时pm2.5

### **一开始：采用效果很差 （学习率大小，导致loss漂了）**

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g96u65vynuj20gn0ccq4t.jpg" width=400/>

### **修改学习率0.005 普通GD**

**效果没Adagrad快**。但是在测试集上的效果**优于**Adagrad，但参数更新速度在这里比Adagrad慢。

```
iter:20 | loss:[201.67385072]
……
iter:200 | loss:[30.757839]
花费时间：60.049662590026855
loss:% [7220.35658241]
```



### **使用Adagrad更新，学习率从1->0.5**

（学习率1，loss后期在震动；0.5会平稳些）

```
iter:20 | loss:[122.99833381]
……
iter:200 | loss:[19.42781484]  #增加训练次数，效果不明显
花费时间：68.32394790649414
loss:% [8406.52617828]
```

**效果图：**

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g96u8c8riyj20gh0caq4r.jpg" width=400/>

## 正则化

添加正则项，进行过拟合限制。正则项系数初始值应该设置为多少，好像也没有一个比较好的准则。建议一开始将正则项系数λ设置为0，先确定一个比较好的learning rate。然后固定该learning rate，给λ一个值（比如1.0），然后根据validation accuracy，将λ增大或者减小10倍（增减10倍是粗调节，当你确定了λ的合适的数量级后，比如λ = 0.01,再进一步地细调节，比如调节为0.02，0.03，0.009之类。）

> https://www.cnblogs.com/HL-space/p/10676637.html

```
lamda=0.1 0.001时，没效果
lamda=1时，loss:% [6643.93824649] 不明显
lamda=10时，loss:% [7120.235234221] #效果不大
```



## 调试总结

普通gradient decent方法，学习率通常先用小的1e-3，加快速度可慢慢调大。

Adagrad方法，学习率可稍微偏大，0.1-1之间。两者更新的速度、效果因问题而异。

正则化，效果不明显。。