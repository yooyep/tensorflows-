# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# matplotlib没有中文字体，动态解决
from pylab import mpl
plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#%%
#李宏毅老师讲完regression的demo，展示w b参数的回归问题
#https://www.bilibili.com/video/av10590361?p=4

#--------------- 数据 及展示 ---------------
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

##显示数据
#plt.plot(x_data,y_data,'*')
##最佳曲线
#xx=np.linspace(0,600,600,dtype=np.float32)
#w_star=2.67
#b_star=-188.4
#yy=w_star*xx+b_star
#plt.plot(xx,yy,'-r',label='最佳')
#plt.legend()
#plt.show()

#%%
#--------------- 画loss function的等高线图 ---------------
x = np.arange(-200,-100,1) #b的参数
y = np.arange(-5,5,0.1) #w的参数
Z = np.zeros([len(x),len(y)])
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b=x[i]
        w=y[j]
        # meshgrid吐出结果：行为w即y，列为x即b。对于同等来说，b变化对loss的影响小
        Z[j][i]=np.mean(np.square(y_data - (w*x_data+b)))

#%%
#--------------- 训练 ---------------
w = -4
b = -120
iteration = 140000
w_history = [w]
b_history = [b]

lr = 1.5
#Adagrad的参数更新
lr_b = 0
lr_w = 0 

import time
start_time = time.time()
for i in range(iteration):
    m=len(x_data)
    y_hat = (w*x_data+b)
    grad_b = -2*np.sum(y_data - y_hat)/m
    grad_w = -2*np.sum(np.dot(y_data - y_hat,x_data))/m
    
    #Adagrad的参数更新
    lr_b = lr_b + grad_b**2
    lr_w = lr_b + grad_w**2
    
    b = b - lr*grad_b/np.sqrt(lr_b)
    w = w - lr*grad_w/np.sqrt(lr_w)
    
    b_history.append(b)
    w_history.append(w)
    loss = np.sum(np.square(y_data - (w*x_data+b)))
    if i%1000==0:
        print('step:{} | w:{:.4f} | b:{:.4f} | loss:{:.2f}'.format(i,w,b,loss))
        print('step:%d, w:%.4f, b:%.4f, loss:%.2f' %(i,w,b,loss))
end_time = time.time()
print("大约需要时间：",end_time-start_time)


#%%
# plot the figure
contour = plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))  # 填充等高线
plt.clabel(contour,fontsize=10,colors='k')

plt.plot([-188.4], [2.67], 'x', ms=12, mew=3, color="orange")
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$')
plt.ylabel(r'$w$')
plt.title("线性回归 学习率=%s" %lr)
plt.show()


