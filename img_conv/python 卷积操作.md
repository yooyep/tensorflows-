

# python 卷积操作

> 参考链接
>
> https://blog.csdn.net/zouxy09/article/details/49080029
>
> https://zhuanlan.zhihu.com/p/33773140

面试官卷积是怎么算的嘛？

点乘(×)  应该是想讲一下im2col，然后在矩阵相乘；或者cuba实现并行的卷积计算

## im2col的实现

> https://blog.csdn.net/dwyane12138/article/details/78449898

优点：加速卷积计算

假设卷积核的尺寸为2*2，输入图像尺寸为3*3。

将卷积一次要处理的小窗展开成一行或一列(取决于内存的读取，matlab列、caffe行)。

3*3的图像平铺成9*4，filter也平铺成4*1，矩阵相乘x*w，完整一张图片的卷积操作。

![image.png](https://ws1.sinaimg.cn/large/cdbd77ealy1g8wl3qonizj20gy0643z1.jpg)



## filter的填充方式 same full

> https://zhuanlan.zhihu.com/p/62760780

same，filter的中心点与边角重叠，填充filter的尺寸的一半(整除)。卷积后输入输出尺寸一样(步长为1)。如下图

full，filter的右下角与边角重叠，刚开始相交就卷积。

valid，不进行填充，直接卷积

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8wlaj0is8j20fx0eojrl.jpg" width=400/>

## 图片卷积结果

#### 啥也不做

```python
#什么都不做
fil = np.array([[0,0,0],
				[0,1,0],
				[0,0,0]]) 
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8wlcyhyqdj20e705azpp.jpg"/>

#### 锐化滤波

锐化，先找到边缘，再加到原有图像上。计算当前点与周围点的差别，将这个差别加到原有位置上。另外，中间点的权值要比所有的权值和大于1，意味着这个像素要保持原来的值。

```python
#锐化
fil = np.array([[-1,-1,-1],
				[-1, 9,-1],
				[-1,-1,-1]]) 
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8wli7idmwj20e5056grl.jpg"/>

将原有的核放大，更加精细的锐化效果。

```python
fil = np.array([[-1,-1,-1,-1,-1],
				[-1, 2, 2, 2,-1],
				[-1, 2, 8, 2,-1],
                [-1, 2, 2, 2,-1],
                [-1,-1,-1,-1,-1]])  
```

强调边缘信息

```python
fil = np.array([[ 1, 1, 1],
				[ 1,-7, 1],
				[ 1, 1, 1]])   
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8wm7tdghyj20e9050tev.jpg"/>

#### 边缘有亮度

滤波后的图像会很暗(黑色是0)，只有边缘的地方是有亮度的。

```python
fil = np.array([[-0,-0,-0,-0,-0],
				[-0, 0, 0, 0,-0],
				[-1,-1, 2, 0,-0],
                [-0, 0, 0, 0,-0],
                [-0,-0,-0,-0,-0]]) 
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8xhz694chj20ll07n7dv.jpg"/>



#### 浮雕

使用相邻像素值之差来表示当前像素值，从而得到图像的边缘特征，最后加上固定数值150得到浮雕效果

```
fil = np.array([[-1, 0, 0],
				[ 0, 1, 0],
				[ 0, 0, 0]])
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8xh8ylezrj20oi08pwq2.jpg"/>

#### 模糊化

 我们可以将当前像素和它的四邻域的像素一起取平均，然后再除以5，或者直接在滤波器的5个地方取0.2的值即可。

```
fil = np.array([[ 0, 0.2, 0],
				[0.2, 0, 0.2],
				[ 0, 0.2, 0]])
```

<img src="https://ws1.sinaimg.cn/large/cdbd77ealy1g8xi6fogcjj20ld07i7e3.jpg"/>



## 代码实现

#### 读取图像

```python
#%% 
#用于卷积图像的操作
#参考文章:https://blog.csdn.net/zouxy09/article/details/49080029
import matplotlib.pyplot as plt # plt 用于显示图片
from PIL import Image
import numpy as np

#lena = Image.open('test.jpg')
lena = Image.open('test_house.jpg')
lena = np.array(lena)
#lena = plt.imread('test.jpg')  #此时flower是np.array
```

#### 卷积操作1

```python
#%%
#卷积操作实现2 ，直接对应相乘
def conv(image,weight):
    height,width = image.shape
    h,w=weight.shape
    	#卷积后的h,w
    new_h = height-h+1
    new_w = width-w+1
    new_img = np.zeros((new_h,new_w))
    #卷积操作
    for i in range(new_h):
    		for j in range(new_w):
    			new_img[i,j] = np.sum(image[i:i+h,j:j+w] * weight)
    new_img = new_img.clip(0, 255)
    return new_img

#test
#n1 = np.arange(9).reshape(3,3)+1
#fil = np.array([[-1,1],[1,1]])
#n2 = conv(n1,fil)
#print(n2)


#对图像进行卷积
#mode有same和valid，不填写为valid
#ff为
def conv_img(img,fil,mode='same',ff='conv'):
    if mode == 'same':
        h = fil.shape[0] // 2
        w = fil.shape[1] // 2
        img = np.pad(img, ((h, h), (w, w),(0, 0)), 'constant')
        
    conv_r = eval(ff)(img[:,:,0],fil)
    conv_g = eval(ff)(img[:,:,1],fil)
    conv_b = eval(ff)(img[:,:,2],fil)
    
    dstack=np.dstack([conv_r,conv_g,conv_b]).astype('uint8')
    return dstack



#均值模糊
fil = np.array([[1/9,1/9,1/9],
				[1/9,1/9,1/9],
				[1/9,1/9,1/9]])    
    
new_img = conv_img(lena,fil)
#n1 = new_img+100
#new_img = n1.clip(0, 255)

plt.subplot(1,2,1)
plt.imshow(lena)
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(new_img)
plt.axis('off')
plt.show()

```

## 附件

