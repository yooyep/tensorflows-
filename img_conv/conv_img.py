# -*- coding: utf-8 -*-
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

#%%
#写im2col，将卷积的窗口 平铺成block_size*1
#函数默认stride=1
def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1 #列上/竖着 平移次数
    sy = mtx_shape[1] - block_size[1] + 1 #横着平移次数
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='C')
    return result

def conv1(img,weight):
    height,width = img.shape
    h,w = weight.shape
	#卷积后的h,w
    new_h = height-h+1
    new_w = width-w+1
    
    mtx1 = im2col(img,weight.shape)
    res_cov = np.dot(mtx1,weight.ravel('C'))
    res_cov = res_cov.reshape(new_h,new_w)
    #print(mtx1) #竖着排列的
    res_cov = res_cov.clip(0, 255)
    return res_cov

#调用
#mtx = np.arange(9).reshape(3,3)+1
#fil = np.array([[-1,1],[1,1]])
#conv1(mtx,fil)


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


#%%
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
  
#什么都不做
fil = np.array([[0,0,0],
				[0,1,0],
				[0,0,0]]) 
#锐化
fil = np.array([[-1,-1,-1],
				[-1, 9,-1],
				[-1,-1,-1]]) 
#更加锐化
fil = np.array([[-1,-1,-1,-1,-1],
				[-1, 2, 2, 2,-1],
				[-1, 2, 8, 2,-1],
                [-1, 2, 2, 2,-1],
                [-1,-1,-1,-1,-1]])  

#强调边缘
fil = np.array([[ 1, 1, 1],
				[ 1,-7, 1],
				[ 1, 1, 1]])   
    
#边缘有亮度
fil = np.array([[-0,-0,-0,-0,-0],
				[-0, 0, 0, 0,-0],
				[-1,-1, 2, 0,-0],
                [-0, 0, 0, 0,-0],
                [-0,-0,-0,-0,-0]]) 
    
#浮雕效果
fil = np.array([[-1, 0, 0],
				[ 0, 1, 0],
				[ 0, 0, 0]])
    
#均值模糊
fil = np.array([[ 0, 0.2, 0],
				[0.2, 0, 0.2],
				[ 0, 0.2, 0]])
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


