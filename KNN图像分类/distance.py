"""
距离计算方法
"""
import math
import numpy as np

#欧式距离
def L2(x,y):
    a=x-y;
    a=np.square(a)
    a=np.sqrt(a.sum())
    return a


if __name__=='__main__':
    x=np.random.randint(0,5,size=3)
    y=np.random.randint(0,5,size=3)
    dis1=L2(x,y)
    print(x,y,dis1)
