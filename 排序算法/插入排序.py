"""
插入排序的实现
插入排序的动画实现

插入排序的工作原理是，对于每个未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
"""

import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib import cm
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.00*height, '%s' % float(height))
"""
插入排序,最坏情况n²,平均情况n²
1.认为index=0处已经排序，
2.排第二个元素，如果第二个元素<第一个元素，则互换他们位置
...
3.排第n个元素，前面n-1个元素已经排序，此时问题变更为找n元素合适位置，则我们使用简单排序，n与其他n-1个元素比较，从末向前，每比较一个元素后，排列位置
"""
def insert_sort(arr):
    length=len(arr)
    c = cm.rainbow( np.array(arr) / 25)
    plt.bar(range(0,length),arr,color=c)
    plt.ion()
    plt.show()
    plt.pause(0.2)
    for i in range(1,length):
        for j in range(i,0,-1):
            if arr[j]<arr[j-1]:
                arr[j],arr[j-1]=arr[j-1],arr[j]

                #画图
                plt.cla()
                c = cm.rainbow( np.array(arr) / 25)
                a=plt.bar(range(0,length),arr,color=c)
                autolabel(a)
                plt.pause(2)
    plt.ioff()
    plt.show()
    return arr

if __name__ == "__main__":
    arr=random.sample(range(20),8)
    print('arr:',arr)
    arr=insert_sort(arr)
    print('after sort:',arr)