"""
冒泡排序原理(默认从小到大)：
1.从前到后，相邻的两个元素相比较，如果arr[i]>arr[i+1]，则两个元素交换位置，直到len-巡回次数处停止 ，记为第一次巡回。此时最大的元素已经被选出排在了最后。
2.循环执行过程1，直到没有元素发生交换
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.00*height, '%s' % float(height))

def bubble_sort(arr):

    #画图
    a=plt.bar(range(len(arr)),arr,color=cm.rainbow(np.array(arr)/25))
    autolabel(a)
    plt.ion()
    plt.show()
    plt.pause(0.2)

    length=len(arr)
    right=length
    need_sort=True
    count=0
    while right>=2 and need_sort:
        count+=1
        need_sort=False
        for j in range(1,right):
            if arr[j-1]>arr[j]:
                arr[j-1],arr[j]=arr[j],arr[j-1]
                right=j
                need_sort=True

                #画图
                plt.cla()
                a=plt.bar(range(len(arr)),arr,color=cm.rainbow(np.array(arr)/25))
                autolabel(a)
                plt.pause(1.5)

    print("一共循环了：{0}次".format(count))
    return arr

if __name__ == "__main__":
    arr=random.sample(range(20),9)
    print("arr:",arr) 
    arr=bubble_sort(arr)
    print("after sort:",arr)   
        