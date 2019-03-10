"""
快速排序：
平均时间复杂度nLog(n)
最差n²
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.00*height, '%s' % float(height))

def quick_sort(arr,left,right):
    # print('left:',left,"\tright:",right)

    if left==right:
        return
    i=left
    j=right
    while i!=j:
        while j>i and arr[j]>arr[left]:
            j-=1
        while i<j and arr[i]<arr[left]:
            i+=1
        
        #交换i，j位置
        arr[i],arr[j]=arr[j],arr[i]
        #画图
        plt.cla()
        a=plt.bar(range(len(arr)),arr,color=cm.rainbow(np.array(arr)/25))
        autolabel(a)
        plt.pause(1.5)

    #交换基准和j的位置
    arr[left],arr[j]=arr[j],arr[left]
    quick_sort(arr,left,i)
    quick_sort(arr,j+1,right)


if __name__ == "__main__":
    arr=random.sample(range(20),9)
    print("arr:",arr)
    quick_sort(arr,0,len(arr)-1)
    print("after sort:",arr)