"""
全链接神经网络的实现
1.实现手写数字识别
2.展示每一层的权重是如何变化的，他每次全连接到底做了什么
3.做法：
        创建一个三层的神经网络
        输入层784
        隐藏层100
        输出层10
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pandas as pd

#导入mnist数据集
mnist=input_data.read_data_sets('mnist_data',one_hot=True)

"""
切分数据集的方法
"""
def data_split(traindata,labels,size=0.3):
    indexs=np.arange(len(traindata))
    np.random.shuffle(indexs)
    index_split=int(len(traindata)*size)
    x_train,x_test,y_train,y_test=traindata[indexs[:-index_split]],traindata[indexs[-index_split:]],labels[indexs[:-index_split]],labels[indexs[-index_split:]]
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=data_split(mnist.train.images,mnist.train.labels)
print("数据切分完成：x_train.shape:",x_train.shape,"\ty_test.shape:",y_test.shape)

#定义全连接神经网络
tf_x=tf.placeholder(tf.float32,[None,x_train.shape[1]])
tf_y=tf.placeholder(tf.float32,[None,y_train.shape[1]])

# with tf.name_scope("l1"):
l1=tf.layers.dense(tf_x,100,tf.nn.relu)#隐藏层

# with tf.name_scope("prediction"):
prediction=tf.layers.dense(l1,10,tf.nn.softmax)#输出层

acc=tf.metrics.accuracy(tf.argmax(tf_y,1),tf.argmax(prediction,1))

loss=tf.losses.softmax_cross_entropy(tf_y,prediction)
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

#查看训练过程
losses=[]
acces=[]

plt.ion()
plt.figure()

#训练
for i in range(301):
    loss_,train_step_,prediction_,acc_=sess.run([loss,train_step,prediction,acc],{tf_x:x_train,tf_y:y_train})
    
    if i%10==0:

        #获取神经网络中的权重、偏置数据
        variables=tf.trainable_variables()
        # #1.查看100个隐藏层的的权重形状28*28
        # #2.查看输出层的权重10*10
        # l1_weights=sess.run(variables[0].value()).T.reshape(100,28,28)#784*100=>100*784=>100*28*28
        # l2_weights=sess.run(variables[2].value()).T.reshape(10,10,10)#100*10=>10*100=>10*10*10
        # l1_weights=np.array(l1_weights)
        # l2_weights=np.array(l2_weights)

        # #转化为灰度图像
        # l1_weights=np.square(l1_weights-np.min(l1_weights)+1e-4)
        # l1_weights=(l1_weights/np.max(l1_weights)*255).astype(int)

        # l2_weights=np.square(l2_weights-np.min(l2_weights)+1e-4)
        # l2_weights=(l2_weights/np.max(l2_weights)*255).astype(int)

                #1.查看100个隐藏层的的权重形状28*28
        #2.查看输出层的权重10*10
        l1_weights=sess.run(variables[0].value()).T#784*100=>100*784=>100*28*28
        l2_weights=sess.run(variables[2].value()).T#100*10=>10*100=>10*10*10
        l1_weights=np.array(l1_weights)
        l2_weights=np.array(l2_weights)

        #转化为灰度图像
        l1_weights=np.square(l1_weights-np.min(l1_weights,axis=1).reshape(100,1)+1e-4)
        l1_weights=(l1_weights/np.max(l1_weights,axis=1).reshape(100,1)*255).astype(int).reshape(100,28,28)

        l2_weights=np.square(l2_weights-np.min(l2_weights,axis=1).reshape(10,1)+1e-4)
        l2_weights=(l2_weights/np.max(l2_weights,axis=1).reshape(10,1)*255).astype(int).reshape(10,10,10)

        for j in range(l1_weights.shape[0]):
                plt.subplot(7,17,j+1)
                plt.imshow(l1_weights[j],cmap=plt.cm.binary)

        for j in range(l2_weights.shape[0]):
                plt.subplot(7,17,j+l1_weights.shape[0]+1)
                plt.imshow(l2_weights[j],cmap=plt.cm.binary)


        losses.append(loss_)
        acces.append(acc_[1])

        # plt.figure()
        plt.subplot(7,17,l2_weights.shape[0]+l1_weights.shape[0]+1)
        plt.plot(range(len(losses)),losses,color='red')
        plt.subplot(7,17,l2_weights.shape[0]+l1_weights.shape[0]+2)
        plt.plot(range(len(acces)),acces,'g')
        plt.pause(0.1)

print("运行结束")
plt.ioff()
plt.show()


