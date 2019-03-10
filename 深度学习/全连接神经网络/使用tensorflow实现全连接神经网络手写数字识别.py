"""
全链接神经网络的实现
1.实现手写数字识别
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

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

l1=tf.layers.dense(tf_x,200,tf.nn.relu)
l2=tf.layers.dense(l1,100,tf.nn.relu)
prediction=tf.layers.dense(l2,10,tf.nn.softmax)

acc=tf.metrics.accuracy(tf.argmax(tf_y,1),tf.argmax(prediction,1))

loss=tf.losses.softmax_cross_entropy(tf_y,prediction)
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

#查看训练过程
losses=[]
acces=[]
acces_test=[]
acces_r=[]
plt.ion()

#训练
for i in range(301):
    loss_,train_step_,prediction_,acc_=sess.run([loss,train_step,prediction,acc],{tf_x:x_train,tf_y:y_train})
    
    if i%10==0:
        losses.append(loss_)
        acces.append(acc_[1])
        acc_test=sess.run(acc,{tf_x:x_test,tf_y:y_test})
        acces_test.append(acc_test[1])

        #测试验证集的精度
        acc_r=sess.run(acc,{tf_x:mnist.test.images,tf_y:mnist.test.labels})
        acces_r.append(acc_r[1])

        plt.cla()
        plt.subplot(2,1,1)
        plt.plot(range(len(losses)),losses,color='red')
        plt.subplot(2,1,2)
        plt.plot(range(len(acces)),acces,'g')
        # print(acc_,acc_test)
        plt.plot(range(len(acces_test)),acces_test,'r--')
        plt.plot(range(len(acces_test)),acces_r,'b-.')
        plt.pause(0.1)

plt.ioff()
plt.show()


print("验证集的精度为：",acc_r[1])