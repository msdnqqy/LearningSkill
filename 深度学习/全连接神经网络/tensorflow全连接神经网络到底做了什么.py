"""
全链接神经网络的实现
1.实现手写数字识别
2.展示每一层的权重是如何变化的，他每次全连接到底做了什么
3.做法：
        先把神经网络实现为输入层，输出层的简单2层神经网络
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
# l1=tf.layers.dense(tf_x,100,tf.nn.relu)#隐藏层

# with tf.name_scope("prediction"):
prediction=tf.layers.dense(tf_x,10,tf.nn.softmax)#输出层

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
# df=pd.DataFrame([],columns=["l1_weights","l1_biases","prediction_weights","prediction_biases"])
df=pd.DataFrame([],columns=["origin_img","prediction_weights","prediction_biases","prediction_class"])
for i in range(201):
    loss_,train_step_,prediction_,acc_=sess.run([loss,train_step,prediction,acc],{tf_x:x_train,tf_y:y_train})
    
    if i%5==0:

        #获取神经网络中的权重、偏置数据
        variables=tf.trainable_variables()
        # variable_name = [v.name for v in tf.trainable_variables()]
        # print("权重和偏置：",variable_name)
        #输出：：'dense/kernel:0', 'dense/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0']
        
        item=[]
        origin_img=mnist.train.images[0].reshape(28,28)
        prediction_class=sess.run(prediction,{tf_x:[mnist.train.images[0]],tf_y:[mnist.train.labels[0]]})
        item.append(origin_img)
        for variable_item in variables:
                name=variable_item.name
                value=sess.run(variable_item.value())#tensor=>真实值
                item.append(value)

        item.append(prediction_class)
        df.loc[i]=item

        #weights处理：weights*原始图片/max*255
        imgs=[]
        for j in range(10):
                value_np=np.array(item[1].T[j]).reshape(28,28)
                origin_np=np.array(origin_img)
                # img=value_np*origin_np
                img=value_np
                img=(img-np.min(img)+1e-3)/np.max(img)*255
                imgs.append(img.astype(int))

        # plt.cla()
        # plt.figure()
        plt.subplot(4,6,1)
        plt.title("label:{0}--predict:{1},\n item[3]:{2}".format(np.argmax(mnist.train.labels[0]),np.argmax(item[3],axis=1),item[3][0]))
        plt.imshow(item[0],plt.cm.binary)

        img_out=[]
        for j in range(10):
                plt.subplot(4,6,j+2)
                plt.imshow(imgs[j],plt.cm.binary)

                img=imgs[j]*item[0]
                img_out.append(img)

        img_out=np.array(img_out)
        img_out=np.square(img_out-np.min(img_out)+1)
        img_out=img_out/np.max(img_out)*255

        for j in range(len(img_out)):
                plt.subplot(4,6,j+14)
                plt.imshow(img_out[j],plt.cm.binary)

        losses.append(loss_)
        acces.append(acc_[1])

        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(range(len(losses)),losses,color='red')
        # plt.subplot(2,1,2)
        # plt.plot(range(len(acces)),acces,'g')
        plt.pause(0.1)


print(df.head(10))
plt.ioff()
plt.show()


