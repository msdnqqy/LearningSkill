"""
加载mnist数据集
1.加载训练集：load_train(num,datatype,weights)
2.加载测试集：load_test(num,datatype,weights)
3.获取数据集长度：get_data_length()
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class Load:
    def __init__(self):
        #加载mnist数据集
        self.MNIST=input_data.read_data_sets('mnist_data',one_hot=True)
        print('----加载MNIST完成----',self.MNIST.train.images.shape)

    def get_data_length(self,type='train'):
        if type=='test':
            return self.MNIST.test.images.shape[0]
        else :
            return self.MNIST.train.images.shape[0]

    def load_train(self,num=2000,ordertype='order',weights=None):
        return self.__load_data_mnist(num,ordertype,'train',weights)

    def load_test(self,num=2000,ordertype='order',weights=None):
        return self.__load_data_mnist(num,ordertype,'test',weights)

    #根据weights加载num个训练集,type=顺序，随机，None
    def __load_data_mnist(self,num=2000,ordertype='order',datatype='train',weights=None):

        #如果没有权重，也没有加载类型，默认按照顺序加载处理
        if weights is None and ordertype is None :
            if datatype == 'train':
                return self.MNIST.train.next_batch(num)
            else :
                return self.MNIST.test.next_batch(num)

        elif ordertype is not None:
            if ordertype=='order':
                if datatype == 'train':
                    return self.MNIST.train.next_batch(num)
                else:
                    return self.MNIST.test.next_batch(num)
            elif ordertype=='random':
                indexs=np.random.randint(0,self.get_data_length(),size=num)
                if datatype == 'train':
                    return self.MNIST.train.images[indexs],self.MNIST.train.labels[indexs]
                else:
                    return self.MNIST.test.images[indexs],self.MNIST.test.labels[indexs]

        elif weights is not None:
            indexs=np.random.choice(np.arange(0,self.get_data_length()),replace=True,p=weights)
            indexs = np.random.randint(0, self.get_data_length(), size=num)
            if datatype == 'train':
                return self.MNIST.train.images[indexs], self.MNIST.train.labels[indexs]
            else:
                return self.MNIST.test.images[indexs], self.MNIST.test.labels[indexs]

        else:
            return np.array([]),np.array([])




if __name__=='__main__':
    load=Load()
    len=load.get_data_length()
